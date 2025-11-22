import os
import torch
import torchaudio
from abc import ABC, abstractmethod
import torchaudio.transforms as T
from pesq import pesq
import numpy as np
from pystoi import stoi
from torch import nn
from utils.data_utils import mel_spectrogram
from utils.class_registry import ClassRegistry
from collections import OrderedDict
from models.metric_models import Wav2Vec2MOS, UTMOSV2, WV_MOS, DNSMOSPredictor
import wvmos
from tqdm import tqdm
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from .Mos_Evaluation.dnsmos import DNSMOS_local

metrics_registry = ClassRegistry()

class ResampleMetric(ABC):
    def __init__(self, config, target_sr):
        self.target_sr = target_sr
        self.device = config.exp.device
        sr = config.mel.out_sr if 'out_sr' in config.mel else config.mel.in_sr
        self.resampler = T.Resample(
                                orig_freq=sr,
                                new_freq=self.target_sr,
                                resampling_method="sinc_interp_kaiser").to(self.device)
        
    def resample(self, wav):
        return self.resampler(wav.unsqueeze(0)).squeeze(0)
    
    @abstractmethod
    def __call__(self, real_batch, gen_batch) -> float:
        pass

@metrics_registry.add_to_registry(name='l1_mel_diff')
class L1MelDiff:
    def __init__(self, config):
        self.n_fft = config.mel.n_fft
        self.num_mels = config.mel.num_mels
        self.sampling_rate = config.mel.out_sr if 'out_sr' in config.mel else config.mel.in_sr
        self.hop_size = config.mel.hop_size
        self.win_size = config.mel.win_size
        self.fmin = config.mel.fmin
        self.fmax = config.mel.fmax_for_loss
        self.device = config.exp.device
        self.l1_loss = nn.L1Loss()

    def __call__(self, real_batch, gen_batch) -> float:
        real_mel = real_batch['mel_for_loss'].to(self.device)
        gen_mel = mel_spectrogram(gen_batch['gen_wav'], self.n_fft, self.num_mels,
                                    self.sampling_rate, self.hop_size, self.win_size,
                                    self.fmin, self.fmax, center=False)
        score = self.l1_loss(real_mel, gen_mel).item()
        return float(score)

@metrics_registry.add_to_registry(name='pesq_base')
class PesqBase(ResampleMetric):
    def __init__(self, config, mode):
        super().__init__(config, 16000)
        self.mode = mode

    def __call__(self, real_batch, gen_batch) -> float:
        scores = []
        for real_wav, gen_wav in zip(real_batch['wav'], gen_batch['gen_wav']):
            real_wav_resampled = self.resample(real_wav.to(self.device))
            gen_wav_resampled = self.resample(gen_wav.to(self.device))

            real_wav_np = real_wav_resampled.cpu().numpy()
            gen_wav_np = gen_wav_resampled.cpu().numpy()

            try:
                pesq_score = self.sliding_pesq(real_wav_np, gen_wav_np, self.target_sr)
            except Exception as e:
                tqdm.write(f'Something went wrong in wb_pesq metric: {e}')
                pesq_score = 1.0

            scores.append(pesq_score)

        return float(np.mean(scores).item())

    def sliding_pesq(self, ref, deg, sr, chunk_sec=20.0, stride_sec=5.0):
        chunk_len = int(sr * chunk_sec)
        stride_len = int(sr * stride_sec)
        total_len = min(len(ref), len(deg))

        if total_len < chunk_len:
            try:
                return pesq(sr, ref[:chunk_len], deg[:chunk_len], self.mode)
            except:
                return 1.0

        scores = []
        for start in range(0, total_len - chunk_len + 1, stride_len):
            ref_chunk = ref[start:start + chunk_len]
            deg_chunk = deg[start:start + chunk_len]

            if len(ref_chunk) < int(chunk_len * 0.8):
                break

            try:
                score = pesq(sr, ref_chunk, deg_chunk, self.mode)
                scores.append(score)
            except Exception as e:
                tqdm.write(f'PESQ error on chunk {start}:{start+chunk_len}: {e}')
                continue

        return float(np.mean(scores)) if scores else 1.0

@metrics_registry.add_to_registry(name='wb_pesq')
class WbPesq(PesqBase):
    def __init__(self, config):
        super().__init__(config, mode='wb')

@metrics_registry.add_to_registry(name='nb_pesq')
class NbPesq(PesqBase):
    def __init__(self, config):
        super().__init__(config, mode='nb')
    
@metrics_registry.add_to_registry(name='stoi')
class STOI:
    def __init__(self, config):
        self.sr = config.mel.out_sr if 'out_sr' in config.mel else config.mel.in_sr
        self.device = config.exp.device

    def __call__(self, real_batch, gen_batch) -> float:
        scores = []
        for real_wav, gen_wav in zip(real_batch['wav'], gen_batch['gen_wav']):
            real_wav_np = real_wav.cpu().numpy()
            gen_wav_np = gen_wav.cpu().numpy()

            stoi_score = stoi(real_wav_np, gen_wav_np, self.sr)
            scores.append(stoi_score)

        return float(np.mean(scores).item())

# @metrics_registry.add_to_registry(name='si_sdr')
# class SISDR:
#     def __init__(self, config):
#         self.device = config.exp.device

#     def __call__(self, real_batch, gen_batch) -> float:
#         real_wavs = real_batch['wav'].to(self.device).squeeze(1)
#         gen_wavs = gen_batch['gen_wav'].to(self.device).squeeze(1)

#         alpha = (gen_wavs * real_wavs).sum(
#             dim=1, keepdim=True
#         ) / real_wavs.square().sum(dim=1, keepdim=True)
#         real_wavs_scaled = alpha * real_wavs
#         e_target = real_wavs_scaled.square().sum(dim=1)
#         e_res = (gen_wavs - real_wavs).square().sum(dim=1)
#         si_sdr = 10 * torch.log10(e_target / e_res).cpu().numpy()

#         return float(np.mean(si_sdr).item())

def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix) :]] = weights[key]
    return result


# @metrics_registry.add_to_registry(name="mosnet")
# class MOSNet(ResampleMetric):
#     def __init__(self, config):
#         self.model = Wav2Vec2MOS("metrics/weights/wave2vec2mos.pth").to(config.exp.device)
#         super().__init__(config, self.model.sample_rate)

#     def __call__(self, real_batch, gen_batch) -> float:
#         mos_scores = []

#         for gen_wav in gen_batch['gen_wav']:
#             gen_wav = gen_wav / gen_wav.abs().max()
#             gen_wav_resampled = self.resample(gen_wav)

#             input_values = self.model.processor(
#                 gen_wav_resampled.cpu().numpy(), return_tensors="pt", sampling_rate=self.model.sample_rate
#             ).input_values.to("cuda")

#             with torch.no_grad():
#                 mos_score = self.model(input_values).item()
#             mos_scores.append(mos_score)

#         return float(np.mean(mos_scores).item())

# @metrics_registry.add_to_registry(name="utmos")
# class UTMOSMetric:
#     def __init__(self, config):
#         orig_sr = config.mel.out_sr if 'out_sr' in config.mel else config.mel.in_sr
#         self.utmos = UTMOSV2(orig_sr=orig_sr, device=config.exp.device).to(config.exp.device)
#         self.utmos.eval()
    
#     def __call__(self, real_batch, gen_batch) -> float:
#         with torch.no_grad():
#             moses = self.utmos(gen_batch['gen_wav'])
#         return float(torch.mean(moses).item())

@metrics_registry.add_to_registry(name="utmos")
class UTMOSMetric:
    def __init__(self, config):
        self.device = config.exp.device
        self.orig_sr = config.mel.out_sr if 'out_sr' in config.mel else config.mel.in_sr
        self.utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        ).to(self.device)
        self.utmos.eval()
    
    def __call__(self, real_batch, gen_batch) -> float:
        with torch.no_grad():
            moses = self.utmos(gen_batch['gen_wav'], sr=real_batch['noisy_sr'])
        return float(torch.mean(moses).item())
    
    


@metrics_registry.add_to_registry(name="wv-mos")
class WVMosMetric(ResampleMetric):
    def __init__(self, config):
        super().__init__(config, 16000)
        self.device = config.exp.device
        cuda = self.device.startswith('cuda')

        self.wvmos = WV_MOS(cuda = cuda)
        self.wvmos.to(self.device)
        self.processor = wvmos.wv_mos.Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def __call__(self, real_batch, gen_batch) -> float:
        with torch.no_grad():
            resampled = self.resample(gen_batch['gen_wav'].squeeze())

            input = self.processor(
                resampled,
                return_tensors="pt",
                padding=True,
                sampling_rate=16000
            ).input_values.to(self.device)
    
            score = self.wvmos(input)
            score = torch.mean(score).item()
            
        return float(score)

# @metrics_registry.add_to_registry(name="legacy_dnsmos")
# class DNSMosMetric:
#     def __init__(self, config):
#         self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(
#             fs=config.mel.out_sr,
#             personalized=True,
#             device=config.exp.device
#         )
    
#     def __call__(self, real_batch, gen_batch) -> float:
#         with torch.no_grad():
#             score = torch.mean(self.dnsmos(gen_batch['gen_wav']))
#         return float(score.item())

@metrics_registry.add_to_registry(name="dnsmos")
class DNSMosP835Metric:
    def __init__(self, config):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        p808_model_path = os.path.join(current_dir, "Mos_Evaluation/DNSMOS", "model_v8.onnx")

        primary_model_path = os.path.join(current_dir, "Mos_Evaluation/DNSMOS", "sig_bak_ovr.onnx")
        use_gpu = True
        self.model = DNSMOS_local(
            primary_model_path=primary_model_path,
            p808_model_path=p808_model_path,
            use_gpu=use_gpu,
            convert_to_torch=use_gpu,
        )
        self.in_sr = config.mel.out_sr
        self.device = config.exp.device
    
    def __call__(self, real_batch, gen_batch) -> float:
        with torch.no_grad():
            score = self.model(aud = gen_batch['gen_wav'][0], input_fs = real_batch['noisy_sr'])
            
        return {
            "dnsmos": score['OVRL'],
            "P808_mos": score['P808']
        }

@metrics_registry.add_to_registry(name="vcr")
class VCR:
    def __init__(self, config):
        self.vad_model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.device = config.exp.device
        self.vad_model = self.vad_model.to(self.device)
        (self.get_speech_timestamps,self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.utils
        self.sr = config.mel.in_sr
        self.out_sr = config.mel.out_sr
        
    def get_voice_portion_sum(self, audio_batch):
        voice_sums = []
        for audio in audio_batch:
            # Resample from self.out_sr to 16kHz
            if self.out_sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=self.out_sr, new_freq=16000).to(audio.device)
                audio = resampler(audio.unsqueeze(0)).squeeze(0)
            speech_timestamps = self.get_speech_timestamps(
                audio.to(self.device), self.vad_model, sampling_rate=self.sr, return_seconds=True
            )

            voice_sum = 0
            flag = bool(speech_timestamps)
            for timestamp in speech_timestamps:
                start_sample = int(timestamp["start"] * self.sr)
                end_sample = int(timestamp["end"] * self.sr)
                voice_sum += (end_sample - start_sample)
            voice_sums.append(voice_sum)
        return torch.tensor(voice_sums)
    
    def __call__(self, real_batch, gen_batch) -> float:
        real_voice_sums = self.get_voice_portion_sum(real_batch['wav'])
        gen_voice_sums = self.get_voice_portion_sum(gen_batch['gen_wav'])
        
        vad_diffs = real_voice_sums - gen_voice_sums
        vad_diff_ratios = vad_diffs / real_voice_sums
        
        return vad_diff_ratios.mean().item()