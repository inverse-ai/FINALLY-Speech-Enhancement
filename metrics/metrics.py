import os
import torch
from abc import ABC, abstractmethod
import torchaudio.transforms as T
from pesq import pesq
import numpy as np
from pystoi import stoi
from models.metric_models import WV_MOS
import wvmos
from tqdm import tqdm
from .dnsmos import DNSMOS_local

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

class UTMOSMetric:
    def __init__(self, config):
        self.orig_sr = config.mel.out_sr if 'out_sr' in config.mel else config.mel.in_sr
        self.utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        ).to(config.exp.device)
        self.utmos.eval()
    
    def __call__(self, real_batch, gen_batch) -> float:
        with torch.no_grad():
            moses = self.utmos(gen_batch['gen_wav'], sr=self.orig_sr)
        return float(torch.mean(moses).item())

class WVMosMetric(ResampleMetric):
    def __init__(self, config):
        super().__init__(config, 16000)
        self.device = config.exp.device
        
        self.wvmos = WV_MOS(device=self.device)
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
            ).input_values.to(resampled.device)
    
            score = self.wvmos(input)
            score = torch.mean(score).item()
            
        return float(score)

class DNSMosP808Metric:
    def __init__(self, config):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        p808_model_path = os.path.join(current_dir, "dns_weights", "model_v8.onnx")

        primary_model_path = os.path.join(current_dir, "dns_weights", "sig_bak_ovr.onnx")
        use_gpu = True
        self.model = DNSMOS_local(
            primary_model_path=primary_model_path,
            p808_model_path=p808_model_path,
            use_gpu=use_gpu,
            convert_to_torch=use_gpu,
            device=config.exp.device
        )
        self.in_sr = config.mel.out_sr
    
    def __call__(self, real_batch, gen_batch) -> float:
        with torch.no_grad():
            score = self.model(aud = gen_batch['gen_wav'][0], input_fs = self.in_sr)
            
        return {
            "dnsmos": score['OVRL'],
            "P808_mos": score['P808']
        }

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

class WbPesq(PesqBase):
    def __init__(self, config):
        super().__init__(config, mode='wb')

class NbPesq(PesqBase):
    def __init__(self, config):
        super().__init__(config, mode='nb')
    
class STOI:
    def __init__(self, config):
        self.sr = config.mel.out_sr if 'out_sr' in config.mel else config.mel.in_sr

    def __call__(self, real_batch, gen_batch) -> float:
        scores = []
        for real_wav, gen_wav in zip(real_batch['wav'], gen_batch['gen_wav']):
            real_wav_np = real_wav.cpu().numpy()
            gen_wav_np = gen_wav.cpu().numpy()

            stoi_score = stoi(real_wav_np, gen_wav_np, self.sr)
            scores.append(stoi_score)

        return float(np.mean(scores).item())

class SISDR:
    def __init__(self, config):
        self.device = config.exp.device

    def __call__(self, real_batch, gen_batch) -> float:
        real_wavs = real_batch['wav'].to(self.device).squeeze(1)
        gen_wavs = gen_batch['gen_wav'].to(self.device).squeeze(1)

        alpha = (gen_wavs * real_wavs).sum(
            dim=1, keepdim=True
        ) / real_wavs.square().sum(dim=1, keepdim=True)
        real_wavs_scaled = alpha * real_wavs
        e_target = real_wavs_scaled.square().sum(dim=1)
        e_res = (gen_wavs - real_wavs).square().sum(dim=1)
        si_sdr = 10 * torch.log10(e_target / e_res).cpu().numpy()

        return float(np.mean(si_sdr).item())
    
def get_metric(metric_name, config):
    metric_classes = {
        "wb_pesq": WbPesq,
        "nb_pesq": NbPesq,
        "stoi": STOI,
        "si_sdr": SISDR,
        "utmos": UTMOSMetric,
        "wvmos": WVMosMetric,
        "dnsmos": DNSMosP808Metric,
    }
    
    if metric_name not in metric_classes:
        raise ValueError(f"Metric '{metric_name}' is not recognized.")
    
    return metric_classes[metric_name](config)