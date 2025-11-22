import utmosv2
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio.transforms as T
import torchvision
import utmosv2
from transformers import AutoModel
import torch
import wvmos
import os
import librosa
import numpy as np
from utils.model_utils import requires_grad
import librosa
import onnxruntime as ort

class Wav2Vec2MOS(nn.Module):
    sample_rate = 16_000

    def __init__(self, path, freeze=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        self.freeze = freeze

        self.dense = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        ).to(device)

        if self.freeze:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        state_dict = torch.load(path)["state_dict"]
        self.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
        self.eval()
        self.cuda()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.to(device)

    def forward(self, x):
        x = self.encoder(x)["last_hidden_state"]
        x = self.dense(x)
        return x.mean(dim=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()

class SSLEncoder(nn.Module):
    def __init__(self, sr: int, model_name: str, freeze: bool):
        super().__init__()
        self.sr = sr
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        if x.dim() == 3:
            x = x.squeeze(1)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        std = std.clamp(min=1e-5)
        x = (x - mean) / std

        x = x.to(self.model.device)
        outputs = self.model(input_values=x, output_hidden_states=True)
        return outputs.hidden_states

class UTMOSV2(nn.Module):
    def __init__(self, orig_sr, device, mel_on_cpu=False):
        super().__init__()
        self.orig_sr = orig_sr
        self.target_sr = 16000
        self.ssl_duration = 3
        self.spec_frames_frame_sec = 1.4
        self.spec_frames_num_frames = 2

        spec_cfg_defaults = dict(
            n_fft=4096,
            hop_length=32,
            n_mels=512,
            shape=(512, 512),
            norm=80,
        )
        self.spec_cfgs = [
            dict(**spec_cfg_defaults, win_length=4096),
            dict(**spec_cfg_defaults, win_length=2048),
            dict(**spec_cfg_defaults, win_length=1024),
            dict(**spec_cfg_defaults, win_length=512),
        ]
        self.valid_transform = torchvision.transforms.Resize((512, 512))
        self.spec_frames_mixup_inner = True
        self.spec_frames_mixup_alpha = 0.4
        
        self.utmos_wrapper = utmosv2.create_model(device=device)
        self.utmos = self.utmos_wrapper._model 
        self.utmos.ssl.encoder = SSLEncoder(
            sr=self.target_sr,
            model_name="facebook/wav2vec2-base",
            freeze=True
        ).to(device)

        self.resample = T.Resample(
            orig_freq=self.orig_sr,
            new_freq=self.target_sr,
            resampling_method="sinc_interp_kaiser")
        self.mel_spectrograms = nn.ModuleList([
            T.MelSpectrogram(
                sample_rate=self.target_sr,
                n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"],
                win_length=cfg["win_length"],
                n_mels=cfg["n_mels"],
                power=2.0,
            ) for cfg in self.spec_cfgs
        ])
        self.mel_on_cpu = mel_on_cpu

        requires_grad(self, False)

    def _extend_audio(self, y: torch.Tensor, length: int, mode: str = "tile") -> torch.Tensor:
        if y.shape[-1] >= length:
            return y[:, :length]
        elif mode == "tile":
            n = length // y.shape[-1]
            if length % y.shape[-1] != 0:
                n += 1

            y = y.repeat(1, n)[:, :length]
            return y
        else:
            raise NotImplementedError

    def _select_random_start(self, y: torch.Tensor, length: int) -> torch.Tensor:
        start = torch.randint(0, y.shape[-1] - length + 1, (1,)).item()
        return y[:, start : start + length]

    def _make_melspec(self, y: torch.Tensor, mel_transform: T.MelSpectrogram, norm: float) -> torch.Tensor:
        if self.mel_on_cpu:
            device = y.device
            spec = mel_transform(y.cpu()).to(device)
        else:
            spec = mel_transform(y)

        spec = spec.float()
        spec = spec.clamp(min=1e-5)
        ref_value = spec.amax(dim=[1, 2], keepdim=True).clamp(min=1e-5)
        spec_db = 10 * (torch.log10(spec) - torch.log10(ref_value))

        if norm is not None:
            spec_db = (spec_db + norm) / norm

        spec_db = spec_db.unsqueeze(1).repeat(1, 3, 1, 1)

        return spec_db


    def _getitem_ssl(self, y: torch.Tensor) -> torch.Tensor:
        length = int(self.ssl_duration * self.target_sr)
        y = self._extend_audio(y, length, mode="tile")
        y = self._select_random_start(y, length)
        return y

    def _getitem_multispec(self, y: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        specs = []
        length = int(self.spec_frames_frame_sec * self.target_sr)
        y = self._extend_audio(y, length)

        for _ in range(self.spec_frames_num_frames):
            y1 = self._select_random_start(y, length)
            batch_specs = []
            
            for mel_transform, spec_cfg in zip(self.mel_spectrograms, self.spec_cfgs):
                spec = self._make_melspec(y1, mel_transform, spec_cfg["norm"])

                if self.spec_frames_mixup_inner:
                    y2 = self._select_random_start(y, length)
                    spec2 = self._make_melspec(y2, mel_transform, spec_cfg["norm"])

                    lmd = torch.distributions.Beta(
                        self.spec_frames_mixup_alpha,
                        self.spec_frames_mixup_alpha
                    ).sample()
                    spec = lmd * spec + (1 - lmd) * spec2

                spec = self.valid_transform(spec)
                batch_specs.append(spec)

            specs.append(torch.stack(batch_specs, dim=1))

        return torch.cat(specs, dim=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.orig_sr != self.target_sr:
            y = self.resample(y)

        if self.mel_on_cpu:
            self.mel_spectrograms = self.mel_spectrograms.cpu()

        ssl_input = self._getitem_ssl(y)
        spec_input = self._getitem_multispec(y)

        d = torch.zeros((y.shape[0], 10), device=ssl_input.device)
        d[:, 1] = 1

        return self.utmos(ssl_input, spec_input, d)

class WV_MOS(nn.Module):
    def __init__(self, cuda=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = wvmos.wv_mos.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        path = os.path.join(os.path.expanduser('~'), ".cache/wv_mos/wv_mos.ckpt")
        
        self.dense = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(device)
        
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        
        checkpoint = torch.load(
            path,
            weights_only=False,
            map_location='cpu' if not cuda else 'cuda'
        )['state_dict']
        self.load_state_dict(
            wvmos.wv_mos.extract_prefix('model.', checkpoint)
        )
        
        self.eval()
        if cuda:
            self.cuda()
        
    def forward(self, x):
        x = self.encoder(x)['last_hidden_state']
        x = self.dense(x)
        x = x.mean(dim=[1,2], keepdims=True)
        return x

class DNSMOSPredictor:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.sampling_rate = 16000
        self.input_length = 9.01

        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, aud, input_fs, is_personalized_MOS):
        fs = self.sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(self.input_length*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - self.input_length)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+self.input_length)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict
