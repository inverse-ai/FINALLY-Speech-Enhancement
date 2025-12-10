import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch import nn
from torch_pesq import PesqLoss

class WavLMLoss(nn.Module):
    def __init__(self, target_sr=16000):
        super().__init__()
        self.target_sr = target_sr

    def _extract_features(self, wavlm, input):
        extract_features = wavlm.feature_extractor(input)
        extract_features = extract_features.transpose(1, 2)
        return extract_features

    def forward(self, real_wav, gen_wav, wavlm, sr):
        if len(real_wav.shape) == 3:
            real_wav = real_wav.squeeze(1)
        if len(gen_wav.shape) == 3:
            gen_wav = gen_wav.squeeze(1)
        
        if sr != self.target_sr:
            real_wav = torchaudio.functional.resample(real_wav, orig_freq=sr, new_freq=self.target_sr)
            gen_wav = torchaudio.functional.resample(gen_wav, orig_freq=sr, new_freq=self.target_sr)

        real_wavlm_conv_features = self._extract_features(wavlm, real_wav)
        gen_wavlm_conv_features = self._extract_features(wavlm, gen_wav)

        wavlm_loss = F.mse_loss(real_wavlm_conv_features, gen_wavlm_conv_features)
        
        return wavlm_loss
    
class STFTLoss(nn.Module):
    def __init__(self, fft_size=1024, hop_length=256):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length

        self.stft = T.Spectrogram(n_fft=fft_size, hop_length=hop_length, power=1)

    def forward(self, real_wav, gen_wav, wavlm, sr):
        if len(real_wav.shape) == 3:
            real_wav = real_wav.squeeze(1)
        if len(gen_wav.shape) == 3:
            gen_wav = gen_wav.squeeze(1)

        real_stft = self.stft(real_wav)
        gen_stft = self.stft(gen_wav)
        stft_loss = F.l1_loss(real_stft, gen_stft)
        
        return stft_loss

class GeneratorLoss(nn.Module):
    def forward(self, discs_gen_out):
        loss = 0
        for disc_gen_out in discs_gen_out:
            one_disc_loss = F.mse_loss(disc_gen_out, torch.ones_like(disc_gen_out))
            loss += one_disc_loss
        return loss

class DiscriminatorLoss(nn.Module):        
    def forward(self, discs_real_out, discs_gen_out):
        loss = 0
        for disc_real_out, disc_gen_out in zip(discs_real_out, discs_gen_out):
            real_one_disc_loss = F.mse_loss(disc_real_out, torch.ones_like(disc_real_out))
            gen_one_disc_loss = F.mse_loss(disc_gen_out, torch.zeros_like(disc_gen_out))
            loss += (real_one_disc_loss + gen_one_disc_loss)
        return loss

class FeatureLoss(nn.Module):
    def forward(self, fmaps_real, fmaps_gen):
        loss = 0.0
        K = len(fmaps_gen)
        if K == 0:
            return torch.tensor(0.0)
        L = len(fmaps_real[0])
        
        for k in range(K):
            for l in range(L):
                D_k_l_real = fmaps_real[k][l]
                D_k_l_generated = fmaps_gen[k][l]
                numerator = torch.mean(torch.abs(D_k_l_real - D_k_l_generated))
                denominator = torch.mean(torch.abs(D_k_l_real)) + 1e-8
                loss += numerator / denominator
        
        return loss / (K * L)

class UTMOSLoss(nn.Module):
    def __init__(
        self,
        sample_rate=48000,
        device=None
    ):
        super().__init__()
        utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        ).to(device)

        utmos.train()

        self.utmos = utmos
        self.sample_rate = sample_rate
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def forward(self, gen_wav):
        if gen_wav.ndim == 3:
            gen_wav = gen_wav.squeeze(1)

        gen_wav = gen_wav.to(self.device)
        if not gen_wav.requires_grad:
            gen_wav.requires_grad_(True)

        mos = self.utmos(gen_wav, sr=self.sample_rate)

        normalized = 1 + 4 * self.sigmoid(mos)
        return -normalized.mean()
class PesqLoss_(PesqLoss):
    def __init__(self, factor=0.5, sample_rate=48000):
        super().__init__(factor=factor, sample_rate=sample_rate)

    def forward(self, real_wav, gen_wav):
        batch_size = real_wav.shape[0]
        pesq_loss_list = []

        for i in range(batch_size):
            if real_wav[i].sum().item() == 0 or gen_wav[i].sum().item() == 0:
                continue
            loss = super().forward(real_wav[i].squeeze(), gen_wav[i].squeeze())[0]
            pesq_loss_list.append(loss)

        pesq_loss = torch.stack(pesq_loss_list)

        return torch.mean(pesq_loss)
    