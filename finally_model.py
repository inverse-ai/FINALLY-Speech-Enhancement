import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn.functional as F
from load_dataset import mel_spectrogram
from hifi_gan_gen import HiFiGeneratorBackbone
import torch.nn as nn
from modules import SpectralMaskNet, SpectralUnet, WaveUnet, ResidualBlock1d, WaveUnetUpsampler
from wavlm.wavlm import WavLM, WavLMConfig
from utils import o
    
class Finally(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spectralunet = SpectralUnet(in_channels=1, out_channels=1)  # Output: (B, 512, T_mel)
        self.resblock = ResidualBlock1d(1536, 3)
        self.conv = nn.Conv1d(1536,512,1)
        self.hifi = HiFiGeneratorBackbone()  # Expects input: (B, 512 + D_wavlm, T_mel)
        self.waveunet = WaveUnet(in_channels=2, out_channels=1)
        self.spectralmasknet = SpectralMaskNet()
        self.waveunet_upsampler = WaveUnetUpsampler(in_channels=1, out_channels=1)

        # WavLM components
        checkpoint = torch.load('wavlm/WavLM-Large.pt')
        cfg = WavLMConfig(checkpoint['cfg'])
        self.wavlm = WavLM(cfg)  # Initialize with your config
        self.wavlm.load_state_dict(checkpoint['model'])
        for param in self.wavlm.parameters():
            param.requires_grad = False  # Freeze WavLM

    def forward(self, x, use_upsample_waveunet = True):
        x_orig = x.clone()  # (B, T_audio)

        # --- WavLM Feature Extraction ---
        with torch.no_grad():
            # Extract features from raw waveform
            wavlm_features, _ = self.wavlm.extract_features(x_orig.squeeze(1))  # (B, T_wavlm, D_wavlm)
            wavlm_features = wavlm_features.transpose(1, 2)  # (B, D_wavlm, T_wavlm)

        # --- Spectral Processing ---
        x_mel = mel_spectrogram(x_orig, 1024, 80, 16000, 256, 1024, 0, 8000)  # (B, 80, T_mel)
        x = self.spectralunet(x_mel)  # (B, 512, T_mel)
        # --- Concatenate WavLM features ---
        # Interpolate WavLM features to match T_mel dimension
        wavlm_features = F.interpolate(wavlm_features, size=x.shape[-1], mode='nearest')  # (B, D_wavlm, T_mel)
        x = torch.cat([x, wavlm_features], dim=1)  # (B, 512 + D_wavlm, T_mel)


        x = self.resblock(x)
        x = self.conv(x)
        x = F.leaky_relu(x)
        x = self.hifi(x)  # (B, 1, T_audio)
        x = torch.stack((x, x_orig), dim=1)  # (B, 2, T_audio)
        x = self.waveunet(x)  # (B, 1, T_audio)
        x = self.spectralmasknet(x)  # (B, 1, T_audio)
        if use_upsample_waveunet:
            x = self.waveunet_upsampler(x)
        x = torch.tanh(x)
        return x.squeeze(1)