import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import o
from stft import STFT
from pesq_loss.torch_pesq import PesqLoss
from utmos_loss.utmos import Score

def feature_loss(fmap_r, fmap_g):
    """
    Compute the relative feature matching loss as described in the paper.
    
    Args:
        fmap_r: List of feature maps from real samples for each discriminator
                Each element is a list of layer outputs for that discriminator
        fmap_g: List of feature maps from generated samples (same structure as fmap_r)
    
    Returns:
        Relative feature matching loss
    """
    loss = 0.0
    K = len(fmap_r)  # Number of discriminators
    L = len(fmap_r[0]) if K > 0 else 0  # Number of layers per discriminator
    
    for k in range(K):  # Loop through discriminators
        for l in range(L):  # Loop through layers
            # Get real and generated features for this layer
            D_k_l_real = fmap_r[k][l]
            D_k_l_generated = fmap_g[k][l]
            
            # Compute L1 distance between features
            diff = torch.abs(D_k_l_real - D_k_l_generated)
            numerator = torch.mean(diff)
            
            # Compute mean L1 norm of real features
            denominator = torch.mean(torch.abs(D_k_l_real))
            
            # Add to loss (with protection against division by zero)
            layer_loss = numerator / (denominator + 1e-6)
            loss += layer_loss
    
    # Normalize by total number of discriminator layers (K*L)
    return loss / (K * L) if (K * L) > 0 else torch.tensor(0.0)

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    # Targets for real and fake data
    loss = 0
    losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        real_target = torch.ones_like(dr)  # b = 1 for real data
        fake_target = torch.zeros_like(dg) # a = 0 for fake data

        # Calculate LSGAN loss for real and fake data
        real_loss = F.mse_loss(dr, real_target) / 2
        fake_loss = F.mse_loss(dg, fake_target) / 2
        losses.append(real_loss + fake_loss)
        loss += real_loss + fake_loss
    return loss, losses

def generator_loss(disc_outputs):
    # In LSGAN, the target for generated data is 1
    loss = 0
    for dg in disc_outputs:
        target = torch.ones_like(dg)
        loss += F.mse_loss(dg, target) / 2
    return loss

def stft(x, n_fft=1024, hop_length=256, win_length=1024):
    window = torch.hann_window(win_length).to(x.device)
    stft_result = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True, center=True, pad_mode='reflect'
    )
    return torch.abs(stft_result)

def lmos_loss(clean_mel_loss, gen_mel, clean_audio, gen_audio, wavlm):
    clean_features, _ = wavlm.extract_features(clean_audio)
    gen_features, _ = wavlm.extract_features(gen_audio)
    wavlm_loss = F.mse_loss(clean_features, gen_features)

    # clean_stft = stft(clean_audio)
    # gen_stft = stft(gen_audio)
    # stft_loss = F.l1_loss(clean_stft, gen_stft)
    # return 100 * wavlm_loss + stft_loss
    
    mel_loss = F.l1_loss(clean_mel_loss, gen_mel)
    return 100 * wavlm_loss + mel_loss



def pesq_loss(clean_audio, gen_audio):
    pesq = PesqLoss(1, sample_rate=48000, device=clean_audio.device)
    with torch.no_grad():
        loss = pesq.mos(clean_audio, gen_audio)
        return loss
    
def utmos_loss(audio, sr, score):
    return score.calculate_wav(audio,sr)

def human_feedback_loss(clean_audio, gen_audio, score, lambda_utmos=-20, lambda_pesq=-2, sr=48000):
    loss_pesq = pesq_loss(clean_audio,gen_audio).mean()
    loss_utmos = utmos_loss(gen_audio,sr,score).mean()
    return (lambda_utmos*loss_utmos+lambda_pesq*loss_pesq)


