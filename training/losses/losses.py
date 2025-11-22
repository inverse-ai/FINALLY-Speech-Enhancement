import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
from torch import nn
from .loss_builder import losses_registry
from torch_pesq import PesqLoss
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from transformers import Wav2Vec2ForCTC
import soxr
from utils.model_utils import unwrap_model, requires_grad
from models.metric_models import UTMOSV2

@losses_registry.add_to_registry(name='feature_loss')
class FeatureLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

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

@losses_registry.add_to_registry(name='disc_loss')
class DiscriminatorLoss(nn.Module): 
    def __init__(self, device=None):
        super().__init__()
        self.device = device  
   
    def forward(self, discs_real_out, discs_gen_out):
        loss = 0
        for disc_real_out, disc_gen_out in zip(discs_real_out, discs_gen_out):
            real_one_disc_loss = F.mse_loss(disc_real_out, torch.ones_like(disc_real_out))
            gen_one_disc_loss = F.mse_loss(disc_gen_out, torch.zeros_like(disc_gen_out))
            loss += (real_one_disc_loss + gen_one_disc_loss)
        loss /= len(discs_real_out)
        return loss

@losses_registry.add_to_registry(name='gen_loss')
class GeneratorLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def forward(self, discs_gen_out):
        loss = 0
        for disc_gen_out in discs_gen_out:
            one_disc_loss = F.mse_loss(disc_gen_out, torch.ones_like(disc_gen_out))
            loss += one_disc_loss
        loss /= len(discs_gen_out)
        return loss

@losses_registry.add_to_registry(name='l1_mel_loss')
class L1Loss(nn.L1Loss):
    def forward(self, gen_mel, real_mel):
        return super().forward(gen_mel, real_mel)

@losses_registry.add_to_registry(name='lmos')
class LMOSLoss(nn.Module):
    def __init__(self, target_sr=16000, fft_size=1024, hop_length=256, device=None):
        super().__init__()
        self.target_sr = target_sr
        self.fft_size = fft_size
        self.hop_length = hop_length

        self.stft = T.Spectrogram(n_fft=fft_size, hop_length=hop_length, power=1)

    def _extract_features(self, wavlm, input):
        outputs = wavlm(input, output_hidden_states=True)  # Add batch dim
        layer_0_features = outputs.hidden_states[1]  # First transformer layer
        return layer_0_features

    def forward(self, real_wav, gen_wav, wavlm):
        # it should be guaranteed that wavlm weights not require grad in eval mode
        if len(real_wav.shape) == 3:
            real_wav = real_wav.squeeze(1)
        if len(gen_wav.shape) == 3:
            gen_wav = gen_wav.squeeze(1)

        real_features = self._extract_features(wavlm, real_wav)
        gen_features = self._extract_features(wavlm, gen_wav)

        feature_loss = F.mse_loss(real_features, gen_features)

        real_stft = self.stft(real_wav)
        gen_stft = self.stft(gen_wav)
        stft_loss = F.l1_loss(real_stft, gen_stft)

        # print("Feature Loss:", feature_loss.item(), "STFT Loss:", stft_loss.item())
        lmos_loss = 100 * feature_loss + stft_loss
        # lmos_loss = stft_loss
        return lmos_loss


@losses_registry.add_to_registry(name='feature_loss')
class LMOSLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()

    def _extract_features(self, wavlm, input):
        outputs = wavlm(input, output_hidden_states=True)  # Add batch dim
        layer_0_features = outputs.hidden_states[1]  # First transformer layer
        return layer_0_features

    def forward(self, real_wav, gen_wav, wavlm):
        # it should be guaranteed that wavlm weights not require grad in eval mode
        if len(real_wav.shape) == 3:
            real_wav = real_wav.squeeze(1)
        if len(gen_wav.shape) == 3:
            gen_wav = gen_wav.squeeze(1)

        real_features = self._extract_features(wavlm, real_wav)
        gen_features = self._extract_features(wavlm, gen_wav)

        feature_loss = F.mse_loss(real_features, gen_features)

        loss = 100 * feature_loss
        return loss
    
@losses_registry.add_to_registry(name='stft_loss')
class LMOSLoss(nn.Module):
    def __init__(self, target_sr=16000, fft_size=1024, hop_length=256, device=None):
        super().__init__()
        self.target_sr = target_sr
        self.fft_size = fft_size
        self.hop_length = hop_length

        self.stft = T.Spectrogram(n_fft=fft_size, hop_length=hop_length, power=1)

    def forward(self, real_wav, gen_wav, wavlm):
        # it should be guaranteed that wavlm weights not require grad in eval mode
        if len(real_wav.shape) == 3:
            real_wav = real_wav.squeeze(1)
        if len(gen_wav.shape) == 3:
            gen_wav = gen_wav.squeeze(1)

        real_stft = self.stft(real_wav)
        gen_stft = self.stft(gen_wav)
        stft_loss = F.l1_loss(real_stft, gen_stft)

        return stft_loss

# @losses_registry.add_to_registry(name='pesq_prev')
# class PesqLoss_(PesqLoss):
#     def __init__(self, factor=0.5, sample_rate=48000):
#         super().__init__(factor=factor, sample_rate=sample_rate)

#     def forward(self, real_wav, gen_wav):
#         return -torch.mean(super().mos(real_wav.squeeze(), gen_wav.squeeze()))
    
    
# @losses_registry.add_to_registry(name='pesq')
# class PESQLossv2(nn.Module):
#     """
#     PESQ-based loss using torchmetrics.audio.PESQ.
#     Higher PESQ -> lower loss.
#     """

#     def __init__(self, sample_rate: int = 48000, factor: float = 0.5):
#         """
#         Args:
#             fs (int): sampling rate of the audio (8000 or 16000)
#         """
#         super().__init__()
#         if sample_rate == 8000:
#             mode = "nb"
#         elif sample_rate >= 16000:
#             mode = "wb"
#         else:
#             raise ValueError("Sampling rate must be 8000 or >=16000")
        
#         self.sample_rate = sample_rate
#         self.mode = mode
#         self.pesq_metric = PerceptualEvaluationSpeechQuality(self.sample_rate, self.mode)

#     def forward(self, reference: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             reference (torch.Tensor): (B, T) reference audio
#             enhanced (torch.Tensor): (B, T) enhanced audio
#         Returns:
#             torch.Tensor: scalar PESQ loss
#         """
#         # Compute PESQ score
#         pesq_scores = self.pesq_metric(enhanced, reference)  # returns tensor of shape (B,)
        
#         # Convert to loss (higher PESQ -> lower loss)
#         loss = -pesq_scores.mean()
#         return loss

# @losses_registry.add_to_registry(name='pesq')
# class PesqLoss_(PesqLoss):
#     def __init__(self, factor=0.5, sample_rate=48000):
#         super().__init__(factor=factor, sample_rate=sample_rate)

#     def forward(self, real_wav, gen_wav):
#         device = gen_wav.device
#         batch_size = real_wav.shape[0]
#         pesq_mos_list = []

#         for i in range(batch_size):
#             # Get PESQ MOS for each pair individually
#             if real_wav[i].sum().item() == 0 or gen_wav[i].sum().item() == 0:
#                 continue
#             pesq_val = super().mos(real_wav[i].squeeze(), gen_wav[i].squeeze())[0]

#             # Append the PESQ value
#             pesq_mos_list.append(pesq_val)

#         # Stack into a single tensor
#         pesq_mos = torch.stack(pesq_mos_list)

#         return -torch.mean(pesq_mos)

@losses_registry.add_to_registry(name='pesq')
class PesqLoss_(PesqLoss):
    def __init__(self, factor=1, sample_rate=48000, device=None):
        self.device = device
        super().__init__(factor=factor, sample_rate=sample_rate)

    def forward(self, real_wav, gen_wav):
        return torch.mean(super().forward(real_wav.squeeze(), gen_wav.squeeze()))
        

@losses_registry.add_to_registry(name='utmos_prev')
class UTMOSLoss(nn.Module):
    def __init__(
        self,
        sample_rate=48000,
        use_grad_chp=True,
        device=None
    ):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        utmos = UTMOSV2(orig_sr=sample_rate, device=device)

        n_gpus = torch.cuda.device_count()
        n_gpus = 1
        if n_gpus > 1:
            utmos = nn.DataParallel(utmos)

        self.utmos = utmos.to(device)
        self.utmos.eval()
        requires_grad(self.utmos, False)

        if use_grad_chp:
            unwrap_model(self.utmos).utmos.ssl.encoder.model.gradient_checkpointing_enable()
            for backbone in unwrap_model(self.utmos).utmos.spec_long.backbones:
                backbone.set_grad_checkpointing(True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, gen_wav):
        if gen_wav.ndim == 3:
            gen_wav = gen_wav.squeeze(1)
        
        mos = self.utmos(gen_wav)
        normalized = 1 + 4 * self.sigmoid(mos)
        # print("UTMOS scores:", normalized.detach().cpu().numpy())
        return -normalized.mean()
        return -mos.mean()



@losses_registry.add_to_registry(name='utmos')
class UTMOSLossV2(nn.Module):
    def __init__(self, sample_rate=48000, use_grad_chp=True, device=None):
        super().__init__()
        self.device = device

        # === Load new UTMOS model from torch.hub ===
        # We replace UTMOSV2 with the pretrained UTMOS22 Strong model
        utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        ).to(device)

        # IMPORTANT: keep .train() so gradients flow backward
        utmos.train()

        # DataParallel (optional, same as before)
        n_gpus = torch.cuda.device_count()
        n_gpus = 1
        if n_gpus > 1:
            utmos = nn.DataParallel(utmos)

        self.utmos = utmos
        self.sample_rate = sample_rate
        self.device = device
        self.sigmoid = nn.Sigmoid()

        # No gradient checkpointing here, since the torch.hub model
        # doesn’t expose the same structure (encoder/spec_long/backbones)
        # So skip this part safely.

    def forward(self, gen_wav):
        # Input format normalization (same as original)
        if gen_wav.ndim == 3:
            gen_wav = gen_wav.squeeze(1)  # shape: (B, T)

        # Ensure requires_grad=True for backward
        gen_wav = gen_wav.to(self.device)
        if not gen_wav.requires_grad:
            gen_wav.requires_grad_(True)

        # === Forward through UTMOS ===
        mos = self.utmos(gen_wav, sr=self.sample_rate)

        # === Normalize and return loss ===
        normalized = 1 + 4 * self.sigmoid(mos)
        return -normalized.mean()
    
    
    
# @losses_registry.add_to_registry(name='phoneme_loss')
# class PhonemeFeatureL1Loss(nn.Module):
#     """
#     Phoneme-level L1 loss based on the early layers of a pretrained Wav2Vec2 model.
#     """

#     def __init__(self, sample_rate=48000, device="cuda:1", freeze_extractor=False):
#         super().__init__()
        
#         checkpoint="facebook/wav2vec2-xlsr-53-phon-cv-babel-ft"
#         #self.processor = Wav2Vec2Processor.from_pretrained(checkpoint)
#         base_model = Wav2Vec2ForCTC.from_pretrained(checkpoint)

#         # Keep only the feature extractor and the first transformer layer
#         base_model.wav2vec2.encoder.layers = nn.ModuleList(
#             [base_model.wav2vec2.encoder.layers[0]]
#         )

#         self.feature_model = base_model.wav2vec2.to(device)
#         self.sample_rate = sample_rate
#         self.fs = 16000
#         self.device = device

#         # Optionally freeze the extractor
#         if freeze_extractor:
#             for param in self.feature_model.parameters():
#                 param.requires_grad = False

#         self.l1 = nn.L1Loss()

#     def forward(self, real_wav: torch.Tensor, gen_wav: torch.Tensor):
#         """
#         Args:
#             real_wav: [B, T] or [T]
#             gen_wav: [B, T] or [T]
#         Returns:
#             L1 loss between ref and inf feature representations
#         """
#         if real_wav.dim() == 3:
#             real_wav = real_wav.mean(dim=1)
#             gen_wav = gen_wav.mean(dim=1)
#         if real_wav.dim() == 1:
#             real_wav = real_wav.unsqueeze(0)
#             gen_wav = gen_wav.unsqueeze(0)

#         if self.fs != self.sample_rate:
#             resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=self.fs).to(self.device)
#             real_wav = resampler(real_wav)
#             gen_wav = resampler(gen_wav)
        
#         # Extract features (shared model)
#         ref_feat = self.feature_model(real_wav).last_hidden_state  # [B, T', D]
#         inf_feat = self.feature_model(gen_wav).last_hidden_state  # [B, T', D]

#         # Match time dimension
#         min_len = min(ref_feat.size(1), inf_feat.size(1))
#         ref_feat = ref_feat[:, :min_len, :]
#         inf_feat = inf_feat[:, :min_len, :]

#         loss = self.l1(ref_feat, inf_feat)
#         return loss
    
    
from .estoi_loss import DifferentiableESTOILoss
# @losses_registry.add_to_registry(name='estoi_loss')
# class ESTOILoss(nn.Module):
#     def __init__(self, sample_rate=16000, device='cuda:1'):
#         super().__init__()
#         self.sample_rate = sample_rate
#         self.device = device
#         self.estoi_loss_fn = DifferentiableESTOILoss(extended=True).to(self.device)
#     def forward(self, real_wav, gen_wav):
#         if real_wav.ndim == 1:
#             real_wav = real_wav.unsqueeze(0)
#         if gen_wav.ndim == 1:
#             gen_wav = gen_wav.unsqueeze(0)
#         if gen_wav.ndim == 3:
#             gen_wav = gen_wav.mean(dim=1)
#         if real_wav.ndim == 3:
#             real_wav = real_wav.mean(dim=1)
#         loss = self.estoi_loss_fn(clean_wav = real_wav, noisy_wav = gen_wav, fs=self.sample_rate)
#         return loss

