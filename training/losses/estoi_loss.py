import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-10  # Use a slightly larger epsilon for numerical stability

def resample_waveform(wave: torch.Tensor, orig_sr: int, target_sr: int):
    """Resamples a waveform to a target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return wave
    
    # Ensure wave is in (B, 1, T) for F.interpolate
    was_1d = wave.dim() == 1
    if was_1d:
        wave = wave.unsqueeze(0)
    
    if wave.dim() == 2:
        wave = wave.unsqueeze(1)
    
    B, C, T = wave.shape
    new_len = int(math.ceil(T * (target_sr / orig_sr)))
    
    wave_res = F.interpolate(wave, size=new_len, mode='linear', align_corners=False)
    
    # Return to original dimensions
    if wave.shape[1] == 1: # (B, 1, T) -> (B, T)
        wave_res = wave_res.squeeze(1)
    if was_1d: # (B, T) -> (T)
        wave_res = wave_res.squeeze(0)
        
    return wave_res

def build_third_octave_obm(sample_rate, n_fft, num_bands=15, min_freq=150.0, device='cpu', dtype=torch.float32):
    """Builds the 1/3-octave band matrix."""
    n_bins = n_fft // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_bins, device=device, dtype=dtype)
    
    band_factor = 2 ** (1.0 / 6.0)
    cf = min_freq * (2 ** (torch.arange(num_bands, device=device, dtype=dtype) / 3.0))
    lower = cf / band_factor
    upper = cf * band_factor

    obm = torch.zeros((num_bands, n_bins), device=device, dtype=dtype)
    for i in range(num_bands):
        mask = (freqs >= lower[i]) & (freqs <= upper[i])
        obm[i, mask] = 1
        
    # FIX: Correctly normalize each band to have unit energy using broadcasting and torch.where.
    # This prevents the IndexError by avoiding inconsistent mask shapes during indexing.
    obm_sum = obm.sum(dim=1, keepdim=True) # Shape (15, 1)
    
    # Create the reciprocal factor: 1 / (obm_sum + EPS) where sum > 0, and 1.0 otherwise.
    # The 1.0 ensures that zero-rows (bands with no bins) remain zero after multiplication.
    safe_reciprocal = torch.where(
        obm_sum > 0, 
        1.0 / (obm_sum + EPS), 
        torch.tensor(1.0, dtype=dtype, device=device)
    ) # Shape (15, 1)
    
    # Perform element-wise multiplication with broadcasting to normalize.
    obm = obm * safe_reciprocal 
    
    return obm, cf

def stft_mag(wave: torch.Tensor, n_fft: int, hop_length: int, win_length: int):
    """Computes the STFT magnitude of a waveform."""
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    
    window = torch.hann_window(win_length, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=True)
    return spec.abs()  # (B, n_bins, T_frames)

class DifferentiableESTOILoss(nn.Module):
    """
    A differentiable implementation of the Short-Time Objective Intelligibility (STOI) and
    Extended STOI (ESTOI) measures, designed to be used as a loss function in PyTorch.

    This implementation is carefully crafted to be mathematically sound and consistent
    with the original papers and reference implementations.
    """
    def __init__(self,
                 target_fs: int = 10000,
                 n_frame: int = 256,
                 n_fft: int = 512,
                 num_bands: int = 15,
                 min_freq: int = 150,
                 n_segment: int = 30,
                 beta: float = -15.0,
                 dyn_range: float = 40.0,
                 extended: bool = False):
        super().__init__()
        self.target_fs = target_fs
        self.n_frame = n_frame
        self.hop_length = n_frame // 2
        self.n_fft = n_fft
        self.num_bands = num_bands
        self.min_freq = min_freq
        self.n_segment = n_segment
        self.beta = beta
        self.dyn_range = dyn_range
        self.extended = extended
        
        # Register OBM as a buffer so it's moved to the correct device
        obm, _ = build_third_octave_obm(target_fs, n_fft, num_bands, min_freq, dtype=torch.float32)
        self.register_buffer('_obm', obm, persistent=False)

    @staticmethod
    def _remove_silent_frames_waveform(clean_wav, dyn_range, frame_len, hop_len):
        """
        Identifies and returns indices of non-silent frames from the time-domain waveform.
        This is the mathematically correct approach as per the original STOI implementation.
        """
        if clean_wav.dim() == 1:
            clean_wav = clean_wav.unsqueeze(0)
        
        # Create overlapping frames
        frames = F.unfold(
            clean_wav.unsqueeze(1).unsqueeze(-1),
            kernel_size=(frame_len, 1),
            stride=(hop_len, 1)
        ).squeeze(-1) # (B, frame_len, num_frames)
        
        # Calculate RMS energy in dB for each frame
        frame_energy_rms = torch.sqrt(torch.mean(frames**2, dim=1) + EPS)
        frame_energy_db = 20 * torch.log10(frame_energy_rms + EPS)
        
        max_energy_db = frame_energy_db.max(dim=1, keepdim=True)[0]
        
        # Identify valid (non-silent) frames
        valid_mask = frame_energy_db >= (max_energy_db - dyn_range)
        
        return [torch.where(mask)[0] for mask in valid_mask]

    @staticmethod
    def _row_col_normalize(x):
        """
        Performs the row and column normalization for Extended STOI.
        This operation is critical for ESTOI and must be mathematically correct.
        `x` has shape (J, M, N) - (segments, bands, time_frames_per_segment)
        """
        # Column normalization (over time N)
        x_centered_cols = x - x.mean(dim=2, keepdim=True)
        x_norm_cols = x_centered_cols / (torch.linalg.norm(x_centered_cols, dim=2, keepdim=True) + EPS)
        
        # Row normalization (over bands M)
        x_norm_cols_t = x_norm_cols.permute(0, 2, 1) # (J, N, M)
        x_centered_rows = x_norm_cols_t - x_norm_cols_t.mean(dim=2, keepdim=True)
        x_norm_rows = x_centered_rows / (torch.linalg.norm(x_centered_rows, dim=2, keepdim=True) + EPS)
        
        return x_norm_rows.permute(0, 2, 1) # Back to (J, M, N)

    def forward(self, clean_wav: torch.Tensor, noisy_wav: torch.Tensor, fs: int):
        """
        Calculates the differentiable (1 - STOI) loss.
        Args:
            clean_wav (Tensor): The clean reference speech waveform. Shape: (B, T) or (T,).
            noisy_wav (Tensor): The noisy or enhanced speech waveform. Shape: (B, T) or (T,).
            fs (int): The sample rate of the waveforms.
        Returns:
            Tensor: A scalar tensor representing the average (1 - STOI) loss.
        """
        if clean_wav.shape != noisy_wav.shape:
            raise ValueError("clean_wav and noisy_wav must have the same shape")
        
        # Resample if necessary
        if fs != self.target_fs:
            clean_wav = resample_waveform(clean_wav, fs, self.target_fs)
            noisy_wav = resample_waveform(noisy_wav, fs, self.target_fs)

        B = clean_wav.shape[0] if clean_wav.dim() > 1 else 1

        # Pad waveforms to be divisible by hop_length
        pad = (self.hop_length - (clean_wav.shape[-1] % self.hop_length)) % self.hop_length
        if pad > 0:
            clean_wav = F.pad(clean_wav, (0, pad))
            noisy_wav = F.pad(noisy_wav, (0, pad))
        
        # 1. Correctly identify silent frames from the time-domain clean waveform
        valid_indices_per_batch = self._remove_silent_frames_waveform(
            clean_wav, self.dyn_range, self.n_frame, self.hop_length
        )

        # 2. Compute STFTs
        clean_spec = stft_mag(clean_wav, self.n_fft, self.hop_length, self.n_frame)
        noisy_spec = stft_mag(noisy_wav, self.n_fft, self.hop_length, self.n_frame)
        
        # 3. Apply 1/3-octave band matrix
        obm = self._obm.to(dtype=clean_spec.dtype, device=clean_spec.device)
        clean_tob = torch.sqrt(torch.matmul(obm, clean_spec**2) + EPS)
        noisy_tob = torch.sqrt(torch.matmul(obm, noisy_spec**2) + EPS)

        scores = []
        for b in range(B):
            valid_idx = valid_indices_per_batch[b]
            
            if valid_idx.numel() < self.n_segment:
                scores.append(torch.tensor(0.0, device=clean_wav.device, dtype=clean_wav.dtype))
                continue

            # Select only the valid (non-silent) frames
            xb = clean_tob[b, :, valid_idx]  # Shape: (M, T_valid)
            yb = noisy_tob[b, :, valid_idx]  # Shape: (M, T_valid)
            
            # 4. Create overlapping segments using unfold
            # unfold(dimension, size, step)
            xb_seg = xb.unfold(1, self.n_segment, 1).permute(1, 0, 2) # (J, M, N)
            yb_seg = yb.unfold(1, self.n_segment, 1).permute(1, 0, 2) # (J, M, N)
            
            if self.extended:
                # 5a. Extended STOI calculation
                xb_n = self._row_col_normalize(xb_seg)
                yb_n = self._row_col_normalize(yb_seg)
                
                # As per paper: correlation over N, then average over J and M
                d_ext = (xb_n * yb_n).sum(dim=2)  # Shape: (J, M)
                score = d_ext.mean()
                scores.append(score)
            else:
                # 5b. Standard STOI calculation
                # Normalization
                norm_consts = torch.linalg.norm(xb_seg, dim=2) / (torch.linalg.norm(yb_seg, dim=2) + EPS)
                yb_norm = yb_seg * norm_consts.unsqueeze(-1)
                
                # Clipping
                clip_val = 10**(-self.beta / 20.0)
                y_primes = torch.min(yb_norm, xb_seg * (1 + clip_val))
                
                # Mean subtraction and final correlation
                y_primes_centered = y_primes - y_primes.mean(dim=2, keepdim=True)
                xb_centered = xb_seg - xb_seg.mean(dim=2, keepdim=True)
                
                y_primes_norm = y_primes_centered / (torch.linalg.norm(y_primes_centered, dim=2, keepdim=True) + EPS)
                xb_norm = xb_centered / (torch.linalg.norm(xb_centered, dim=2, keepdim=True) + EPS)
                
                # J*M is the total number of intermediate intelligibility measures
                J, M = xb_norm.shape[0], xb_norm.shape[1]
                d = (y_primes_norm * xb_norm).sum() / (J * M)
                scores.append(d)

        estoi_scores = torch.stack(scores)
        loss = (1.0 - estoi_scores).mean()
        return loss

# Quick test to ensure the loss runs
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    torch.manual_seed(42)
    B = 4
    sr = 16000
    T = int(2.5 * sr) # 2.5 seconds of audio
    
    # Create some dummy audio data
    t = torch.linspace(0, T/sr, T, device=device)
    clean = (0.8 * torch.sin(2 * math.pi * 440 * t) + 0.2 * torch.sin(2 * math.pi * 880 * t)).unsqueeze(0).repeat(B, 1)
    noise = (torch.randn_like(clean) * 0.1).to(device)
    enhanced = clean + noise
    enhanced.requires_grad_()
    print("enhanced wav grad:", enhanced.grad)
    
    print(f"Waveform shape: {clean.shape}")

    # Test standard STOI loss
    loss_fn = DifferentiableESTOILoss(target_fs=10000, extended=False).to(device)
    loss_val = loss_fn(clean, enhanced, fs=sr)
    print(f"Standard STOI Loss (1 - d): {loss_val.item():.4f}")
    print(f"Implied STOI score (d): {(1 - loss_val.item()):.4f}")

    # Test extended STOI loss
    loss_fn_ext = DifferentiableESTOILoss(target_fs=10000, extended=True).to(device)
    loss_val_ext = loss_fn_ext(clean, enhanced, fs=sr)
    loss_val_ext.backward()
    
    print("enhanced wav grad after backward:", enhanced.grad)
    print(f"Extended STOI Loss (1 - d_ext): {loss_val_ext.item():.4f}")
    print(f"Implied ESTOI score (d_ext): {(1 - loss_val_ext.item()):.4f}")

    # Test perfect match case (should result in loss close to 0)
    loss_perfect = loss_fn(clean, clean, fs=sr)
    print(f"Loss on identical signals (should be ~0): {loss_perfect.item():.6f}")