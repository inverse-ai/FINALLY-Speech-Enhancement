import torch
from torch_pesq import PesqLoss
from librosa.util import normalize
from scipy.io.wavfile import read

# Initialize PESQ calculator
pesq = PesqLoss(0.5, sample_rate=16000)  # Note: PESQ typically uses 16kHz

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

# Load your files
reference, ref_sr = load_wav("../../../data/vctk/clean_testset_wav/p232_003.wav")
print(reference.shape)
reference = normalize(reference)
reference = torch.FloatTensor(reference)

degraded, deg_sr = load_wav("../../../data/vctk/noisy_testset_wav/p232_003.wav")
degraded = normalize(degraded)
degraded = torch.FloatTensor(degraded)
# Calculate scores
with torch.no_grad():
    mos = pesq.mos(reference, degraded)
    
loss = pesq(reference, degraded)  # For training

print(f"MOS-LQO: {mos.item():.2f}, Loss: {loss.item():.4f}")

# Backpropagation (if training)
# loss.backward()