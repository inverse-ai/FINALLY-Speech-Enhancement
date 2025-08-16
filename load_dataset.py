import random
import torch
import torchaudio
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    key = str(fmax) + '_' + str(y.device)
    if key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate,   n_fft=n_fft,    n_mels=num_mels,    fmin=fmin,  fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y, (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.abs(spec)
    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)
    return spec

def load_wav(full_path):
    data, sampling_rate = torchaudio.load(full_path)
    return data, sampling_rate

def write_wav(full_path, data, sampling_rate):
    data = data.unsqueeze(0)
    torchaudio.save(full_path, data.cpu(), sampling_rate)

def get_dataset_filelist_noisy(input_training_file):
    with open(input_training_file, 'r', encoding='utf-8') as fi:
        return [x for x in fi.read().split('\n') if len(x) > 0]


class Load_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_clean_dir, input_noisy_dir, training_files, shuffle=True):
        self.input_clean_dir = input_clean_dir
        self.input_noisy_dir = input_noisy_dir
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)

    def __getitem__(self, index):
        clean_file = self.input_clean_dir + '/' + self.audio_files[index]
        noisy_file = self.input_noisy_dir + '/' + self.audio_files[index]

        clean_audio, _ = load_wav(clean_file)
        clean_audio = clean_audio.squeeze(0)
        
        noisy_audio, _ = load_wav(noisy_file)
        noisy_audio = noisy_audio.squeeze(0)
        
        return (clean_audio, noisy_audio)

    def __len__(self):
        return len(self.audio_files)