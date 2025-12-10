import os
import random
import torch
import torchaudio
import numpy as np
import omegaconf
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from omegaconf import OmegaConf
from tqdm import tqdm

MAX_WAV_VALUE = 32768.0

def debug_msg(str):
    tqdm.write('-'*20)
    tqdm.write(str)
    tqdm.write('-'*20)

def apply_inheritance(global_config):
    def apply_inheritance_impl(config):
        nonlocal global_config
        if isinstance(config, omegaconf.dictconfig.DictConfig):
            if 'inherit' in config:
                parent_config_name = config.inherit
                if parent_config_name in global_config:
                    parent_config = global_config[parent_config_name]
                    config = OmegaConf.merge(parent_config, config)
                    del config['inherit']
            
            for key, value in config.items():
                config[key] = apply_inheritance_impl(value)
        return config
    return apply_inheritance_impl(global_config)

def include_resolver(path):
    full_path = os.path.abspath(path)
    return OmegaConf.load(full_path)

def load_config():
    conf_cli = OmegaConf.from_cli()
    config_path = conf_cli.exp.config_path
    OmegaConf.register_new_resolver("include", include_resolver)
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(conf_file, conf_cli)
    config = apply_inheritance(config)
    return config

def save_config(config, out_dir, base_name="config"):
    save_dir = os.path.join(out_dir, "configs")
    os.makedirs(save_dir, exist_ok=True)
    
    i = 1
    while True:
        filename = f"{base_name}_{i}.yaml"
        path = os.path.join(save_dir, filename)
        if not os.path.exists(path):
            OmegaConf.save(config, path)
            return
        i += 1

def save_wavs_to_dir(wav_batch, name_batch, path, sample_rate, format):
    os.makedirs(path, exist_ok=True)
    for name, wav in zip(name_batch, wav_batch):
        if not name.endswith('.' + format):
            name += '.' + format
        file_path = os.path.join(path, name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if len(wav.shape) < 2:
            wav = wav.unsqueeze(0)
        torchaudio.save(file_path, wav.cpu(), sample_rate)

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)


    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                    center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def read_file_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file_list = [line.strip() for line in f.readlines() if line.strip()]
    return file_list

def get_chunk(audio, segment_size):
    if audio.shape[0] >= segment_size:
        max_audio_start = audio.shape[0] - segment_size
        audio_start = random.randint(0, max_audio_start)
        audio = audio[audio_start : audio_start + segment_size]
    else:
        audio = torch.nn.functional.pad(audio, ((segment_size - audio.shape[0]) // 2, (segment_size - audio.shape[0] + 1) // 2), "constant",)
    
    return audio
