import os
import random
import torch
import torchaudio
import torch.utils.data
import numpy as np
import omegaconf
import scipy
import librosa
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

def load_config():
    conf_cli = OmegaConf.from_cli()
    config_path = conf_cli.exp.config_path
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(conf_file, conf_cli)
    config = apply_inheritance(config)
    return config

def save_wavs_to_dir(wav_batch, name_batch, path, sample_rate):
    os.makedirs(path, exist_ok=True)
    for name, wav in zip(name_batch, wav_batch):
        # if not name.endswith('.wav'):
        #     name += '.wav'
        file_path = os.path.join(path, name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if len(wav.shape) < 2:
            wav = wav.unsqueeze(0)
        torchaudio.save(file_path, wav.cpu(), sample_rate)
        # print(f"saved to file: {file_path}")

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)
    
    # tqdm.write(f"max value of y before mel spectrogram: {torch.max(y).item()}")
    # tqdm.write(f"min value of y before mel spectrogram: {torch.min(y).item()}")    
    # if torch.min(y) < -1.:
    #     tqdm.write(f'min value is {torch.min(y).item()}')
    # if torch.max(y) > 1.:
    #     tqdm.write(f'max value is {torch.max(y).item()}')
    
    # total = y.numel()
    # less_than_neg1 = (y < -1).sum().item() / total * 100
    # greater_than_1 = (y > 1).sum().item() / total * 100

    # tqdm.write(f"{less_than_neg1:.2f}% of values < -1, {greater_than_1:.2f}% of values > 1")


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

def split_audios(audios, segment_size, split, ret_bounds=False):
    audios = [torch.FloatTensor(torch.from_numpy(audio)).unsqueeze(0) for audio in audios]
    audio_start = 0

    if split:
        if audios[0].size(1) >= segment_size:
            max_audio_start = audios[0].size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            audios = [
                audio[:, audio_start : audio_start + segment_size]
                for audio in audios
            ]
        else:
            audios = [
                torch.nn.functional.pad(
                    audio,
                    (0, segment_size - audio.size(1)),
                    "constant",
                )
                for audio in audios
            ]
    
    audios = [audio.squeeze(0).numpy() for audio in audios]

    if ret_bounds:
        audio_end = audios[0].size if not split else audio_start + segment_size
        return audios, audio_start, audio_end
    
    return audios

def low_pass_filter(audio: np.ndarray, max_freq,
                    lp_type="default", orig_sr=16000):
    if lp_type == "random":
        lp_type = random.choice(['default', 'decimate'])
    if lp_type == "default":
        tmp = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=max_freq * 2, res_type="polyphase"
        )
    elif lp_type == "decimate":
        sub = orig_sr / (max_freq * 2)
        assert int(sub) == sub
        tmp = scipy.signal.decimate(audio, int(sub))
    else:
        raise NotImplementedError
    # soxr_hq is faster and better than polyphase,
    # but requires additional libraries installed
    # the speed difference is only 4 times, we can live with that
    tmp = librosa.resample(tmp, orig_sr=max_freq * 2, target_sr=16000, res_type="polyphase")
    return tmp[: audio.size]
