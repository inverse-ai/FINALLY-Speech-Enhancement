import torch
import os
import torchaudio
from torch.utils.data import Dataset
from utils.data_utils import read_file_list

class InferenceDataset(Dataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        noisy_wavs_dir,
        clean_wavs_dir
    ):
        if clean_wavs_dir:
            clean_wavs_dir = os.path.join(root, clean_wavs_dir)
        noisy_wavs_dir = os.path.join(root, noisy_wavs_dir)
        self.files_list = read_file_list(files_list_path)

        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.in_sr = mel_conf.in_sr
        self.out_sr = mel_conf.out_sr
        self.resamplers = {}

    def __getitem__(self, index):
        filename = self.files_list[index]
        clean_wav, clean_sr = torchaudio.load(os.path.join(self.clean_wavs_dir, filename))
        noisy_wav, noisy_sr = torchaudio.load(os.path.join(self.noisy_wavs_dir, filename))
        
        if clean_wav.shape[0] > 1:
            clean_wav = clean_wav.mean(dim=0, keepdim=True)
        if noisy_wav.shape[0] > 1:
            noisy_wav = noisy_wav.mean(dim=0, keepdim=True)
        
        clean_wav_orig = clean_wav.clone()
        noisy_wav_orig = noisy_wav.clone()

        if self.in_sr != noisy_sr:
            key = (noisy_sr, self.in_sr)
            if key not in self.resamplers:
                self.resamplers[key] = torchaudio.transforms.Resample(
                                        orig_freq= noisy_sr,
                                        new_freq= self.in_sr,
                                        resampling_method="sinc_interp_kaiser"
                                    )
            noisy_wav = self.resamplers[key](noisy_wav)
            
        if self.out_sr != clean_sr:
            key = (clean_sr, self.out_sr)
            if key not in self.resamplers:
                self.resamplers[key] = torchaudio.transforms.Resample(
                                        orig_freq=clean_sr,
                                        new_freq=self.out_sr,
                                        resampling_method="sinc_interp_kaiser"
                                    )
            clean_wav = self.resamplers[key](clean_wav)
            
        clean_pad = clean_wav.shape[-1] - noisy_wav.shape[-1] * (self.out_sr // self.in_sr)
        if clean_pad > 0:
            clean_wav = clean_wav[:, :-clean_pad]
        elif clean_pad < 0:
            clean_wav = torch.nn.functional.pad(clean_wav, (0, -clean_pad))
        
        base = 1024
        remainder = noisy_wav.shape[-1] % base
        pad_size = (base - remainder) if remainder != 0 else 0  
        noisy_wav = torch.nn.functional.pad(noisy_wav, (0, pad_size))

        noisy_wav = torch.clamp(noisy_wav, min=-1, max=1)
        clean_wav = torch.clamp(clean_wav, min=-1, max=1)
        

        return {
            'clean_wav_orig': clean_wav_orig.squeeze(),
            'noisy_wav_orig': noisy_wav_orig.squeeze(),
            'clean_sr': clean_sr,
            'noisy_sr': noisy_sr,
            'input_wav': noisy_wav.squeeze(),
            'wav': clean_wav.squeeze(),
            'name': filename,
            'pad': pad_size
        }

    def __len__(self):
        return len(self.files_list)