import torch
import random
import numpy as np
import os
import torchaudio

from torch.utils.data import Dataset

from utils.data_utils import read_file_list, get_chunk
import datasets_manager.augmentations_modules as augmentations_modules
from tqdm import tqdm

class AugmentedDataset(Dataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        seed=42,
        augs_conf=tuple()
    ):
        self.root = root
        self.files_list = read_file_list(files_list_path)
        self.segment_size = mel_conf.segment_size
        self.in_sr = mel_conf.in_sr
        self.out_sr = mel_conf.out_sr
        self.seed = seed
        self.augs_conf = augs_conf
        self.resamplers = {}

    def normalize(self, wav):
        percentile = 0.999
        target_peak = 0.75
        clip_val = torch.quantile(wav.abs(), percentile)
        wav = torch.clamp(wav, -clip_val, clip_val)
        max_val = wav.abs().max()
        if max_val > 0 and max_val < target_peak:
            wav = wav / max_val * target_peak    
        return wav
        

    def _apply_augs(self, wav):
        result = wav.clone()
        orig_len = result.shape[-1]

        for aug in self.augs_conf:
            name = aug['name']
            args = aug['args'].copy()
            try:
                if name == 'noise':
                    args['noise_files_path'] = args['noise_files_path']['train']
                    aug_func = augmentations_modules.RandomNoise(sr=self.out_sr, **args)
                elif name == 'impulse_response':
                    args['ir_files_path'] = args['ir_files_path']['train']
                    aug_func = augmentations_modules.RandomImpulseResponse(sr=self.out_sr, **args)
                elif name == 'acrusher':
                    aug_func = augmentations_modules.RandomAcrusher(sr=self.out_sr, **args)
                elif name == 'crystalizer':
                    aug_func = augmentations_modules.RandomCrystalizer(sr=self.out_sr, **args)
                elif name == 'vibrato':
                    aug_func = augmentations_modules.RandomVibrato(sr=self.out_sr, **args)
                elif name == 'flanger':
                    aug_func = augmentations_modules.RandomFlanger(sr=self.out_sr, **args)
                elif name == 'codec':
                    aug_func = augmentations_modules.RandomCodec(sr=self.out_sr, **args)
                elif name == 'bandwidth_limitation':
                    aug_func = augmentations_modules.RandomBandwidthLimitation(sr=self.out_sr, **args)
                else:
                    tqdm.write(f"Unknown augmentation: {name}, skipping...")
                    continue
                result = aug_func(result)
            except Exception as e:
                tqdm.write(f"Failed to apply {aug.__class__.__name__}: {e}")
        return result[:,:orig_len]

    def __len__(self):
        return len(self.files_list)

class TrainingDataset(AugmentedDataset):
    def __init__(
        self,
        root,
        files_list_path,
        mel_conf,
        seed=42,
        augs_conf=tuple()
    ):
        super().__init__(
            root=root,
            files_list_path=files_list_path,
            mel_conf=mel_conf,
            seed=seed,
            augs_conf=augs_conf,
        )
        self.segment_size = mel_conf.segment_size
    
    def __getitem__(self, index):
        filename = self.files_list[index]

        wav, sr = torchaudio.load(os.path.join(self.root, filename))            
        wav = torch.nan_to_num(wav)
        wav = torch.clamp(wav, min=-1, max=1)

        if self.out_sr != sr:
            key = (sr, self.out_sr)
            if key not in self.resamplers:
                self.resamplers[key] = torchaudio.transforms.Resample(
                                        orig_freq=sr,
                                        new_freq=self.out_sr,
                                        resampling_method="sinc_interp_kaiser"
                                    )
            wav = self.resamplers[key](wav)

        wav = get_chunk(wav.squeeze(0), self.segment_size)

        augmented = self._apply_augs(wav.unsqueeze(0)).squeeze(0)
        augmented = torch.nan_to_num(augmented)
        augmented = torch.clamp(augmented, min=-1, max=1)
            
        if self.in_sr != self.out_sr:
            key = (self.out_sr, self.in_sr)
            if key not in self.resamplers:
                self.resamplers[key] = torchaudio.transforms.Resample(
                                        orig_freq=self.out_sr,
                                        new_freq=self.in_sr,
                                        resampling_method="sinc_interp_kaiser"
                                    )
            augmented = self.resamplers[key](augmented)

        return {
            'input_wav': augmented.squeeze(),
            'wav': wav.squeeze(),
        }
        
    def __len__(self):
        return len(self.files_list)

class ValidationDataset(Dataset):
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