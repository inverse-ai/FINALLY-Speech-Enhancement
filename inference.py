import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.nn.utils.weight_norm` is deprecated"
)

import os, sys
sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from utils.model_utils import setup_seed, requires_grad
from utils.data_utils import load_config, save_config, save_wavs_to_dir
import torch
import torch.nn as nn
from tqdm import tqdm
from metrics import metrics
from datasets_manager import inference_datasets, dataloaders
from transformers import WavLMModel
import numpy as np
from scipy import stats
from models.finally_models import FinallyGenerator

class inference:
    def __init__(self, config):
        self.config = config
        self.device = config.exp.device
        
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True)
        
        self.wavlm = wavlm.to(self.device)
        self.wavlm.eval()
        requires_grad(self.wavlm, False)
    
    def setup_inference(self):
        self.setup_experiment_dir()

        self.setup_models()

        self.setup_inf_metrics()

        self.setup_inference_dataset()
        self.setup_inference_dataloader()
        
    def setup_experiment_dir(self):
        self.exp_dir = self.config.exp.exp_dir

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            tqdm.write(f'Experiment directory \'{self.exp_dir}\' created.')

        self.inference_out_dir = os.path.join(self.exp_dir, 'inference_out')
        if not os.path.exists(self.inference_out_dir):
            os.makedirs(self.inference_out_dir)
            tqdm.write(f'Subdirectory \'{self.inference_out_dir}\' created.')

        tqdm.write('Experiment dir successfully initialized')
        
    def setup_models(self):
        checkpoint_path = self.config.gen.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Generator checkpoint path \'{checkpoint_path}\' does not exist.')
        self.model_gen = FinallyGenerator(**self.config.gen.args)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model_gen.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model_gen.to(self.device)
        self.model_gen.eval()
        tqdm.write('Generator model loaded for inference.')
        
    def setup_inf_metrics(self):
        self.metrics = []
        for metric_name in self.config.inference.metrics:
            metric = metrics.get_metric(metric_name, self.config)
            self.metrics.append((metric_name, metric))
        tqdm.write('Inference metrics successfully initialized')
        
    def setup_inference_dataset(self):
        self.inference_dataset = inference_datasets.InferenceDataset(
                self.config.data.inference_data_root,
                self.config.data.inference_data_file_path,
                self.config.mel,
                **self.config.data.dataset_args
            )
        tqdm.write('Dataset for inference successfully initialized')
        
    def setup_inference_dataloader(self):
        self.inference_dataloader = dataloaders.DataLoader(
            self.inference_dataset,
            batch_size=self.config.data.inference_batch_size,
            shuffle=False,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )
        tqdm.write('Inference dataloader successfully initialized')
        
    def _compute_metrics(self, batch, gen_batch, metrics_dict):
        for metric_name, metric in self.metrics:
            value = metric(batch, gen_batch)

            # If metric returns a dict, store each key separately
            if isinstance(value, dict):
                for k, v in value.items():
                    assert isinstance(v, float), f"Metric {k} must return float, got {type(v)}"
                    key = f'inf_{k}'
                    if key not in metrics_dict:
                        metrics_dict[key] = []
                    metrics_dict[key].append(v)
            else:
                assert isinstance(value, float), f"Each metric result must be float, but {metric_name} returned {type(value)}"
                key = f'inf_{metric_name}'
                if key not in metrics_dict:
                    metrics_dict[key] = []
                metrics_dict[key].append(value)
                
    def _avg_computed_metrics(self, metrics_dict):
        ci_dict = {}
        
        for key in metrics_dict.keys():
            values = np.array(metrics_dict[key])
            mean = np.mean(values)
            metrics_dict[key] = float(mean)

            if len(values) > 1:
                std = np.std(values, ddof=1)
                sem = std / np.sqrt(len(values))
                t = stats.t.ppf(0.975, df=len(values)-1)
                u = t * sem
            else:
                u = 0.0

            ci_dict[key] = float(u)

        return ci_dict

    def _print_metrics(self, metrics_dict, ci_dict):
        log_dir = os.path.join(self.inference_out_dir, self.config.exp.run_name)
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"inf_log.txt")

        with open(log_file, "a") as f:
            for key in metrics_dict.keys():
                mean = metrics_dict[key]
                u = ci_dict[key]
                line = f"{key}: {mean:.4f} ± {u:.4f}"
                
                tqdm.write(line)
                f.write(line + "\n")

    @torch.no_grad()
    def apply_wavlm(self, batch):
        wav = batch['input_wav'].to(self.device)
        outputs = self.wavlm(input_values=wav, output_hidden_states=True)
        raw_features = outputs.last_hidden_state.permute(0, 2, 1)
        return raw_features

    def synthesize_wavs(self, batch):
        gen = self.model_gen
        gen.eval()
        
        result_dict = {
            'gen_wav': [],
            'name': []
        }

        with torch.no_grad():
            for name, input_wav in zip(batch['name'], batch['input_wav']):
                input_wav = input_wav.to(self.device)[None,None]
                wavlm_features = self.apply_wavlm(batch)
                gen_wav = gen(input_wav, wavlm_features).squeeze()
                
                result_dict['gen_wav'].append(gen_wav)
                result_dict['name'].append(name)
        
        result_dict['gen_wav'] = torch.stack(result_dict['gen_wav'])
        return result_dict
    
    @torch.no_grad()
    def inference(self):
        metrics_dict = {}

        for batch in tqdm(self.inference_dataloader, desc=f"Inference Progress"):
            gen_batch = self.synthesize_wavs(batch)
            run_inf_dir = os.path.join(self.inference_out_dir, self.config.exp.run_name)

            in_sr = self.config.mel.in_sr
            out_sr = self.config.mel.out_sr
                
            if self.config.inference.save_samples:
                if 'input_wav' in batch:
                    save_wavs_to_dir(batch['input_wav'], batch['name'],
                                    os.path.join(run_inf_dir, 'input'), in_sr, 'wav')
                if 'clean_wav_orig' in batch:
                    save_wavs_to_dir(batch['clean_wav_orig'], batch['name'],
                                    os.path.join(run_inf_dir, 'ground_truth'), batch['clean_sr'], 'wav')

                pad = batch['pad'].item()
                if pad != 0:
                    gen_batch['gen_wav'] = gen_batch['gen_wav'][:, : -out_sr // in_sr * pad]                
                
                save_wavs_to_dir(gen_batch['gen_wav'], gen_batch['name'],
                                    os.path.join(run_inf_dir, 'generated'), out_sr, 'wav')

            self._compute_metrics(batch, gen_batch, metrics_dict)
        
        ci_dict = self._avg_computed_metrics(metrics_dict)

        tqdm.write('Inference completed:')
        self._print_metrics(metrics_dict, ci_dict)

if __name__ == "__main__":
    config = load_config()
    
    if config.gen.args.use_upsamplewaveunet:
        config.mel.out_sr = 48000
    else:
        config.mel.out_sr = 16000
        
    config.mel.segment_size = config.mel.segment_size * (config.mel.out_sr // config.mel.in_sr)

    conf_cli = OmegaConf.from_cli()
    config = OmegaConf.merge(config, conf_cli)

    exp_dir = getattr(config.exp, "exp_dir", ".")
    run_dir = os.path.join(exp_dir, "inference_out", config.exp.run_name)
    
    os.makedirs(run_dir, exist_ok=True)
    save_config(config, run_dir)

    setup_seed(config.exp.seed)

    enhacher = inference(config)
    enhacher.setup_inference()
    enhacher.inference()
