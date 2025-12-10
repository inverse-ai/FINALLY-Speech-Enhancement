import os
import subprocess
import torch
import numpy as np
import torch.nn as nn
from autoclip.torch import QuantileClip

from scipy import stats
from abc import abstractmethod

from datasets_manager.dataloaders import InfiniteLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.finally_models import FinallyGenerator, MultiScaleSTFTDiscriminator
from trainers.optimizers import AdamW_
from trainers.schedulers import ExponentialScheduler
from datasets_manager.datasets import TrainingDataset, ValidationDataset
from trainers.loss_builder import LossBuilder
from metrics import metrics

class BaseTrainerHelpers:
    def __init__(self, config):
        self.config = config
        self.device = config.exp.device
        self.step=1
        if 'train' in config:
            self.start_step = config.train.start_step
            self.step = self.start_step
        self.multi_gpu = False
    
    def _create_model(self, model_config, model_type):
        if model_type == 'gen':
            model = FinallyGenerator(**model_config['args'])
        elif model_type == 'disc':
            model = MultiScaleSTFTDiscriminator(**model_config['args'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        checkpoint_path = model_config.get('checkpoint_path')
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            tqdm.write(f'Loading checkpoint for {model_type} from {checkpoint_path}...')
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            if checkpoint_path:
                tqdm.write(f'Warning: Checkpoint not found at {checkpoint_path}. Initializing {model_type} from scratch.')
            else:
                tqdm.write(f'No checkpoint specified for {model_type}. Initializing from scratch.')

        model = model.to(self.device)

        return model

    def _create_optimizer(self, model_config, model_type):
        if model_type == 'gen':
            optimizer = AdamW_(self.model_gen.parameters(), **model_config['optimizer']['args'])
        elif model_type == 'disc':
            optimizer = AdamW_(self.model_disc.parameters(), **model_config['optimizer']['args'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        load_from_checkpoint = True
        if 'load_optimizer_from_checkpoint' in model_config \
            and not model_config.load_optimizer_from_checkpoint:
            tqdm.write(
                f'load_optimizer_from_checkpoint for model {model_type} set to false, ' +
                'initializing optimizer from scratch'
            )
            load_from_checkpoint = False
        
        if load_from_checkpoint:
            checkpoint_path = model_config.get('checkpoint_path')

            if checkpoint_path is not None and os.path.isfile(checkpoint_path):
                tqdm.write(f'Loading optimizer state for {model_type} from {checkpoint_path}...')
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict']['optimizer'])
                    except Exception as e:
                        tqdm.write(f'An error occured when loading checkpoint for optimizer for {model_type}: {e}')
                else:
                    tqdm.write(f'Warning: optimizer_state_dict not found in {checkpoint_path}. '
                               + 'Starting fresh optimizer for {model_type}.')

        clip_args = {
            'quantile': 1.0
        }
        
        if 'clip_quantile' in self.config.train:
            clip_args['quantile'] = self.config.train.clip_quantile
            tqdm.write(f'Clip quantile set to {clip_args["quantile"]} for {model_type}')
        if 'clip_history' in self.config.train:
            clip_args['history'] = self.config.train.clip_history
            tqdm.write(f'Clip history set to {clip_args["history"]} for {model_type}')

        return QuantileClip.as_optimizer(
                    optimizer=optimizer,
                    global_threshold=True,
                    **clip_args
                )

    def _create_scheduler(self, model_config, model_type):
        if model_type == 'gen':
            return ExponentialScheduler(self.optimizer_gen, self.start_step, **model_config['scheduler']['args'])
        elif model_type == 'disc':
            return ExponentialScheduler(self.optimizer_disc, self.start_step, **model_config['scheduler']['args'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    def _compute_metrics(self, batch, gen_batch, metrics_dict, action):
        for metric_name, metric in self.metrics:
            value = metric(batch, gen_batch)

            # If metric returns a dict, store each key separately
            if isinstance(value, dict):
                for k, v in value.items():
                    assert isinstance(v, float), f"Metric {k} must return float, got {type(v)}"
                    key = f'{action}_{k}'
                    if key not in metrics_dict:
                        metrics_dict[key] = []
                    metrics_dict[key].append(v)
            else:
                assert isinstance(value, float), f"Each metric result must be float, but {metric_name} returned {type(value)}"
                key = f'{action}_{metric_name}'
                if key not in metrics_dict:
                    metrics_dict[key] = []
                metrics_dict[key].append(value)

    
    def _avg_computed_metrics(self, metrics_dict, action):
        ci_dict = {}
        
        for key in metrics_dict.keys():
            if not key.startswith(f"{action}_"):
                continue
            
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

    def _print_metrics(self, metrics_dict, ci_dict, action='val', steps=None):
        if action == 'val':
            log_dir = os.path.join(self.checkpoints_dir, self.config.exp.run_name)
        else:
            log_dir = os.path.join(self.inference_out_dir, self.config.exp.run_name)
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{action}_log.txt")

        with open(log_file, "a") as f:
            if action == 'val':
                print("steps:", steps)
            if steps is not None:
                f.write(f"After {steps} steps:\n")

            for key in metrics_dict.keys():
                mean = metrics_dict[key]
                u = ci_dict[key]
                line = f"{key}: {mean:.4f} ± {u:.4f}"
                
                tqdm.write(line)
                f.write(line + "\n")

class BaseTrainer(BaseTrainerHelpers):
    def __init__(self, config):
        super().__init__(config)

    def setup_training(self):
        self.setup_experiment_dir()

        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()

        self.setup_val_metrics()
        
        self.setup_train_dataset()
        self.setup_val_dataset()
        
        self.setup_trainval_dataloaders()

    def setup_experiment_dir(self):
        self.exp_dir = self.config.exp.exp_dir

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            tqdm.write(f'Experiment directory \'{self.exp_dir}\' created.')

        self.inference_out_dir = os.path.join(self.exp_dir, 'inference_out')
        self.checkpoints_dir = os.path.join(self.exp_dir, 'checkpoints')

        for dir_path in [self.inference_out_dir, self.checkpoints_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                tqdm.write(f'Subdirectory \'{dir_path}\' created.')

        tqdm.write('Experiment dir successfully initialized')
    
    def setup_models(self):
        self.model_gen = self._create_model(self.config.gen, 'gen')
        tqdm.write(f'Generator successfully initialized')
        if 'disc' in self.config:
            self.model_disc = self._create_model(self.config.disc, 'disc')
            tqdm.write(f'Discriminator successfully initialized')

    def setup_optimizers(self):
        self.optimizer_gen = self._create_optimizer(self.config.gen, 'gen')
        self.scheduler_gen = self._create_scheduler(self.config.gen, 'gen')
        tqdm.write(f'Optimizers and schedulers for generator successfully initialized')
        if 'disc' in self.config:
            self.optimizer_disc = self._create_optimizer(self.config.disc, 'disc')
            self.scheduler_disc = self._create_scheduler(self.config.disc, 'disc')
            tqdm.write(f'Optimizers and schedulers for discriminator successfully initialized')

    def setup_losses(self):
        self.gen_loss_builder = LossBuilder(self.device, self.config.gen['losses'])
        tqdm.write(f'Loss functions for generator successfully initialized')
        if 'disc' in self.config:
            self.disc_loss_builder = LossBuilder(self.device, self.config.disc['losses'])
            tqdm.write(f'Loss functions for discriminator successfully initialized')

    def setup_val_metrics(self):
        self.metrics = []
        for metric_name in self.config.train.val_metrics:
            metric = metrics.get_metric(metric_name, self.config)
            self.metrics.append((metric_name, metric))
        tqdm.write('Validation metrics successfully initialized')

    def setup_train_dataset(self):
        self.train_dataset = TrainingDataset(
                self.config.data.train_data_root,
                self.config.data.train_data_file_path,
                self.config.mel,
                **self.config.data.train_dataset_args
            )
        tqdm.write('Train dataset successfully initialized')

    def setup_val_dataset(self):
        self.val_dataset = ValidationDataset(
                self.config.data.val_data_root,
                self.config.data.val_data_file_path,
                self.config.mel,
                **self.config.data.val_dataset_args
        )
        tqdm.write('Validation dataset successfully initialized')

    def setup_trainval_datasets(self):
        self.train_dataset = TrainingDataset(
                self.config.data.trainval_data_root,
                self.config.data.train_data_file_path,
                self.config.mel,
                **self.config.data.dataset_args
            )

        self.val_dataset = ValidationDataset(
                self.config.data.trainval_data_root,
                self.config.data.val_data_file_path,
                self.config.mel,
                **self.config.data.dataset_args
        )
        tqdm.write('Datasets for train and validation successfully initialized')
    
    def setup_train_dataloader(self):
        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )
        tqdm.write('Train dataloader successfully initialized')

    def setup_val_dataloader(self):
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.data.workers,
            pin_memory=True,
        )
        tqdm.write('Val dataloader successfully initialized')

    def setup_trainval_dataloaders(self):
        self.setup_train_dataloader()
        self.setup_val_dataloader()

    def to_train(self):
        self.model_gen.train()
        if 'disc' in self.config:
            self.model_disc.train()

    def to_eval(self):
        self.model_gen.eval()

    def step_schedulers(self):
        self.scheduler_gen.step()
           
        if 'disc' in self.config:
            self.scheduler_disc.step()

    def training_loop(self):
        with tqdm(total=self.config.train.steps, desc='Training Progress', unit='step') as progress:
            progress.update(self.step - 1)
            for self.step in range(self.start_step, self.config.train.steps + 1):
                self.to_train()
                
                losses_dict = self.train_step()
                self.step_schedulers()
                
                losses_to_save = {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in losses_dict.items()
                }

                progress.set_postfix({
                    'gen_lr': self.optimizer_gen.param_groups[0]['lr'],
                    'disc_lr': self.optimizer_disc.param_groups[0]['lr'] if 'disc' in self.config else 'N/A'
                } | {
                    "gen_loss": f"{losses_to_save['total_loss_gen']:.8f}",
                    "disc_loss": f"{losses_to_save['total_loss_ms-stft']:.8f}" if 'disc' in self.config else 'N/A'
                })
                progress.update(1)
                
                # create directory if it doesn't exist
                log_dir = os.path.join(self.checkpoints_dir, self.config.exp.run_name)
                os.makedirs(log_dir, exist_ok=True)
                loss_log_path = os.path.join(log_dir, "loss_log.txt")
                with open(loss_log_path, "a") as f:
                    loss_str = ", ".join(f"{k}={v:.8f}" for k, v in losses_to_save.items())
                    f.write(f"Step {self.step}: {loss_str}\n")
               
                if self.step % self.config.train.checkpoint_step == 0:
                    self.save_checkpoint()

                if self.step % self.config.train.val_step == 0:
                    self.validate()

    
    def save_checkpoint(self):
        model_names = ['gen']
        if 'disc' in self.config:
            model_names.append('disc')
        for model_name in model_names:
            try:
                dir_path = os.path.join(self.checkpoints_dir, self.config.exp.run_name, model_name)
                os.makedirs(dir_path, exist_ok=True)
                if model_name == 'gen':
                    model_to_save = self.model_gen
                    optimizer_to_save = self.optimizer_gen
                elif model_name == 'disc':
                    model_to_save = self.model_disc
                    optimizer_to_save = self.optimizer_disc
                checkpoint = {
                    'state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer_to_save.state_dict()
                }

                path = os.path.join(dir_path, f'{model_name}_checkpoint_{self.step}_{self.config.exp.run_name}.pth')
                torch.save(checkpoint, path)

                tqdm.write(f'Checkpoint for {model_name} on step {self.step} saved to {path}')
            except Exception as e:
                tqdm.write(f'An error occured when saving checkpoint for {model_name} on step {self.step}: {e}')

    @torch.no_grad()
    def validate(self):
        self.to_eval()
        metrics_dict = {}
        in_sr = self.config.mel.in_sr
        out_sr = self.config.mel.out_sr

        for batch in tqdm(self.val_dataloader, desc=f"Validating Progress on {self.step}"):
            gen_batch = self.synthesize_wavs(batch)
            pad = batch['pad'].item()
            if pad != 0:
                gen_batch['gen_wav'] = gen_batch['gen_wav'][:, : -(out_sr // in_sr) * pad]
            self._compute_metrics(batch, gen_batch, metrics_dict, action='val')

        ci_dict = self._avg_computed_metrics(metrics_dict, action='val')

        tqdm.write('Validation completed:')
        self._print_metrics(metrics_dict, ci_dict, action='val', steps=self.step)
        
        run_dir = os.path.join(self.checkpoints_dir, self.config.exp.run_name)
        val_log_file = os.path.join(run_dir, "val_log.txt")

        subprocess.run([
            "python", "utils/plot.py",
            "--file", val_log_file,
            "--step", str(self.config.train.val_step)
        ], check=True)
    
    @abstractmethod
    def synthesize_wavs(self, batch):
        pass

    @abstractmethod
    def train_step(self):
        pass
