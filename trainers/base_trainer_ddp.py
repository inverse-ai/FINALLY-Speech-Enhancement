"""
DDP-enabled Base Trainer for FINALLY

This extends the base trainer with Distributed Data Parallel support.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from trainers.base_trainer import BaseTrainer
from models.finally_models import FinallyGenerator, MultiScaleSTFTDiscriminator


class BaseTrainerDDP(BaseTrainer):
    """Base trainer with DDP support."""

    def __init__(self, config):
        super().__init__(config)

        # Get distributed info from config
        if hasattr(config, 'distributed') and config.distributed.enabled:
            self.rank = config.distributed.rank
            self.local_rank = config.distributed.local_rank
            self.world_size = config.distributed.world_size
            self.is_main_process = config.distributed.is_main_process
            self.is_distributed = True
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.is_main_process = True
            self.is_distributed = False

        if self.is_main_process:
            print(f"DDP Trainer initialized: world_size={self.world_size}, "
                  f"is_distributed={self.is_distributed}")

    def _create_model(self, model_config, model_type):
        """Create model and optionally wrap with DDP."""
        if model_type == 'gen':
            model = FinallyGenerator(**model_config['args'])
        elif model_type == 'disc':
            model = MultiScaleSTFTDiscriminator(**model_config['args'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint BEFORE wrapping with DDP
        checkpoint_path = model_config.get('checkpoint_path')
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            if self.is_main_process:
                tqdm.write(f'Loading checkpoint for {model_type} from {checkpoint_path}...')
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            if self.is_main_process:
                if checkpoint_path:
                    tqdm.write(f'Warning: Checkpoint not found at {checkpoint_path}. '
                               f'Initializing {model_type} from scratch.')
                else:
                    tqdm.write(f'No checkpoint specified for {model_type}. Initializing from scratch.')

        # Move to device
        model = model.to(self.device)

        # Wrap with DDP if distributed
        if self.is_distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True  # Required: some params don't receive grads
            )
            if self.is_main_process:
                tqdm.write(f'{model_type} wrapped with DDP')

        return model

    def setup_train_dataloader(self):
        """Setup training dataloader with DistributedSampler for DDP."""
        self.train_sampler = None

        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )

        # Use regular DataLoader instead of InfiniteLoader for DDP
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            drop_last=True,
            num_workers=self.config.data.workers,
            pin_memory=True,
            persistent_workers=True if self.config.data.workers > 0 else False
        )

        # Create an infinite iterator
        self._train_iter = None
        self._epoch = 0

        if self.is_main_process:
            tqdm.write(f'Train dataloader initialized (distributed={self.is_distributed})')

    def _get_train_batch(self):
        """Get next training batch, handling epoch boundaries for DDP."""
        if self._train_iter is None:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self._epoch)
            self._train_iter = iter(self.train_dataloader)

        try:
            batch = next(self._train_iter)
        except StopIteration:
            # Start new epoch
            self._epoch += 1
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self._epoch)
            self._train_iter = iter(self.train_dataloader)
            batch = next(self._train_iter)

        return batch

    def setup_experiment_dir(self):
        """Setup experiment directory (only on main process)."""
        if self.is_main_process:
            super().setup_experiment_dir()

        # Sync all processes
        if self.is_distributed:
            dist.barrier()

        # Set paths on all processes
        self.exp_dir = self.config.exp.exp_dir
        self.inference_out_dir = os.path.join(self.exp_dir, 'inference_out')
        self.checkpoints_dir = os.path.join(self.exp_dir, 'checkpoints')

    def save_checkpoint(self):
        """Save checkpoint (only on main process)."""
        if not self.is_main_process:
            # Wait for main process to finish saving
            if self.is_distributed:
                dist.barrier()
            return

        model_names = ['gen']
        if 'disc' in self.config:
            model_names.append('disc')

        for model_name in model_names:
            try:
                dir_path = os.path.join(self.checkpoints_dir, self.config.exp.run_name, model_name)
                os.makedirs(dir_path, exist_ok=True)

                if model_name == 'gen':
                    model = self.model_gen
                    optimizer = self.optimizer_gen
                elif model_name == 'disc':
                    model = self.model_disc
                    optimizer = self.optimizer_disc

                # Get state dict from DDP module if wrapped
                if isinstance(model, DDP):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()

                checkpoint = {
                    'state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': self.step,
                    'epoch': self._epoch
                }

                path = os.path.join(dir_path, f'{model_name}_checkpoint_{self.step}_{self.config.exp.run_name}.pth')
                torch.save(checkpoint, path)
                tqdm.write(f'Checkpoint for {model_name} on step {self.step} saved to {path}')

            except Exception as e:
                tqdm.write(f'Error saving checkpoint for {model_name} on step {self.step}: {e}')

        # Sync after saving
        if self.is_distributed:
            dist.barrier()

    def validate(self):
        """Run validation (only on main process)."""
        if not self.is_main_process:
            # Wait for main process to finish validation
            if self.is_distributed:
                dist.barrier()
            return

        super().validate()

        # Sync after validation
        if self.is_distributed:
            dist.barrier()

    def training_loop(self):
        """Training loop with DDP support."""
        # Only show progress bar on main process
        disable_tqdm = not self.is_main_process

        with tqdm(total=self.config.train.steps, desc='Training Progress',
                  unit='step', disable=disable_tqdm) as progress:
            progress.update(self.step - 1)

            for self.step in range(self.start_step, self.config.train.steps + 1):
                self.to_train()

                losses_dict = self.train_step()
                self.step_schedulers()

                if self.is_main_process:
                    losses_to_save = {
                        k: (v.item() if isinstance(v, torch.Tensor) else v)
                        for k, v in losses_dict.items()
                    }

                    progress.set_postfix({
                        'gen_lr': self.optimizer_gen.param_groups[0]['lr'],
                        'disc_lr': self.optimizer_disc.param_groups[0]['lr'] if 'disc' in self.config else 'N/A'
                    } | {
                        "gen_loss": f"{losses_to_save.get('total_loss_gen', 0):.8f}",
                        "disc_loss": f"{losses_to_save.get('total_loss_ms-stft', 0):.8f}" if 'disc' in self.config else 'N/A'
                    })
                    progress.update(1)

                    # Log losses
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

    def _log(self, message):
        """Log message (only on main process)."""
        if self.is_main_process:
            tqdm.write(message)
