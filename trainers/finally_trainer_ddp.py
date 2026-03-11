"""
DDP-enabled FINALLY Trainer

Key features:
- Uses no_sync() context manager for gradient accumulation with DDP
- Proper gradient synchronization only on last accumulation step
- World-size aware effective batch size calculation
"""

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext
from tqdm import tqdm

from utils.model_utils import requires_grad
from trainers.base_trainer_ddp import BaseTrainerDDP
from transformers import WavLMModel
import os


class FinallyBaseTrainerDDP(BaseTrainerDDP):
    """Base DDP trainer with WavLM integration."""

    def __init__(self, config):
        super().__init__(config)

        # Load WavLM (not wrapped with DDP - it's frozen)
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True)
        self.wavlm = wavlm.to(self.device)
        self.wavlm.eval()
        requires_grad(self.wavlm, False)

        if self.is_main_process:
            tqdm.write('WavLM model loaded (frozen, not wrapped with DDP)')

    @torch.no_grad()
    def apply_wavlm(self, batch):
        """Extract WavLM features."""
        wav = batch['input_wav'].to(self.device)
        outputs = self.wavlm(input_values=wav, output_hidden_states=True)
        raw_features = outputs.last_hidden_state.permute(0, 2, 1)
        return raw_features

    def synthesize_wavs(self, batch):
        """Generate waveforms for validation."""
        # Get underlying model if wrapped with DDP
        gen = self.model_gen.module if isinstance(self.model_gen, DDP) else self.model_gen
        gen.eval()

        result_dict = {
            'gen_wav': [],
            'name': []
        }

        with torch.no_grad():
            for name, input_wav in zip(batch['name'], batch['input_wav']):
                input_wav = input_wav.to(self.device)[None, None]
                batch_single = {'input_wav': input_wav.squeeze(1)}
                wavlm_features = self.apply_wavlm(batch_single)
                gen_wav = gen(input_wav, wavlm_features).squeeze()

                result_dict['gen_wav'].append(gen_wav)
                result_dict['name'].append(name)

        result_dict['gen_wav'] = torch.stack(result_dict['gen_wav'])
        return result_dict


class FinallyTrainerStageOneDDP(FinallyBaseTrainerDDP):
    """Stage 1 DDP trainer (generator only, no discriminator)."""

    def __init__(self, config):
        super().__init__(config)

        self.sub_batch_size = config.data.train_batch_size

        # Calculate effective batch size and accumulation steps
        if hasattr(config.train, 'effective_batch_size'):
            self.effective_batch_size = config.train.effective_batch_size
        else:
            self.effective_batch_size = config.data.train_batch_size * 32  # Default

        # Accumulation steps accounts for world_size
        self.accum_steps = self.effective_batch_size // (self.sub_batch_size * self.world_size)

        if self.is_main_process:
            print(f"DDP Stage 1 Training Config:")
            print(f"  - World size: {self.world_size}")
            print(f"  - Batch per GPU: {self.sub_batch_size}")
            print(f"  - Accumulation steps: {self.accum_steps}")
            print(f"  - Effective batch size: {self.sub_batch_size * self.world_size * self.accum_steps}")

    def _get_no_sync_context(self, model, is_last_step):
        """Get no_sync context for gradient accumulation."""
        if not self.is_distributed:
            return nullcontext()

        if is_last_step:
            # Last step: allow gradient sync
            return nullcontext()
        else:
            # Intermediate steps: disable gradient sync
            return model.no_sync()

    def train_step(self):
        """Training step with gradient accumulation and DDP."""
        gen = self.model_gen
        gen_optimizer = self.optimizer_gen
        gen_loss_builder = self.gen_loss_builder

        gen_optimizer.zero_grad()
        gen_losses_dict_accum = {}

        for i in range(self.accum_steps):
            # Get batch
            batch = self._get_train_batch()
            real_wav = batch['wav'].to(self.device).unsqueeze(1)
            input_wav = batch['input_wav']

            # Compute WavLM features
            batch_sub = {'input_wav': input_wav}
            wavlm_features = self.apply_wavlm(batch_sub)
            input_wav_gpu = input_wav.to(self.device).unsqueeze(1)

            # Determine if this is the last accumulation step
            is_last_step = (i == self.accum_steps - 1)

            # Use no_sync for intermediate steps
            with self._get_no_sync_context(gen, is_last_step):
                gen_out = gen(input_wav_gpu, wavlm_features)

                gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
                    '': dict(
                        gen_wav=gen_out,
                        real_wav=real_wav,
                        wavlm=self.wavlm,
                        sr=self.config.mel.out_sr
                    )
                }, tl_suffix='gen')

                # Scale loss by accumulation steps
                (gen_loss / self.accum_steps).backward()

            # Accumulate losses for logging
            for k, v in gen_losses_dict.items():
                gen_losses_dict_accum[k] = gen_losses_dict_accum.get(k, 0.0) + v

        # Update weights (gradients are now synced across GPUs)
        gen_optimizer.step()

        # Average losses
        gen_losses_dict = {k: v / self.accum_steps for k, v in gen_losses_dict_accum.items()}

        return gen_losses_dict


class FinallyTrainerDDP(FinallyBaseTrainerDDP):
    """Stage 2/3 DDP trainer (GAN with generator and discriminator)."""

    def __init__(self, config, n_disc_iters=2):
        super().__init__(config)

        self.n_disc_iters = n_disc_iters
        self.sub_batch_size = config.data.train_batch_size

        # Calculate effective batch size and accumulation steps
        if hasattr(config.train, 'effective_batch_size'):
            self.effective_batch_size = config.train.effective_batch_size
        else:
            self.effective_batch_size = config.data.train_batch_size * 32

        # Accumulation steps accounts for world_size
        self.accum_steps = self.effective_batch_size // (self.sub_batch_size * self.world_size)

        # Override n_disc_iters from config if specified
        if 'disc' in config and 'n_disc_iters' in config.disc:
            self.n_disc_iters = config.disc.n_disc_iters

        if self.is_main_process:
            print(f"DDP Stage 2/3 Training Config:")
            print(f"  - World size: {self.world_size}")
            print(f"  - Batch per GPU: {self.sub_batch_size}")
            print(f"  - Accumulation steps: {self.accum_steps}")
            print(f"  - Effective batch size: {self.sub_batch_size * self.world_size * self.accum_steps}")
            print(f"  - Discriminator iterations: {self.n_disc_iters}")

    def _get_no_sync_context(self, model, is_last_step):
        """Get no_sync context for gradient accumulation."""
        if not self.is_distributed:
            return nullcontext()

        if is_last_step:
            return nullcontext()
        else:
            return model.no_sync()

    def train_step(self):
        """Training step with GAN, gradient accumulation, and DDP.

        Note: We don't use no_sync() here because GAN training with alternating
        disc/gen updates is complex. DDP syncs on every backward() call.
        This is slightly less efficient but more reliable.
        """
        gen = self.model_gen
        gen_optimizer = self.optimizer_gen
        gen_loss_builder = self.gen_loss_builder
        disc = self.model_disc
        disc_optimizer = self.optimizer_disc
        disc_loss_builder = self.disc_loss_builder

        # Initialize loss accumulators
        disc_losses_dict_accum = {}
        gen_losses_dict_accum = {}

        # ========== DISCRIMINATOR TRAINING ==========
        requires_grad(disc, True)
        disc_optimizer.zero_grad()

        for disc_iter in range(self.n_disc_iters):
            for i in range(self.accum_steps):
                # Get batch
                batch = self._get_train_batch()
                real_wav = batch['wav'].to(self.device).unsqueeze(1)
                input_wav = batch['input_wav']

                # Compute WavLM features
                batch_sub = {'input_wav': input_wav}
                wavlm_features = self.apply_wavlm(batch_sub)
                input_wav_gpu = input_wav.to(self.device).unsqueeze(1)

                # Generate fake samples (no gradient needed for disc training)
                with torch.no_grad():
                    # Access underlying model if DDP wrapped
                    gen_model = gen.module if isinstance(gen, DDP) else gen
                    gen_wav = gen_model(input_wav_gpu, wavlm_features)

                # Discriminator forward/backward
                disc_real_out, _ = disc(real_wav)
                disc_gen_out, _ = disc(gen_wav)

                disc_loss, disc_losses_dict = disc_loss_builder.calculate_loss({
                    '': dict(
                        discs_real_out=disc_real_out,
                        discs_gen_out=disc_gen_out
                    )
                }, tl_suffix='ms-stft')

                # Scale by total steps
                total_disc_steps = self.n_disc_iters * self.accum_steps
                (disc_loss / total_disc_steps).backward()

                # Accumulate losses
                for k, v in disc_losses_dict.items():
                    disc_losses_dict_accum[k] = disc_losses_dict_accum.get(k, 0.0) + v

        # Update discriminator
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=10.0)
        disc_optimizer.step()

        # ========== GENERATOR TRAINING ==========
        requires_grad(disc, False)
        gen_optimizer.zero_grad()

        for i in range(self.accum_steps):
            # Get batch
            batch = self._get_train_batch()
            real_wav = batch['wav'].to(self.device).unsqueeze(1)
            input_wav = batch['input_wav']

            # Compute WavLM features
            batch_sub = {'input_wav': input_wav}
            wavlm_features = self.apply_wavlm(batch_sub)
            input_wav_gpu = input_wav.to(self.device).unsqueeze(1)

            # Generator forward/backward
            gen_wav = gen(input_wav_gpu, wavlm_features)

            disc_real_out, disc_fmaps_real = disc(real_wav)
            disc_gen_out, disc_fmaps_gen = disc(gen_wav)

            gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
                '': dict(
                    gen_wav=gen_wav,
                    real_wav=real_wav,
                    wavlm=self.wavlm,
                    sr=self.config.mel.out_sr
                ),
                'ms-stft': dict(
                    discs_gen_out=disc_gen_out,
                    fmaps_real=disc_fmaps_real,
                    fmaps_gen=disc_fmaps_gen
                ),
            }, tl_suffix='gen')

            # Scale by accumulation steps
            (gen_loss / self.accum_steps).backward()

            # Accumulate losses
            for k, v in gen_losses_dict.items():
                gen_losses_dict_accum[k] = gen_losses_dict_accum.get(k, 0.0) + v

        # Update generator
        gen_optimizer.step()

        # Average losses for logging
        total_disc_steps = self.n_disc_iters * self.accum_steps
        disc_losses_dict = {k: v / total_disc_steps for k, v in disc_losses_dict_accum.items()}
        gen_losses_dict = {k: v / self.accum_steps for k, v in gen_losses_dict_accum.items()}

        return {**gen_losses_dict, **disc_losses_dict}
