import torch
from torch import nn
import tqdm
from utils.model_utils import requires_grad
from trainers.base_trainer import BaseTrainer
from transformers import WavLMModel
import os

class FinallyBaseTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True)

        self.wavlm = wavlm.to(self.device)
        self.wavlm.eval()
        requires_grad(self.wavlm, False)

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

class FinallyTrainerStageOne(FinallyBaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        self.batch_size = config.data.train_batch_size
        if 'train' in config and 'sub_batch_size' in config.train:
            self.sub_batch_size = config.train.sub_batch_size
        else:
            self.sub_batch_size = self.batch_size
        
        if self.batch_size and self.batch_size % self.sub_batch_size != 0:
            tqdm.write('Warning: sub_batch_size do not divide train_batch size.' \
            'So the total batch size with grad accumulation could be smaller than you think.')

    def train_step(self):
        gen = self.model_gen
        gen_optimizer = self.optimizer_gen
        gen_loss_builder = self.gen_loss_builder
        batch = next(self.train_dataloader)
        real_wav = batch['wav'].to(self.device).unsqueeze(1)
        input_wav_full = batch['input_wav']  # Keep on CPU
        
        accum_steps = self.batch_size // self.sub_batch_size
        
        gen_optimizer.zero_grad()
        gen_losses_dict_accum = {}
        
        for i in range(accum_steps):
            start = i * self.sub_batch_size
            end = (i + 1) * self.sub_batch_size
            real_sub = real_wav[start:end]
            
            # Create sub-batch for wavlm
            batch_sub = {'input_wav': input_wav_full[start:end]}
            wavlm_sub = self.apply_wavlm(batch_sub)

            # Get corresponding input_wav slice (moved to device and unsqueezed)
            input_sub = input_wav_full[start:end].to(self.device).unsqueeze(1)
            
            gen_sub = gen(input_sub, wavlm_sub)
            
            gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
                '': dict(
                    gen_wav=gen_sub,
                    real_wav=real_sub,
                    wavlm=self.wavlm,
                    sr = self.config.mel.out_sr
                )
            }, tl_suffix='gen')
            
            for k, v in gen_losses_dict.items():
                gen_losses_dict_accum[k] = gen_losses_dict_accum.get(k, 0.0) + v
            
            (gen_loss / accum_steps).backward()
        
        gen_optimizer.step()
        gen_losses_dict = {k: v / accum_steps for k, v in gen_losses_dict_accum.items()}
        
        return {**gen_losses_dict}
    
class FinallyTrainer(FinallyBaseTrainer):
    def __init__(
            self,
            config,
            n_disc_iters=2
        ):
        super().__init__(config)

        self.n_disc_iters = n_disc_iters
        self.batch_size = config.data.train_batch_size
        if 'train' in config and 'sub_batch_size' in config.train:
            self.sub_batch_size = config.train.sub_batch_size
        else:
            self.sub_batch_size = self.batch_size
        
        if self.batch_size and self.batch_size % self.sub_batch_size != 0:
            tqdm.write('Warning: sub_batch_size do not divide train_batch size.' \
            'So the total batch size with grad accumulation could be smaller than you think.')

    def train_step(self):

        gen = self.model_gen
        gen_optimizer = self.optimizer_gen
        gen_loss_builder = self.gen_loss_builder
        disc = self.model_disc
        disc_optimizer = self.optimizer_disc
        disc_loss_builder = self.disc_loss_builder

        batch = next(self.train_dataloader)
        real_wav = batch['wav'].to(self.device).unsqueeze(1)
        input_wav = batch['input_wav'].to(self.device).unsqueeze(1)

        accum_steps = self.batch_size // self.sub_batch_size
        
        wavlm_features = self.apply_wavlm(batch)

        requires_grad(disc, True)
        detached_gen_wav_list = []
        for disc_idx in range(self.n_disc_iters):
            disc_optimizer.zero_grad()
            disc_losses_dict_accum = {}
            
            for i in range(accum_steps):
                start = i * self.sub_batch_size
                end = (i + 1) * self.sub_batch_size
                if disc_idx == 0:
                    gen_sub = gen(input_wav[start:end], wavlm_features[start:end]).detach()
                    detached_gen_wav_list.append(gen_sub)
                real_sub = real_wav[start:end]
                gen_sub = detached_gen_wav_list[i]

                disc_real_out, _, = disc(real_sub)
                disc_gen_out, _, = disc(gen_sub)

                disc_loss, disc_losses_dict = disc_loss_builder.calculate_loss({
                    '': dict(
                        discs_real_out=disc_real_out,
                        discs_gen_out=disc_gen_out
                    )
                }, tl_suffix='ms-stft')

                for k, v in disc_losses_dict.items():
                    disc_losses_dict_accum[k] = disc_losses_dict_accum.get(k, 0.0) + v
                
                (disc_loss / accum_steps).backward()

            disc_optimizer.step()
            gen_losses_dict = {k: v / accum_steps for k, v in disc_losses_dict_accum.items()}

        requires_grad(disc, False)
        gen_optimizer.zero_grad()

        gen_losses_dict_accum = {}

        for i in range(accum_steps):
            start = i * self.sub_batch_size
            end = (i + 1) * self.sub_batch_size
            real_sub = real_wav[start:end]
            gen_sub = gen(input_wav[start:end], wavlm_features[start:end])

            disc_real_out, disc_fmaps_real = disc(real_sub)
            disc_gen_out, disc_fmaps_gen = disc(gen_sub.detach())
            
            # Concatenate all tensors along batch dimension
            all_real = torch.cat([x.view(x.size(0), -1) for x in disc_real_out], dim=1)
            all_gen  = torch.cat([x.view(x.size(0), -1) for x in disc_gen_out], dim=1)

            # Compute mean across everything
            mean_real = all_real.mean()
            mean_gen  = all_gen.mean()

            # Ensure directory exists
            log_dir = os.path.join(self.checkpoints_dir, self.config.exp.run_name)
            os.makedirs(log_dir, exist_ok=True)

            # Save file
            log_file = os.path.join(log_dir, "disc_log.txt")
            with open(log_file, "a") as f:
                f.write(f"{mean_real.item():.6f} {mean_gen.item():.6f}\n")

            gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
                '': dict(
                    gen_wav=gen_sub,
                    real_wav=real_sub,
                    wavlm=self.wavlm,
                    sr = self.config.mel.out_sr
                ),
                'ms-stft': dict(
                    discs_gen_out=disc_gen_out,
                    fmaps_real=disc_fmaps_real,
                    fmaps_gen=disc_fmaps_gen
                ),
            }, tl_suffix='gen')            

            for k, v in gen_losses_dict.items():
                gen_losses_dict_accum[k] = gen_losses_dict_accum.get(k, 0.0) + v

            (gen_loss / accum_steps).backward()
            
        gen_optimizer.step()

        gen_losses_dict = {k: v / accum_steps for k, v in gen_losses_dict_accum.items()}

        return {**gen_losses_dict, **disc_losses_dict}
