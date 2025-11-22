import torch
from utils.data_utils import mel_spectrogram, debug_msg
from utils.model_utils import requires_grad, closest_power_of_two
from training.trainers.base_trainer import BaseTrainer

from training.trainers.base_trainer import gan_trainers_registry

@gan_trainers_registry.add_to_registry(name='hifi++_trainer')
class HifiPlusPlusTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.gen_name = "a2a_hifi++_gen"
        self.ssd_name = "msd"
        self.mel_kwargs = dict(
            n_fft=config.mel.n_fft,
            num_mels=config.mel.num_mels,
            sampling_rate=config.mel.sampling_rate,
            hop_size=config.mel.hop_size,
            win_size=config.mel.win_size,
            fmin=config.mel.fmin,
            fmax=config.mel.fmax_for_loss
        )

    def train_step(self):
        gen = self.models[self.gen_name]
        gen_optimizer = self.optimizers[self.gen_name]
        gen_loss_builder = self.loss_builders[self.gen_name]
        ssd = self.models[self.ssd_name]
        ssd_optimizer = self.optimizers[self.ssd_name]
        ssd_loss_builder = self.loss_builders[self.ssd_name]

        batch = next(self.train_dataloader)
        real_wav = batch['wav'].to(self.device).unsqueeze(1)
        input_wav = batch['input_wav'].to(self.device).unsqueeze(1)

        gen_wav = gen(input_wav)
        gen_mel = mel_spectrogram(gen_wav.squeeze(1), **self.mel_kwargs)
        real_mel = mel_spectrogram(real_wav.squeeze(1), **self.mel_kwargs)

        requires_grad(ssd, True)
        ssd_optimizer.zero_grad()

        ssd_real_out, ssd_gen_out, _, _ = ssd(real_wav, gen_wav.detach())
        ssd_loss, ssd_losses_dict = ssd_loss_builder.calculate_loss({
            'ssd': dict(
                discs_real_out=ssd_real_out,
                discs_gen_out=ssd_gen_out
            )
        }, tl_suffix='ssd')
        ssd_loss.backward()
        ssd_optimizer.step()

        requires_grad(ssd, False)
        gen_optimizer.zero_grad()

        _, ssd_gen_out, ssd_fmaps_real, ssd_fmaps_gen = ssd(real_wav, gen_wav)

        gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
            '': dict(
                gen_mel=gen_mel,
                real_mel=real_mel
            ),
            'ssd': dict(
                discs_gen_out=ssd_gen_out,
                fmaps_real=ssd_fmaps_real,
                fmaps_gen=ssd_fmaps_gen
            )
        }, tl_suffix='gen')

        gen_loss.backward()
        gen_optimizer.step()

        return {**gen_losses_dict, **ssd_losses_dict}

    def synthesize_wavs(self, batch):
        gen = self.models[self.gen_name]
        gen.eval()
        
        result_dict = {
            'gen_wav': [],
            'name': []
        }

        with torch.no_grad():
            for name, input_wav in zip(batch['name'], batch['input_wav']):
                gen_wav = gen(input_wav.to(self.device)[None,None]).squeeze()
                
                result_dict['gen_wav'].append(gen_wav)
                result_dict['name'].append(name)
        
        result_dict['gen_wav'] = torch.stack(result_dict['gen_wav'])
        return result_dict
