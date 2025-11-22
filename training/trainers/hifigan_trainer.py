import torch
from utils.data_utils import mel_spectrogram
from utils.model_utils import requires_grad
from training.trainers.base_trainer import BaseTrainer

from training.trainers.base_trainer import gan_trainers_registry

@gan_trainers_registry.add_to_registry(name='hifigan_trainer')
class HifiGanTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.gen_name = 'hifigan_gen'    
        self.mpd_name = 'mpd'
        self.msd_name = 'msd'

    def train_step(self):
        gen = self.models[self.gen_name]
        gen_optimizer = self.optimizers[self.gen_name]
        gen_loss_builder = self.loss_builders[self.gen_name]
        mpd = self.models[self.mpd_name]
        mpd_optimizer = self.optimizers[self.mpd_name]
        mpd_loss_builder = self.loss_builders[self.mpd_name]
        msd = self.models[self.msd_name]
        msd_optimizer = self.optimizers[self.msd_name]
        msd_loss_builder = self.loss_builders[self.msd_name]

        batch = next(self.train_dataloader)
        mel = batch['mel'].to(self.device)
        real_wav = batch['wav'].to(self.device)
        real_wav = real_wav.unsqueeze(1)
        mel_for_loss = batch['mel_for_loss'].to(self.device)

        gen_wav = gen(mel)
        gen_mel = mel_spectrogram(gen_wav.squeeze(1),
                                self.config.mel.n_fft,
                                self.config.mel.num_mels,
                                self.config.mel.sampling_rate,
                                self.config.mel.hop_size,
                                self.config.mel.win_size,
                                self.config.mel.fmin,
                                self.config.mel.fmax_for_loss)

        requires_grad(mpd, True)
        requires_grad(msd, True)
        mpd_optimizer.zero_grad()
        msd_optimizer.zero_grad()

        mpd_real_out, mpd_gen_out, _, _ = mpd(real_wav, gen_wav.detach())
        mpd_loss, mpd_losses_dict = mpd_loss_builder.calculate_loss({
            'mpd': dict(
                discs_real_out=mpd_real_out,
                discs_gen_out=mpd_gen_out
            )
        }, tl_suffix='mpd')
        mpd_loss.backward()
        mpd_optimizer.step()

        msd_real_out, msd_gen_out, _, _ = msd(real_wav, gen_wav.detach())
        msd_loss, msd_losses_dict = msd_loss_builder.calculate_loss({
            'msd': dict(
                discs_real_out=msd_real_out,
                discs_gen_out=msd_gen_out
            )
        }, tl_suffix='msd')
        msd_loss.backward()
        msd_optimizer.step()


        requires_grad(mpd, False)
        requires_grad(msd, False)
        gen_optimizer.zero_grad()

        _, mpd_gen_out, mpd_fmaps_real, mpd_fmaps_gen = mpd(real_wav, gen_wav)
        _, msd_gen_out, msd_fmaps_real, msd_fmaps_gen = msd(real_wav, gen_wav)

        gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
            '': dict(
                gen_mel=gen_mel,
                real_mel=mel_for_loss
            ),
            'mpd': dict(
                discs_gen_out=mpd_gen_out,
                fmaps_real=mpd_fmaps_real,
                fmaps_gen=mpd_fmaps_gen
            ),
            'msd': dict(
                discs_gen_out=msd_gen_out,
                fmaps_real=msd_fmaps_real,
                fmaps_gen=msd_fmaps_gen
            )
        }, tl_suffix='gen')

        gen_loss.backward()
        gen_optimizer.step()

        return {**gen_losses_dict, **mpd_losses_dict, **msd_losses_dict}

    def synthesize_wavs(self, batch):
        gen = self.models[self.gen_name]
        gen.eval()
        
        result_dict = {
            'gen_wav': [],
            'name': []
        }

        with torch.no_grad():
            for name, mel in zip(batch['name'], batch['mel']):
                gen_wav = gen(mel.to(self.device)).squeeze(0)
                
                result_dict['gen_wav'].append(gen_wav)
                result_dict['name'].append(name)
        
        result_dict['gen_wav'] = torch.stack(result_dict['gen_wav'])
        return result_dict
