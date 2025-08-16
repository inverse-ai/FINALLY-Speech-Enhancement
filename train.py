import subprocess
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import argparse
import json
import torch
from torch.utils.data import DataLoader

from load_dataset import Load_Dataset, get_dataset_filelist_noisy, mel_spectrogram
from loss_functions import feature_loss, generator_loss, discriminator_loss, lmos_loss, human_feedback_loss
from utils import AttrDict, scan_checkpoint, load_checkpoint, save_checkpoint
from finally_model import Finally
from msstftd import MultiScaleSTFTDiscriminator
from utmos_loss.utmos.score import Score


def load_data(filelist_path, clean_dir, noisy_dir, batch_size, conf):
    filelist = get_dataset_filelist_noisy(filelist_path)
    dataset = Load_Dataset(clean_dir, noisy_dir, filelist, shuffle=True)
    return DataLoader(dataset, num_workers=conf.num_workers, shuffle=True,
                        batch_size=batch_size, pin_memory=False, drop_last=True)

def run_stage(cp_g_path, generator, hp, conf, device):
    stage = hp['name']
    stage_no = hp['stage_no']
    csv_path = os.path.join(conf.checkpoint_path, f"training_log{stage_no}.csv")
    os.makedirs(conf.checkpoint_path, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("Steps,Gen Loss\n")

    steps = 0
    discriminator = None
    if stage != 'stage_1':
        discriminator = MultiScaleSTFTDiscriminator(filters=32, stage_no=stage_no).to(device)
        discriminator.train()
        
    if stage == 'stage_3':
        score = Score(device)

    state_dict_g = load_checkpoint(cp_g_path, device) if cp_g_path else None
    if state_dict_g:
        generator.load_state_dict(state_dict_g['generator'])
        generator.train()
        steps = state_dict_g['steps']

    optim_g = torch.optim.AdamW(generator.parameters(), lr=hp['lr'],
                                betas=(hp['adam_g_b1'], hp['adam_g_b2']))
    if state_dict_g:
        optim_g.load_state_dict(state_dict_g['optim_g'])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp['lr_decay'])

    if discriminator:
        cp_d_path = scan_checkpoint(conf.checkpoint_path, f'd{stage_no}_')
        state_dict_d = load_checkpoint(cp_d_path, device) if cp_d_path else None
        if state_dict_d:
            discriminator.load_state_dict(state_dict_d['discriminator'])
            discriminator.train()

        optim_d = torch.optim.AdamW(discriminator.parameters(), lr=hp['lr'],
                            betas=(hp['adam_d_b1'], hp['adam_d_b2']))
        if state_dict_d:
            optim_d.load_state_dict(state_dict_d['optim_d'])
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp['lr_decay'])
    else:
        optim_d = scheduler_d = None

    dataloader = load_data(hp['input_training_file'], hp['input_clean_dir'], hp['input_noisy_dir'], hp['batch_size'], conf)

    graph_loss_value = 0.0
    batch_buffer = []
    start_time = time.time()
    start_time_batch = time.time()
    grad_batch_iter = hp['grad_batch_iter']

    while steps < hp['steps']:
        for batch in dataloader:
            if steps >= hp['steps']:
                break
            steps += 1

            clean, noisy = batch
            clean, noisy = clean.to(device), noisy.to(device)

            gen_audio = generator(noisy, True if stage == 'stage_3' else False)

            if discriminator:
                batch_buffer.append((clean.unsqueeze(1), gen_audio.unsqueeze(1)))

                if steps % grad_batch_iter == 0 or steps == hp['steps']:
                    for _ in range(2):
                        for real, fake in batch_buffer:
                            y_real, _ = discriminator(real.detach())
                            y_fake, _ = discriminator(fake.detach())
                            loss_d, _ = discriminator_loss(y_real, y_fake)
                            (loss_d / grad_batch_iter).backward()

                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                        optim_d.step()
                        optim_d.zero_grad()
                    batch_buffer.clear()

            # Generator loss
            clean_mel_loss = mel_spectrogram(clean, 1024, 80, hp['sampling_rate'], 256, 1024, 0, None)
            gen_mel = mel_spectrogram(gen_audio, 1024, 80, hp['sampling_rate'], 256, 1024, 0, None) 

            loss_lmos = lmos_loss(clean_mel_loss, gen_mel, clean, gen_audio, generator.wavlm)
            loss = hp['lambda_lmos'] * loss_lmos

            if discriminator:
                _, fmap_r = discriminator(clean.unsqueeze(1))
                y_hat_g, fmap_g = discriminator(gen_audio.unsqueeze(1))
                loss_gen = generator_loss(y_hat_g)
                lambda_fm = hp['lambda_fm'] * min(1.0, steps / hp['linear_warmup_steps'])
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss += hp['lambda_gan'] * loss_gen + lambda_fm * loss_fm

            if stage == 'stage_3':
                loss_hf = human_feedback_loss(clean, gen_audio, score)
                loss += hp['lambda_hf'] * loss_hf

            loss = loss / grad_batch_iter
            graph_loss_value += loss.item()
            loss.backward()

            if steps % grad_batch_iter == 0 or steps == hp['steps']:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optim_g.step()
                optim_g.zero_grad()

            if steps % conf.stdout_interval == 0 or steps == hp['steps']:
                graph_loss_value /= conf.stdout_interval
                print(f"stage: {stage}, Steps: {steps}, Gen Loss: {graph_loss_value:.3f}, Time: {time.time() - start_time_batch:.3f}s")
                start_time_batch = time.time()
                with open(csv_path, 'a') as f:
                    f.write(f"{steps},{graph_loss_value:.3f}\n")
                graph_loss_value = 0.0

            if steps % hp['checkpoint_interval'] == 0 or steps == hp['steps']:
                save_checkpoint(os.path.join(conf.checkpoint_path, f'g{stage_no}_{steps:08d}'),
                                {'generator': generator.state_dict(),
                                'optim_g': optim_g.state_dict(),
                                'steps': steps})
                if discriminator:
                    save_checkpoint(os.path.join(conf.checkpoint_path, f'd{stage_no}_{steps:08d}'),
                                    {'discriminator': discriminator.state_dict(),
                                    'optim_d': optim_d.state_dict()})

            # LR scheduler
            if stage == 'stage_1':
                if steps % hp['lr_decay_interval'] == 0:
                    scheduler_g.step()
            else:
                if steps <= hp['linear_warmup_steps']:
                    lr = hp['lr'] * (steps / hp['linear_warmup_steps'])**2
                    for group in optim_g.param_groups:
                        group['lr'] = lr
                elif steps % (hp['lr_decay_interval_coeff'] * hp['lr_decay_interval']) == 0:
                    scheduler_g.step()
                if steps % hp['lr_decay_interval'] == 0 and steps > hp['linear_warmup_steps']:
                    scheduler_d.step()

    print(f"stage {stage_no} complete in {int(time.time() - start_time)}s")

def train(conf, device):
    torch.manual_seed(conf.seed)
    torch.autograd.set_detect_anomaly(True)
    generator = Finally().to(device)
    generator.train()

    checkpoints = {
        "stage_1": scan_checkpoint(conf.checkpoint_path, "g1_"),
        "stage_2": scan_checkpoint(conf.checkpoint_path, "g2_"),
        "stage_3": scan_checkpoint(conf.checkpoint_path, "g3_")
    }

    for stage in conf.stages:
        print(f"Starting {stage['name']}...")
        if stage['name'] == 'stage_1' and checkpoints['stage_2'] is None and checkpoints['stage_3'] is None:
            run_stage(checkpoints['stage_1'], generator, stage, conf, device)
        elif stage['name'] == 'stage_2' and checkpoints['stage_3'] is None:
            run_stage(checkpoints['stage_2'], generator, stage, conf, device)
        elif stage['name'] == 'stage_3':
            run_stage(checkpoints['stage_3'], generator, stage, conf, device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/config.json')
    args = parser.parse_args()

    with open(args.config) as f:
        json_config = json.load(f)
    conf = AttrDict(json_config)

    os.makedirs(conf.checkpoint_path, exist_ok=True)
    with open(os.path.join(conf.checkpoint_path, 'config.json'), 'w') as f:
        json.dump(json_config, f, indent=2)

    device = torch.device(conf.device if torch.cuda.is_available() else 'cpu')
    train(conf, device)

if __name__ == '__main__':
    main()