from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from utils import AttrDict
from load_dataset import load_wav, write_wav
from finally_model import Finally

h = None
device = None




def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Finally().to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # Get list of input files
    if os.path.isdir(a.input_wavs_dir):
        filelist = [f for f in os.listdir(a.input_wavs_dir) if f.endswith('.wav')]
    else:
        raise ValueError(f"Input directory {a.input_wavs_dir} does not exist")
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            # try:
                # Load and process audio
                print(f"filename : {a.input_wavs_dir}")
                wav_n, sr_n = load_wav(os.path.join(a.input_wavs_dir, filename))
                wav_n = wav_n.to(device)
                init_len = wav_n.shape[1]

                if sr_n != h.sampling_rate:
                    raise ValueError(f"Sample rate {sr_n} doesn't match config {h.sampling_rate}")

                if sr_n != h.sampling_rate:
                    raise ValueError(f"Sample rate {sr_n} doesn't match config {h.sampling_rate}")

                print(f"Processing {filename}, input shape: {wav_n.shape}")

                print(f"deserve : {wav_n.shape}")
                intial_length = wav_n.shape[1]
                if wav_n.shape[1] % h.segment_size != 0:
                    pad_size = h.segment_size - (wav_n.shape[1] % h.segment_size)
                    wav_n = torch.nn.functional.pad(wav_n, (0, pad_size), mode='constant')
                    print("okay")

                # Process audio in chunks
                audio_chunks = []
                for i in range(0, wav_n.shape[1], h.segment_size):
                    chunk_n = wav_n[:, i:i+h.segment_size]
                    with torch.no_grad():
                        if a.phase == 'phase3':
                            new_g_hat = generator(chunk_n, True)
                        else:
                            new_g_hat = generator(chunk_n, False)
                        audio_chunks.append(new_g_hat.squeeze())

                # Concatenate and trim to original length
                if a.phase == 'phase3':
                    audio = torch.cat(audio_chunks, dim=0)[: 3 * init_len]
                else:
                    audio = torch.cat(audio_chunks, dim=0)[: init_len]

                # Save output
                output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '.wav')
                if a.phase == 'phase3':
                    write_wav(output_file, audio, 3 * h.sampling_rate)
                else:
                    write_wav(output_file, audio, h.sampling_rate)
                print(f"Saved {output_file}")


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='/home/ml/Projects/Tonmoy/finally-speech-enhancement/data/phase_1_2/test_files', help='Directory containing input WAV files')
    parser.add_argument('--output_dir', default='/home/ml/Projects/Tonmoy/finally-speech-enhancement/data/phase_1_2/inferred_files', help='Directory to save generated WAVs')
    parser.add_argument('--device', default='cuda:0', help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--phase', default='phase1', help='Phase of the model to use (phase1, phase2, phase3)')
    parser.add_argument('--checkpoint_file', required=True, help='Path to generator checkpoint')
    parser.add_argument('--config', default='outputs/cp/config.json', help='Path to config file')
    a = parser.parse_args()
    
    config_file = a.config
    print(f"file : {config_file}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    global h
    with open(config_file) as f:
        data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    inference(a)


if __name__ == '__main__':
    main()
    
# python inference.py --input_wavs_dir data/test_input --output_dir data/test_inferenced_output --checkpoint_file outputs/g_00000250