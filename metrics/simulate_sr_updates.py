#!/usr/bin/env python3
import os
import sys

import torch
import torchaudio


def process_wav(in_path, out_path):
    """Load a WAV file, resample to 16k → 48k → original, and save it."""
    try:
        waveform, sr = torchaudio.load(in_path)
    except Exception as e:
        print(f"[Error] Could not read '{in_path}': {e}")
        return

    try:
        # Step 1: resample to 16k
        resample_16k = torchaudio.transforms.Resample(sr, 16000)
        wav_16k = resample_16k(waveform)

        # Step 2: resample to 48k
        resample_48k = torchaudio.transforms.Resample(16000, 48000)
        wav_48k = resample_48k(wav_16k)

        # Step 3: resample back to original sample rate
        resample_back = torchaudio.transforms.Resample(48000, sr)
        wav_final = resample_back(wav_48k)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        torchaudio.save(out_path, wav_final, sr)
        print(f"Processed: {os.path.basename(in_path)} (sr={sr})")

    except Exception as e:
        print(f"[Error] Failed to process '{in_path}': {e}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_folder> <output_folder>")
        sys.exit(1)

    in_dir, out_dir = sys.argv[1], sys.argv[2]

    if not os.path.isdir(in_dir):
        print(f"[Error] Input folder '{in_dir}' does not exist or is not a directory.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(in_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("[Error] No .wav files found in input folder.")
        sys.exit(1)

    for fname in wav_files:
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        process_wav(in_path, out_path)


if __name__ == "__main__":
    torch.set_num_threads(1)  # prevent CPU overuse
    main()
