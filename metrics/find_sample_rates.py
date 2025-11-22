#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys

import soundfile as sf


def probe_samplerate(path):
    """Return samplerate (int) or None if unreadable."""
    try:
        info = sf.info(path)
        return int(info.samplerate)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(
        description="List samplerates of WAV files in a folder and save to CSV."
    )
    p.add_argument("indir", help="Input directory containing .wav files")
    p.add_argument("outcsv", help="Output CSV path (will be overwritten if exists)")
    args = p.parse_args()

    indir = args.indir
    outcsv = args.outcsv

    if not os.path.isdir(indir):
        sys.exit(
            f"[Error] Input directory does not exist or is not a directory: {indir}"
        )

    wav_paths = sorted(
        glob.glob(os.path.join(indir, "*.wav"))
        + glob.glob(os.path.join(indir, "*.WAV"))
    )
    if not wav_paths:
        sys.exit("[Error] No .wav files found in the input directory.")

    # ensure parent of output exists
    out_parent = os.path.dirname(os.path.abspath(outcsv))
    if out_parent and not os.path.exists(out_parent):
        os.makedirs(out_parent, exist_ok=True)

    rows = []
    for pth in wav_paths:
        sr = probe_samplerate(pth)
        fname = os.path.basename(pth)
        if sr is None:
            print(f"[Warning] Could not read samplerate for: {fname}")
            rows.append((fname, "N/A"))
        else:
            rows.append((fname, str(sr)))

    try:
        with open(outcsv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "samplerate"])
            writer.writerows(rows)
    except Exception as e:
        sys.exit(f"[Error] Could not write output CSV '{outcsv}': {e}")

    print(f"Saved {len(rows)} entries to: {outcsv}")


if __name__ == "__main__":
    main()
