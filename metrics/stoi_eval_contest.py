#!/usr/bin/env python3
import argparse
import csv
import glob
import logging
import os
import sys

import numpy as np
import soundfile as sf
from pystoi import stoi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def probe_wav(path):
    """Return (samplerate, channels, frames) for path. Exit on failure."""
    try:
        info = sf.info(path)
    except Exception as e:
        sys.exit(f"[Error] Could not read header for '{path}': {e}")
    return info.samplerate, info.channels, info.frames


def load_wavs(ref_path, inf_path):
    """Load two wav files as float32 numpy arrays and validate basic equality.
    Returns (ref, inf, fs). Exits on error."""
    try:
        ref, fs = sf.read(ref_path, dtype="float32", always_2d=False)
    except Exception as e:
        sys.exit(f"[Error] Could not read '{ref_path}': {e}")
    try:
        inf, fs2 = sf.read(inf_path, dtype="float32", always_2d=False)
    except Exception as e:
        sys.exit(f"[Error] Could not read '{inf_path}': {e}")

    if fs != fs2:
        sys.exit(
            f"[Error] Sample rates do not match: {ref_path} ({fs} Hz) vs {inf_path} ({fs2} Hz)"
        )
    # Ensure mono (1-D) arrays
    if ref.ndim != 1 or inf.ndim != 1:
        sys.exit(
            f"[Error] Only mono WAV files supported. Got shapes: {ref_path} {ref.shape}, {inf_path} {inf.shape}"
        )

    if ref.shape != inf.shape:
        sys.exit(
            f"[Error] Audio shapes do not match: {ref_path} {ref.shape} vs {inf_path} {inf.shape}"
        )

    return ref, inf, fs


def stoi_metric(ref, inf, fs=16000):
    """Calculate Extended Short-Time Objective Intelligibility (ESTOI).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        estoi (float): ESTOI value between [0, 1]
    """
    np.random.seed(0)  # make estoi deterministic
    return stoi(ref, inf, fs_sig=fs, extended=True)


def compute_single(ref_path, deg_path):
    """Load, validate and compute STOI for a single pair. Returns float or None."""
    ref, inf, fs = load_wavs(ref_path, deg_path)
    return stoi_metric(ref, inf, fs=fs)


def validate_all_pairs_strict(ref_dir, deg_dir, common_fnames):
    """Strictly validate that for every matched filename, samplerate/channels/frames match.
    Exit with an aggregated error message if any mismatch found."""
    problems = []
    for fname in common_fnames:
        ref_path = os.path.join(ref_dir, fname)
        deg_path = os.path.join(deg_dir, fname)
        try:
            ref_sr, ref_ch, ref_frames = probe_wav(ref_path)
            deg_sr, deg_ch, deg_frames = probe_wav(deg_path)
        except SystemExit as e:
            # probe_wav already exits; re-raise
            raise

        if ref_sr != deg_sr or ref_ch != deg_ch or ref_frames != deg_frames:
            problems.append(
                (fname, (ref_sr, ref_ch, ref_frames), (deg_sr, deg_ch, deg_frames))
            )

    if problems:
        lines = ["[Error] Found mismatched files (samplerate / channels / frames):"]
        for fname, ref_meta, deg_meta in problems:
            lines.append(
                f"  {fname}: ref(sr={ref_meta[0]}, ch={ref_meta[1]}, frames={ref_meta[2]}) "
                f"vs deg(sr={deg_meta[0]}, ch={deg_meta[1]}, frames={deg_meta[2]})"
            )
        lines.append(
            "Aborting. Ensure all matched pairs are identical in sample rate, channels (mono), and length."
        )
        sys.exit("\n".join(lines))


def handle_directories(ref_dir, deg_dir, out_csv_path):
    """Process directories: match filenames, validate strictly, compute STOI, write CSV."""
    ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
    deg_files = sorted(glob.glob(os.path.join(deg_dir, "*.wav")))

    if not ref_files or not deg_files:
        sys.exit("[Error] One or both directories are empty or contain no .wav files.")

    ref_map = {os.path.basename(p): p for p in ref_files}
    deg_map = {os.path.basename(p): p for p in deg_files}

    common = sorted(set(ref_map.keys()) & set(deg_map.keys()))
    missing_ref = sorted(set(deg_map.keys()) - set(ref_map.keys()))
    missing_deg = sorted(set(ref_map.keys()) - set(deg_map.keys()))

    if missing_ref:
        logging.warning(
            f"These files exist in degraded dir but not in reference dir: {missing_ref}"
        )
    if missing_deg:
        logging.warning(
            f"These files exist in reference dir but not in degraded dir: {missing_deg}"
        )

    if not common:
        sys.exit(
            "[Error] No matching .wav filenames found between the two directories."
        )

    # Strict validation of metadata for all matching pairs
    validate_all_pairs_strict(ref_dir, deg_dir, common)
    logging.info(
        f"Validated {len(common)} matching file pairs. Starting STOI computation..."
    )

    results = []
    for fname in common:
        rpath = ref_map[fname]
        dpath = deg_map[fname]
        try:
            score = compute_single(rpath, dpath)
            if score is None:
                logging.warning(
                    f"{fname}: STOI returned no valid score; skipping from results."
                )
            else:
                results.append((fname, score))
                logging.info(f"{fname}: STOI MOS = {score:.4f}")
        except SystemExit as e:
            # compute_single uses sys.exit on validation — that should not happen because we validated above.
            logging.error(f"[Error] Validation failed for pair {fname}: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"[Error] {fname}: unexpected failure -> {e}")

    # Write CSV
    try:
        with open(out_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "stoi_mos"])
            for fname, score in results:
                writer.writerow([fname, f"{score:.6f}"])
            # Average over computed results only
            if results:
                avg = sum(s for _, s in results) / len(results)
                writer.writerow(["AVERAGE", f"{avg:.6f}"])
            else:
                writer.writerow(["AVERAGE", "N/A"])
    except Exception as e:
        sys.exit(f"[Error] Could not write CSV '{out_csv_path}': {e}")

    logging.info(f"Saved results to: {out_csv_path}")
    if results:
        logging.info(f"Average STOI MOS over {len(results)} files: {avg:.4f}")
    else:
        logging.info("No valid STOI scores computed.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute STOI MOS score between WAV files or directories"
    )
    parser.add_argument("ref", help="Reference WAV file or directory")
    parser.add_argument("deg", help="Degraded WAV file or directory")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output CSV path (required when both inputs are directories)",
    )
    args = parser.parse_args()

    # file-file
    if os.path.isfile(args.ref) and os.path.isfile(args.deg):
        score = compute_single(args.ref, args.deg)
        if score is None:
            logging.warning("STOI returned no valid score for the pair.")
            print("STOI MOS: N/A")
        else:
            print(f"STOI MOS: {score:.4f}")

        if args.out:
            try:
                with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["filename", "stoi_mos"])
                    writer.writerow(
                        [
                            os.path.basename(args.ref),
                            f"{score:.6f}" if score is not None else "N/A",
                        ]
                    )
                logging.info(f"Saved result to: {args.out}")
            except Exception as e:
                sys.exit(f"[Error] Could not write CSV '{args.out}': {e}")

    # dir-dir
    elif os.path.isdir(args.ref) and os.path.isdir(args.deg):
        if not args.out:
            sys.exit(
                "[Error] When providing two directories, you must pass an output CSV path via --out"
            )
        out_parent = os.path.dirname(os.path.abspath(args.out))
        if out_parent and not os.path.exists(out_parent):
            try:
                os.makedirs(out_parent, exist_ok=True)
            except Exception as e:
                sys.exit(
                    f"[Error] Could not create output directory '{out_parent}': {e}"
                )
        handle_directories(args.ref, args.deg, args.out)

    else:
        sys.exit(
            "[Error] Both inputs must be either files or directories (matching mode)."
        )


if __name__ == "__main__":
    main()
