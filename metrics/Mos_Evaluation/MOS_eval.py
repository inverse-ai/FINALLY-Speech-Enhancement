import argparse
import concurrent.futures
import glob
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
import logging

from dnsmos import DNSMOS_local

logger = logging.getLogger(__name__)

from os.path import dirname, join, abspath

denoiser_root = abspath(
    join(dirname(__file__), "..", "..", "hybridTransformerDemucsPytorch")
)
sys.path.append(denoiser_root)
project_root = abspath(join(dirname(__file__), "..", ".."))
sys.path.append(project_root)

from hybridTransformerDemucsPytorch.denoiser.enhance import add_flags, enhance


def main(args):
    directories = glob.glob(os.path.join(args.testset_dir, "*"))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p808_model_path = os.path.join(current_dir, "DNSMOS", "model_v8.onnx")

    if args.personalized_MOS:
        primary_model_path = os.path.join(current_dir, "pDNSMOS", "sig_bak_ovr.onnx")
    else:
        primary_model_path = os.path.join(current_dir, "DNSMOS", "sig_bak_ovr.onnx")

    use_gpu = True if torch.cuda.is_available() and args.device == "cuda" else False
    model = DNSMOS_local(
        primary_model_path=primary_model_path,
        p808_model_path=p808_model_path,
        use_gpu=use_gpu,
        convert_to_torch=use_gpu,
    )

    rows = []
    clips = []
    formats = args.format.split(",")

    for fmt in formats:
        pattern = os.path.join(args.testset_dir, "**", f"*.{fmt}")
        clips.extend(glob.glob(pattern, recursive=True))

    is_personalized_eval = args.personalized_MOS

    print(len(clips), "audio clips found in the testset directory")
    if use_gpu:
        for clip in tqdm(clips):
            rows.append(
                model(clip, is_personalized_eval)
            )  # Run on the first clip to warm up the GPU
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.worker) as executor:
            future_to_url = {
                executor.submit(model, clip, is_personalized_eval): clip
                for clip in clips
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_url), total=len(clips)
            ):
                clip = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (clip, exc))
                else:
                    rows.append(data)

    df = pd.DataFrame(rows)

    # Calculate mean values for each metric
    mean_values = df.mean(numeric_only=True)

    # Create a dictionary with mean values, setting non-numeric columns as needed
    mean_values_dict = mean_values.to_dict()
    mean_values_dict["filename"] = (
        "Mean Values"  # Example of setting a non-numeric column
    )

    print("\n")
    for column in df.select_dtypes(include=['number']).columns:
        top_25_mean, top_50_mean, top_75_mean, full_mean = mean_of_top_percentile(df, column)
        print(f"Column: {column} \t 25%: {top_25_mean:.6f} \t 50%: {top_50_mean:.6f} \t 75%: {top_75_mean:.6f} \t 100%: {full_mean:.6f}")

    # Convert the dictionary to a DataFrame to make it a single row
    mean_df = pd.DataFrame([mean_values_dict])

    # Concatenate the original DataFrame with the mean values DataFrame
    final_df = pd.concat([df, mean_df], ignore_index=True)

    if args.csv_path:
        csv_path = args.csv_path
        final_df.to_csv(csv_path)
    
def mean_of_top_percentile(df, column):
    sorted_df = df.sort_values(by=column, ascending=False)
    total_count = len(sorted_df)
    top_25_df = sorted_df.iloc[:int(total_count * 0.25)]
    top_50_df = sorted_df.iloc[:int(total_count * 0.50)]
    top_75_df = sorted_df.iloc[:int(total_count * 0.75)]
    top_100_df = sorted_df.iloc[:int(total_count)]
    return top_25_df[column].mean(), top_50_df[column].mean(), top_75_df[column].mean(), top_100_df[column].mean()


if __name__ == "__main__":
    # Additional arguments for enhance function
    parser = argparse.ArgumentParser(
        "denoiser.enhance",
        description="Speech enhancement using Demucs - Generate enhanced files",
    )

    add_flags(parser)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="enhanced",
        help="directory putting enhanced wav files",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="wav",
        help="format of the audio files to be evaluated. Comma separated values are allowed, e.g. wav,flac,mp3",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="more logging",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--noisy_dir",
        type=str,
        default=None,
        help="directory including noisy wav files",
    )
    group.add_argument(
        "--noisy_json",
        type=str,
        default=None,
        help="json file including noisy wav files",
    )

    parser.add_argument(
        "-t",
        "--testset_dir",
        default=".",
        help="Path to the dir containing audio clips in .wav to be evaluated",
    )
    parser.add_argument(
        "-o", "--csv_path", default=None, help="Path to the csv that saves the results"
    )
    parser.add_argument(
        "-p",
        "--personalized_MOS",
        action="store_true",
        help="Flag to indicate if personalized MOS score is needed or regular",
    )
    parser.add_argument(
        "-w",
        "--worker",
        type=int,
        default=1,
        help="Number of workers for concurrent processing of audio files",
    )
    # Argument for deleting the noisy directory

    args = parser.parse_args()

    # If noisy_dir is given then we will enhance the noisy files using enhance.py then we will evaluate the enhanced files
    if args.noisy_dir or args.noisy_json:
        logging.basicConfig(stream=sys.stderr, level=args.verbose)
        args.testset_dir = args.out_dir
        enhance(args, local_out_dir=args.out_dir)
        args.format = "flac"

    main(args)


"""
Enhance + MOS score

python3 MOS_eval.py \
    --model_path '/home/saiham/Data/1000_HOUR_DATASET/postvad_mixed_split/outputs/exp_fromstart/epoch_20_checkpoint.th' \
    --noisy_dir '/home/saiham/Data/testclips/noisy' \
    --out_dir '/home/saiham/Data/1000_HOUR_DATASET/postvad_mixed_split/outputs/exp_fromstart/enh_dns_20' \
    --device 'cuda' \
    -o '/home/saiham/Data/1000_HOUR_DATASET/postvad_mixed_split/outputs/exp_fromstart/enh_dns_20.csv'


python3 MOS_eval.py --model_path /mnt/Data/noise-reducer-ml/hybridTransformerDemucsPytorch/outputs/exp_/epoch_16_checkpoint.th --noisy_dir /mnt/Data/testclips --out_dir /home/tashin/Documents/mashroor/DNS-ep16 -o /home/tashin/Documents/mashroor/DNS-ep16.csv
For batch_size = 5, 822,Mean Values,3.1116510397616386,3.4143064319362315,4.008222799521583,3.767425775527954
For batch size = 4, 822,Mean Values,3.1117134278418086,3.4143741074269705,4.008208841302969,3.767440165626452

Only MOS score

python3 MOS_eval.py -t '/home/saiham/Data/Tonmoy/Data/Clean_Data/60k_ger_fre_spa' -o '/home/saiham/Data/Tonmoy/Data/Clean_Data/60k_ger_fre_spa.csv' --format wav
"""
