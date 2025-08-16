# Unofficial Implementation of "FINALLY: fast and universal speech enhancement with studio-like quality"

## Introduction

The repository is the implementation of the **FINALLY** speech enhancement model, designed to improve audio quality to **studio-like standards**. It provides fast and universal enhancement across a variety of speech recordings.

## Paper

For more details, see the [FINALLY: fast and universal speech enhancement with studio-like quality](https://arxiv.org/abs/2410.05920) paper

## Environment

We recommend using Conda to manage dependencies.

### Create Conda Environment

```bash
conda create -n finally_env python=3.10 pip
conda activate finally_env
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Here’s a concise **“Pretrained Models”** section you can add to your README:

---

## Pretrained Models

Download the **WavLM-Large** model from [Hugging Face](https://huggingface.co/microsoft/wavlm-large) and place it in the `wavlm/` directory.

## Training

To train the model on your dataset:

```bash
python train.py
```

## Inference

To enhance speech from input audio files:

```bash
python inference.py --input_wavs_dir data/test_data \
                    --output_dir data/inferred_data \
                    --checkpoint_file outputs/g_00000250
```

## Notes

* Ensure the `checkpoint_file` exists before running inference.
* Recommended input formats: WAV files sampled at 16kHz.
