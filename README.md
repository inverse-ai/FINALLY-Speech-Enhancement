## Unofficial Implementation of "FINALLY: fast and universal speech enhancement with studio-like quality"

### Introduction

The repository is the unofficial implementation of the **FINALLY** speech enhancement model, designed to improve audio quality to **studio-like standards**. It provides fast and universal enhancement across a variety of speech recordings.

### Paper

For more details, see the [FINALLY: fast and universal speech enhancement with studio-like quality](https://arxiv.org/abs/2410.05920) paper

### Environment

We recommend using Conda to manage dependencies.

#### Create Conda Environment

```bash
conda create -n finally_env python=3.10 pip
conda activate finally_env
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pretrained Models

Download the **WavLM-Large** model from [Hugging Face](https://huggingface.co/microsoft/wavlm-large) and place it in the `wavlm/` directory.

### Model Size

Trainable parameters per component (in millions):

| Component            | Parameters (M) |
|----------------------|----------------|
| spectralunet         | 4.4            |
| resblock             | 7.1            |
| conv                 | 0.8            |
| hifi                 | 15.5           |
| waveunet             | 10.8           |
| spectralmasknet      | 5.5            |
| waveunet_upsampler   | 3.9            |

- **Total trainable parameters:** 48.0 M  
- **Non-trainable WavLM parameters:** 315.5 M  
- **Total number of parameters (including WavLM):** 363.5 M  


### Data

The `data/` directory contains some dummy audio data for testing and training.

#### Stage 1

`data/stage_1/` contains:

- `clean/` – 18 audio files, 2 seconds each at 16 kHz  
- `noisy/` – 18 audio files, 2 seconds each at 16 kHz  
- `training.txt` – lists the filenames of the audio files  

The clean and noisy directories have **matching filenames**.

#### Stage 2

`data/stage_2/` contains the **same structure and data** as `stage_1`.

#### Stage 3

`data/stage_3/` contains:

- `noisy/` – 18 audio files, 2 seconds each at 16 kHz  
- `clean/` – 18 audio files, 48 kHz  
- `training.txt` – lists the filenames of the audio files 


### Training

To train the model on your dataset:

```bash
python train.py
```

### Inference

To enhance speech from input audio files:

```bash
python inference.py --input_wavs_dir data/test_data \
                    --output_dir data/inferred_data \
                    --checkpoint_file outputs/g_00000250
```

### Notes

* Ensure the `checkpoint_file` exists before running inference.
* Recommended input formats: WAV files sampled at 16kHz.

