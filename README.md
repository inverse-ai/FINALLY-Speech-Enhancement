## Unofficial Implementation of "FINALLY: fast and universal speech enhancement with studio-like quality"

### Introduction

**Explore details and sample results on our GitHub Pages:** [https://inverse-ai.github.io/FINALLY-Speech-Enhancement/](https://inverse-ai.github.io/FINALLY-Speech-Enhancement/), which includes comprehensive information about the **FINALLY** speech enhancement model, audio examples comparing input and enhanced speech, and spectrogram visualizations for easy comparison.

**Try the model live** at [https://noise-reducer.com](https://noise-reducer.com) (with **NR v6.0**) to enhance your audio.

For architecture details, see the [FINALLY: fast and universal speech enhancement with studio-like quality](https://arxiv.org/abs/2410.05920) paper.

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

The model uses **WavLM-Large** from [Hugging Face](https://huggingface.co/microsoft/wavlm-large) as a frozen feature extractor.

- Automatically downloaded via `transformers` when training starts:
```python
from transformers import WavLMModel

wavlm = WavLMModel.from_pretrained(
    "microsoft/wavlm-large",
    output_hidden_states=True,
    force_download=bool(os.getenv("FORCE", False))
)
```

### Model Size

Trainable parameters per component (in millions):

| Component            | Parameters (M) |
|----------------------|----------------|
| SpectralUNet         | 4.5            |
| WavLM Post Processing| 50             |
| HiFi Pre Processing  | 33             |
| HiFi(v1)             | 14             |
| WaveUNet             | 10.7           |
| SpectralMaskNet      | 16.5           |
| WaveUNet Upsampler   | 15             |


- **Total trainable parameters:** 143.7 M
- **Non-trainable WavLM parameters:** 315 M
- **Total number of parameters (including WavLM):** 458.7

### Data Processing

Data handling and preprocessing are implemented in the `datasets_manager/` directory, which includes:

- `datasets.py` – definitions for dataset structures and preprocessing
- `dataloaders.py` – PyTorch dataloaders for training and validation
- `augmentations_modules.py` – audio augmentation utilities
- `inference_datasets.py` – dataset structures and preprocessing for inference

**Note:** Your datasets should be placed inside the `datasets/` directory.  
For more clarity about dataset structure and correct directory paths, refer to the config files and `datasets_manager/datasets.py`.

### Training

To train the model on your dataset, provide a config file and run name.

#### Stage 1: Initial Training
Train the generator only.
```bash
python train.py exp.config_path=configs/stage1_config.yaml exp.run_name=stage1
```

- **New Training:** Set `gen:checkpoint_path: null` when starting from the beginning.
- **Resuming:** Provide the path to the latest Stage 1 checkpoint in `gen:checkpoint_path`.

#### Stage 2: Generative and Adversarial Training
Train with a discriminator.
```bash
python train.py exp.config_path=configs/stage2_config.yaml exp.run_name=stage2
```

- **Setup:** Use the last checkpoint from Stage 1 for `gen:checkpoint_path`.
- **Note:** Set `disc:checkpoint_path: null` when starting Stage 2.

#### Stage 3: High Fidelity Upsampling
Upsample result from 16kHz to 48kHz.
```bash
python train.py exp.config_path=configs/stage3_config.yaml exp.run_name=stage3
```

- **Setup:** Use the last checkpoint from Stage 2 for `gen:checkpoint_path`.
- **Note:** Set `disc:checkpoint_path: null`.
- **Configuration:** Ensure `gen:args:use_upsamplewaveunet: true` is set in the config to enable 16k to 48k upsampling.

Optionally, you can specify the device e.g. `exp.device=cuda:0`.

### Multi-GPU Training (DDP)

We provide support for Distributed Data Parallel (DDP) training to speed up the process using multiple GPUs.

#### Run with Torchrun
To launch training on multiple GPUs (e.g., 2 GPUs), use `torchrun`:

```bash
torchrun --nproc_per_node=2 train_ddp.py \
         exp.config_path=configs/stage3_config_ddp.yaml \
         exp.run_name=stage3_ddp
```

#### Key DDP Configuration Changes
When using DDP (see `configs/stage3_config_ddp.yaml`), pay attention to these parameters:

- **Batch Size:** `data.train_batch_size` is the batch size **per GPU**.
- **Effective Batch Size:** `train.effective_batch_size` is the total batch size across all GPUs and accumulation steps.
- **Auto-accumulation:** The trainer automatically calculates the required gradient accumulation steps based on the world size and target effective batch size.

Relevant files: `train_ddp.py`, `trainers/finally_trainer_ddp.py`, and `configs/*_ddp.yaml`.

### Inference

To enhance speech from input audio files, provide the config and run name. Example:

```bash
python inference.py exp.config_path=configs/inferenc_config.yaml exp.run_name=inference
```

### Evaluation Scores

Our model was trained on both the datasets mentioned in the paper and additional high-quality datasets curated by us.

The table below compares the performance of the model using various metrics.

| Metric       | Paper’s Score | Ours Score |
|--------------|---------------|------------|
| UTMOS        | 4.32          | 4.30       |
| WV-MOS       | 4.87          | 4.62       |
| DNSMOS       | 3.22          | 3.30       |
| PESQ         | 2.94          | 3.22       |
| STOI         | 0.92          | 0.95       |
| SDR          | 4.6           | 6.79       |


### Current Challenges

During our implementation of the FINALLY speech enhancement model, we have identified several areas for improvement:

#### Challenge 1: Stationary Noise in Silence
A tiny amount of stationary noise remains in the enhanced audio, which is particularly audible at high volume during silence sections.

#### Challenge 2: Accent Alteration with UTMOS Loss
When integrating UTMOS loss, we observe that the speaker's accent occasionally changes in low SNR (Signal-to-Noise Ratio) portions. Interestingly, the accent remains preserved when training without UTMOS loss, suggesting a trade-off between perceived quality scores and speaker identity preservation.

#### Challenge 3: Voice Identity Shifts in Low-Intelligibility Speech
The model sometimes exhibits voice identity shifts (the voice sounds like a different person) when the input speech is extremely quiet or masked by heavy noise to the point of being nearly unintelligible.

### Contributing
We invite the research community to help resolve these challenges, or alternative approaches to address these issues. If you have experience with:
- WavLM feature extraction and perceptual losses
- Speech enhancement model training and loss balancing
- Phoneme preservation techniques in generative models

Please feel free to:
- Open an issue to discuss potential solutions
- Submit a pull request with experimental results
- Share relevant research papers or approaches

Your insights and contributions could help improve the quality and robustness of this implementation.
