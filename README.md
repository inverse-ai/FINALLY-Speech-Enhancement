## Unofficial Implementation of "FINALLY: fast and universal speech enhancement with studio-like quality"

### Introduction

**Explore details and sample results on our GitHub Pages:** [https://inverse-ai.github.io/FINALLY-Speech-Enhancement/](https://inverse-ai.github.io/FINALLY-Speech-Enhancement/), which includes comprehensive information about the **FINALLY** speech enhancement model, audio examples comparing input and enhanced speech, and spectrogram visualizations for easy comparison.

**Try the model live** at [https://noise-reducer.com](https://noise-reducer.com) to enhance your own audio in real time.

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


### Data Processing

Data handling and preprocessing are implemented in the `datasets/` directory, which includes:

- `datasets.py` – definitions for dataset structures and preprocessing
- `dataloaders.py` – PyTorch dataloaders for training and validation
- `augmentations.py` – audio augmentation utilities

**Note:** Your datasets should be placed inside the `datasets/` directory.  
For more clarity about dataset structure and correct directory paths, refer to the config files and `datasets.py`.

### Training

To train the model on your dataset, provide a config file and run name. Example:

```bash
python train.py exp.config_path=configs/finally/finally_stage3_config.yaml \
                exp.run_name=stage3_train
```

Optionally, you can specify the device e.g. `exp.device=cuda:0`.

### Inference

To enhance speech from input audio files, provide the config and run name. Example:

```bash
python inference.py exp.config_path=configs/finally_inference48_config.yaml exp.run_name=stage3_inference_40K_steps_VCTK-demand
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

We observed a trade-off when using feature matching loss (extracted from WavLM) in LMOS:

- **Without feature matching loss:**  
  - UTMOS and PESQ scores are lower  
  - Perceptual output quality is better  

- **With feature matching loss:**  
  - UTMOS and PESQ scores improve  
  - Perceptual output quality is slightly compromised  
  - Integration introduces artifacts, which can lead to additional noise  

Balancing quantitative metrics and perceptual quality remains an ongoing challenge.


