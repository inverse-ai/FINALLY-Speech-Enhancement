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

To train the model on your dataset, provide a config file and run name. Example:

```bash
python train.py exp.config_path=configs/stage1_config.yaml \
                exp.run_name=stage1_train
```

Optionally, you can specify the device e.g. `exp.device=cuda:0`.

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

During our implementation of the FINALLY speech enhancement model, we have encountered several technical challenges related to the WavLM-based perceptual loss component of the LMOS regression loss.

#### Challenge 1: Artifacts with Full Feature Projection Pipeline
When extracting WavLM features using the complete feature projection pipeline (including LayerNorm, Linear projection, and Dropout layers), we observe significant artifacts in the output spectrograms during inference. 

**Observations:**
- WavLM convolutional feature loss: ~0.00003–0.00005
- STFT loss: ~0.3–0.4
- With the paper's suggested loss weights (100× for WavLM features, 1× for STFT) [1], the WavLM component dominates the total loss
- This imbalance appears to over-constrain the model, resulting in audible artifacts and spectral distortions

#### Challenge 2: Phoneme Alterations with Simplified Feature Extraction
To mitigate the artifact issue, we attempted using only the convolutional encoder layers without the feature projection components. While this approach successfully eliminates artifacts from the output spectrograms, it introduces a new problem:

**Observations:**
- Output spectrograms are clean and artifact-free
- However, phoneme preservation is compromised in some cases
- The model occasionally generates different phonemes than those present in the input speech
- This suggests insufficient linguistic constraint from the simplified feature space

#### Challenge 3: Artifacts with First Transformer Layer Features
As an alternative approach, we experimented with extracting features from the first transformer layer instead of the convolutional encoder, as the paper mentions both layers showed promising results [1].

**Observations:**
- Similar artifact patterns emerge as in Challenge 1
- The transformer layer features appear to over-constrain the model in a manner similar to the full feature projection pipeline

<!-- #### Open Questions
1. **Optimal Feature Extraction Strategy:** What is the most effective layer or combination of layers from WavLM for perceptual loss in speech enhancement?
2. **Loss Weight Balancing:** Should the 100:1 ratio between WavLM and STFT loss be adjusted for different feature extraction strategies?
3. **Feature Space Analysis:** How can we better understand the feature space structure to prevent both artifacts and phoneme alterations? -->

### Contributing
We invite the research community to help resolve these challenges, or alternative approaches to address these issues. If you have experience with:
- WavLM feature extraction and perceptual losses
- Speech enhancement model training and loss balancing
- Phoneme preservation techniques in generative models

Please feel free to:
- Open an issue to discuss potential solutions
- Submit a pull request with experimental results
- Share relevant research papers or techniques

Your insights and contributions could help improve the quality and robustness of this implementation.
