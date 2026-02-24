# Informed Meta-Learning with INPs

# Informed Neural Processes (INPs)

Implementation of *Towards Automated Knowledge Integration From Human-Interpretable Representations* (Kobalczyk & van der Schaar, ICLR 2025), extended to medical image classification on dermatology datasets.

This repo contains:

- **Meta-regression** on synthetic sinusoids and temperature data (original paper experiments)
- **Medical image classification** (N-way, k-shot) on ISIC 2019 skin lesions, with optional DermNet auxiliary data for increased meta-training diversity

The classification pipeline uses frozen CLIP or BiomedCLIP embeddings with LLM-generated class descriptions as external knowledge (Setup C from the paper).

---

## Setup

### Environment

```bash
conda create -n inps python=3.11 -y
conda activate inps
pip install torch torchvision open-clip-torch==2.23.0 transformers==4.35.2 \
            wandb numpy toml Pillow
```

### Weights & Biases

Training logs to W&B. Set your credentials:

```bash
export WANDB_API_KEY="your_key"
export WANDB_ENTITY="your_entity"
```

Or run `wandb login`.

---

## Medical Image Classification

### Overview

The classification pipeline works in three stages:

1. **Encode** — Pre-compute frozen vision embeddings for all images (runs once per encoder)
2. **Train** — Episodic meta-training on N-way, k-shot tasks sampled from embeddings
3. **Evaluate** — Validation runs automatically during training; test on held-out classes

### Step 1: Get the Data

#### ISIC 2019

Download the ISIC 2019 training data:

```bash
# From https://challenge.isic-archive.com/data/#2019
# You need: ISIC_2019_Training_Input.zip and ISIC_2019_Training_GroundTruth.csv
mkdir -p data/isic2019
# Extract images to data/isic2019/ISIC_2019_Training_Input/
# Place ground truth as data/isic2019/ISIC_2019_Training_GroundTruth.csv
```

Expected layout:

```
data/isic2019/
├── ISIC_2019_Training_Input/
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ... (~25k images)
├── ISIC_2019_Training_GroundTruth.csv
└── descriptions.json              ← see Step 2
```

#### DermNet (optional, for auxiliary training data)

Download from Kaggle: https://www.kaggle.com/datasets/shubhamgoel27/dermnet

```bash
pip install kaggle
kaggle datasets download -d shubhamgoel27/dermnet -p data/dermnet
cd data/dermnet && unzip dermnet.zip && rm dermnet.zip
```

Expected layout:

```
data/dermnet/
├── train/
│   ├── Acne and Rosacea Photos/
│   ├── Atopic Dermatitis Photos/
│   └── ... (23 subfolders)
├── test/
│   └── ... (23 subfolders)
└── descriptions.json              ← see Step 2
```

### Step 2: Generate Class Descriptions

The model uses LLM-generated class descriptions as external knowledge (Setup C). Generate these with GPT-4 or Claude and save as JSON files.

**ISIC descriptions** — save to `data/isic2019/descriptions.json`:

```json
{
  "Melanoma": "An asymmetric lesion with irregular borders and multiple shades of brown, black, or red, often with regression areas or atypical vascular patterns.",
  "Melanocytic nevus": "A symmetric, round-to-oval lesion ...",
  "Basal cell carcinoma": "A pearly or waxy nodule ...",
  "Actinic keratosis": "...",
  "Benign keratosis": "...",
  "Dermatofibroma": "...",
  "Vascular lesion": "...",
  "Squamous cell carcinoma": "..."
}
```

Each description should be 2–3 sentences, under 60 words, focused on visual features (color, shape, texture, borders, distribution, body location). Stay within CLIP's 77-token limit (or 256 tokens for BiomedCLIP).

**DermNet descriptions** — save to `data/dermnet/descriptions.json` with the 19 non-overlapping class names as keys (the 4 classes that overlap with ISIC are automatically excluded).

### Step 3: Choose Your Encoder

Two vision-language encoders are supported:

| Encoder | Vision | Text | Image Size | Embed Dim | Best For |
|---------|--------|------|------------|-----------|----------|
| `clip` | ViT-B/32 | CLIP Transformer | 224×224 | 512 | General baseline |
| `biomedclip` | ViT-B/16 | PubMedBERT | 224×224 | 512 | Medical images |

**For BiomedCLIP**, download the weights first from huggingface.

### Step 4: Build Embedding Caches

Pre-compute frozen image embeddings (run once per encoder):

```bash
# CLIP embeddings
python dataset/isic.py --data-root data/isic2019 --encoder clip
python dataset/dermnet.py --data-root data/dermnet --encoder clip

# BiomedCLIP embeddings
python dataset/isic.py --data-root data/isic2019 --encoder biomedclip --biomedclip-dir data/biomedclip
python dataset/dermnet.py --data-root data/dermnet --encoder biomedclip --biomedclip-dir data/biomedclip
```

This produces `embeddings_clip.pt` and/or `embeddings_biomedclip.pt` in each data directory. On a single GPU, encoding takes about 5–10 minutes per dataset.

**On a SLURM cluster:**

```bash
sbatch encode_biomedclip.sh
```

### Step 5: Train

#### ISIC-only (4 meta-train classes)

```bash
python models/train_classification.py --config config_isic.toml
```

#### ISIC + DermNet mixed (4 + 15 = 19 meta-train classes)

```bash
python models/train_classification.py --config config_isic_dermnet.toml
```

**On a SLURM cluster:**

```bash
sbatch train_inp.sh
```

### Step 6: Monitor & Evaluate

Training logs to W&B with per-shot accuracy breakdown:

- `eval/accuracy_0shot` — zero-shot (knowledge only, no context images)
- `eval/accuracy_1shot` — 1 image per class as context
- `eval/accuracy_5shot` — 5 images per class
- `eval/accuracy_10shot` — 10 images per class

Checkpoints are saved to `saved_models/{project_name}/`. The best checkpoint (by val loss) is saved as `model_best.pth`.

---

## Configuration Reference

All training hyperparameters are set via TOML config files. Key classification-specific fields:

### Dataset

| Field | Default | Description |
|-------|---------|-------------|
| `dataset` | `"isic"` | `"isic"` for ISIC-only, `"isic_dermnet"` for mixed training |
| `data_root` | `"data/isic2019"` | Path to ISIC data |
| `dermnet_data_root` | `"data/dermnet"` | Path to DermNet data (mixed training only) |
| `encoder_type` | `"clip"` | `"clip"` or `"biomedclip"` |
| `biomedclip_dir` | `""` | Local path to BiomedCLIP weights |
| `dermnet_ratio` | `0.5` | Fraction of episodes from DermNet during mixed training |

### Episodes

| Field | Default | Description |
|-------|---------|-------------|
| `n_ways` | `2` | Classes per episode |
| `min_num_context` | `0` | Min shots per class (0 enables zero-shot episodes) |
| `max_num_context` | `10` | Max shots per class |
| `num_targets` | `20` | Query images per class per episode |
| `n_episodes_train` | `10000` | Episodes per training epoch |
| `n_episodes_val` | `1000` | Episodes for validation |

### Knowledge

| Field | Default | Description |
|-------|---------|-------------|
| `use_knowledge` | `true` | Enable Setup C knowledge descriptions |
| `knowledge_mask_rate` | `0.5` | Probability of masking knowledge during training (paper default: 0.5) |

### Training

| Field | Default | Description |
|-------|---------|-------------|
| `lr` | `1e-4` | Learning rate |
| `batch_size` | `32` | Episodes per batch |
| `num_epochs` | `100` | Training epochs |
| `decay_lr` | `10` | LR decay patience (epochs) |
| `beta` | `1.0` | KL weight in ELBO |
| `train_num_z_samples` | `1` | Latent samples during training |
| `test_num_z_samples` | `16` | Latent samples during evaluation |

### Model

| Field | Default | Description |
|-------|---------|-------------|
| `clip_dim` | `512` | Input embedding dimension |
| `hidden_dim` | `512` | Latent/hidden dimension |

---

## Class Splits

### ISIC 2019

| Split | Classes | Code |
|-------|---------|------|
| Train (4) | Melanoma, Melanocytic Nevus, Basal Cell Carcinoma, Actinic Keratosis | MEL, NV, BCC, AK |
| Val (2) | Benign Keratosis, Dermatofibroma | BKL, DF |
| Test (2) | Vascular Lesion, Squamous Cell Carcinoma | VASC, SCC |

### DermNet (19 classes, excluding 4 that overlap with ISIC)

| Split | Classes |
|-------|---------|
| Train (15) | Acne/Rosacea, Atopic Dermatitis, Cellulitis/Bacterial, Eczema, Exanthems/Drug Eruptions, Hair Loss/Alopecia, Herpes/HPV/STDs, Light Diseases/Pigmentation, Lupus/Connective Tissue, Poison Ivy/Contact Dermatitis, Scabies/Infestations, Systemic Disease, Tinea/Fungal, Vasculitis, Warts/Viral |
| Val (4) | Psoriasis/Lichen Planus, Bullous Disease, Nail Fungus/Nail Disease, Urticaria Hives |

---

## Meta-Regression (Original Paper)

### Synthetic Sinusoids

```bash
# Generate data
python dataset/generate_synt_data.py

# Train
python models/train.py --config config.toml

# Evaluate
# See evaluation/evaluate_sinusoids.ipynb
```

### Temperature Data

```bash
python models/train.py --config configs/temperature.toml
# See evaluation/evaluate_temperature.ipynb
```

---

## Citation

```bibtex
@inproceedings{kobalczyk2025towards,
  title={Towards Automated Knowledge Integration From Human-Interpretable Representations},
  author={Kobalczyk, Konrad and van der Schaar, Mihaela},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```