# Face & Skin Condition Analysis System

### Fair, Explainable AI for Facial Skin Condition Detection

[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://dagshub.com)
[![DVC](https://img.shields.io/badge/DVC-data%20versioning-orange)](https://dvc.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Zoro--chi/Face--skin--analysis-black?logo=github)](https://github.com/Zoro-chi/Face-skin-analysis)

## Overview

This project implements a **computer vision system** that detects facial skin conditions—**acne, hyperpigmentation, and wrinkles**—from images while explicitly addressing **model bias across skin tones**, with a strong focus on **darker skin types (Fitzpatrick V–VI)**.

### Key Features

- ✅ Multi-label facial skin condition detection
- ✅ Fine-tuned CNN and Vision Transformer models
- ✅ Skin-tone stratified bias analysis
- ✅ Confidence scoring per prediction
- ✅ Explainable heatmaps using Grad-CAM
- ✅ Experiment tracking with MLflow
- ✅ Dataset & model versioning with DVC
- ✅ Centralized collaboration via DagsHub
- ✅ ONNX-optimized inference pipeline

## Current Results (v1.0)

Latest model metrics (from `outputs/metrics.json`):

- **Overall** — Precision: 0.751, Recall: 0.889, F1: 0.811, AUROC: 0.986
- **Acne** — Precision: 0.545, Recall: 0.750, F1: 0.632, AUROC: 0.976
- **Pigmentation** — Precision: 0.908, Recall: 0.917, F1: 0.912, AUROC: 0.982
- **Wrinkles** — Precision: 0.800, Recall: 1.000, F1: 0.889, AUROC: 0.9998

Fairness highlights (evaluation stratified by Fitzpatrick groups):

- **Dark skin (Fitzpatrick V–VI):** F1 = 0.48 (current)
- **Medium skin (III–IV):** F1 = 0.75 (current)
- **Fairness gap:** 0.26 (target: reduce to <0.15)

Key mitigation steps applied:

- Balanced sampling using `WeightedRandomSampler` to boost exposure for underrepresented skin-tone groups.
- Per-group threshold optimization on validation sets to maximize F1 per Fitzpatrick group.
- Tone-preserving augmentations (CLAHE, geometric transforms) to avoid distorting skin color information.
- Experiment tracking with MLflow and dataset/version control with DVC for reproducibility.

For detailed analysis and reproducible steps see `docs/FAIRNESS_IMPROVEMENTS.md` and `blog_post.md`.

## System Architecture

```
Input Image
→ Face Detection & Alignment
→ Tone-Aware Preprocessing
→ Multi-Label Skin Condition Model
→ Predictions + Confidence
→ Grad-CAM Explainability
→ Bias & Fairness Evaluation
→ ONNX Export & Inference
```

## Project Structure

```
skin-analysis-ai/
├── data/                          # Data directory (DVC tracked)
│   ├── raw/                       # Original datasets
│   ├── processed/                 # Preprocessed data
│   └── augmented/                 # Augmented data
├── preprocessing/                 # Data preprocessing pipeline
│   ├── face_detection.py         # Face detection & alignment
│   ├── augmentation.py           # Data augmentation
│   └── run.py                    # Main preprocessing script
├── models/                        # Model architectures
│   ├── efficientnet.py           # EfficientNet-B3 model
│   ├── vit_model.py              # Vision Transformer model
│   └── base_model.py             # Base model class
├── training/                      # Training scripts
│   ├── train.py                  # Main training script
│   ├── losses.py                 # Custom loss functions
│   └── trainer.py                # Training logic
├── evaluation/                    # Evaluation scripts
│   ├── evaluate.py               # Main evaluation script
│   ├── metrics.py                # Metric calculations
│   └── bias_analysis.py          # Fairness analysis
├── explainability/                # Explainability module
│   ├── gradcam.py                # Grad-CAM implementation
│   └── visualize.py              # Visualization utilities
├── inference/                     # Inference pipeline
│   ├── predict.py                # Single image prediction
│   └── batch_predict.py          # Batch inference
├── onnx/                          # ONNX export & optimization
│   ├── export.py                 # PyTorch to ONNX conversion
│   └── optimize.py               # ONNX optimization
├── utils/                         # Utility functions
│   ├── logger.py                 # Logging configuration
│   ├── config_loader.py          # Config file loader
│   └── helpers.py                # Helper functions
├── configs/                       # Configuration files
│   └── config.yaml               # Main config file
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── outputs/                       # Output directory
│   ├── checkpoints/              # Model checkpoints
│   ├── explainability/           # Grad-CAM visualizations
│   ├── logs/                     # Training logs
│   └── plots/                    # Evaluation plots
├── mlruns/                        # MLflow artifacts
├── dvc.yaml                       # DVC pipeline
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## Installation

### Option 1: Local Development (Mac/Linux/Windows)

#### 1. Clone Repository

```bash
git clone https://dagshub.com/<username>/skin-analysis-ai.git
cd skin-analysis-ai
```

#### 2. Create Conda Environment

```bash
conda create -n face-analysis python=3.12
conda activate face-analysis
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your DagsHub credentials
```

#### 5. Initialize DVC

```bash
dvc init
dvc remote add -d dagshub https://dagshub.com/<username>/skin-analysis-ai.dvc
```

### Option 2: Google Colab (GPU Training)

**Recommended for model training with GPU acceleration!**

1. Open [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Google Colab
2. Select GPU runtime (Runtime → Change runtime type → GPU)
3. Follow notebook instructions

📖 **Detailed Colab workflow:** See [docs/COLAB_WORKFLOW.md](docs/COLAB_WORKFLOW.md)

### Hybrid Workflow (Recommended) 🌟

```
1. Preprocessing → Local (CPU)      ✅ Your Mac
2. Training     → Colab (GPU)       🚀 Free GPU
3. Evaluation   → Local (MLflow)    ✅ Your Mac
```

See [docs/COLAB_WORKFLOW.md](docs/COLAB_WORKFLOW.md) for complete guide.

## Usage

### Data Preprocessing

```bash
# Pull raw data (if using DVC)
dvc pull

# Run preprocessing pipeline
python preprocessing/run.py
```

### Training

```bash
# Train the model
python training/train.py

# With custom config
python training/train.py --config configs/custom_config.yaml
```

### Evaluation

```bash
# Evaluate model performance
python evaluation/evaluate.py

# Bias analysis
python evaluation/bias_analysis.py
```

### Explainability

```bash
# Generate Grad-CAM heatmaps
python explainability/gradcam.py --image path/to/image.jpg
```

### Inference

```bash
# Single image prediction
python inference/predict.py --image sample.jpg

# Batch prediction
python inference/batch_predict.py --input_dir images/ --output_dir results/
```

### ONNX Export

```bash
# Export model to ONNX
python onnx/export.py

# Optimize ONNX model
python onnx/optimize.py
```

## MLOps Workflow

### DVC Pipeline

```bash
# Run full DVC pipeline
dvc repro

# Run specific stage
dvc repro evaluate
```

### MLflow Tracking

```bash
# View MLflow UI
mlflow ui

# Or access DagsHub MLflow UI
# https://dagshub.com/<username>/skin-analysis-ai/experiments
```

## Tech Stack

- **Core ML**: PyTorch, EfficientNet-B3, Vision Transformer
- **Computer Vision**: OpenCV, Albumentations
- **Explainability**: Grad-CAM
- **MLOps**: MLflow, DVC, DagsHub
- **Optimization**: ONNX, ONNX Runtime
- **API**: FastAPI (optional)

## Datasets

- DermNet
- Fitzpatrick17k
- Acne04
- CelebA

## Model Performance

| Metric    | Overall | Light Skin | Medium Skin | Dark Skin |
| --------- | ------- | ---------- | ----------- | --------- |
| Precision | 0.751   | 0.54       | 0.83        | 0.50      |
| Recall    | 0.889   | 0.57       | 0.74        | 0.47      |
| F1-Score  | 0.811   | 0.5525     | 0.7479      | 0.4843    |
| AUROC     | 0.986   | TBD        | TBD         | TBD       |

## Ethical Considerations

⚠️ **Important**: This is a research and educational project.

- Biases are explicitly documented
- **Not a medical diagnostic tool**
- Research and educational use only
- Always consult healthcare professionals for medical advice

## Future Improvements

- [ ] Severity grading for each condition
- [ ] Mobile optimization (TensorFlow Lite)
- [ ] Dermatologist-style reports
- [ ] Additional skin conditions
- [ ] Multi-language support
- [ ] API deployment with FastAPI
