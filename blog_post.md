# Face & Skin Condition Analysis — Fair, Explainable, Production-ready AI

## TL;DR

Face & Skin Condition Analysis is a production-oriented computer vision system that detects acne, hyperpigmentation, and wrinkles while explicitly targeting fairness across skin tones (Fitzpatrick I–VI). This post covers the problem, architecture, fairness-first training strategies, results, and how to get started with the code and models.

## Introduction

Medical AI systems often hide dangerous disparities behind high overall accuracy. In this project I built a multi-label model (EfficientNet-B3 / ViT) and an MLOps pipeline to prioritize equitable performance—especially for darker skin tones (Fitzpatrick V–VI). The goal: a practical, reproducible system that balances strong metrics with interpretability and fairness.

## Why this project

- Dermatology models frequently underperform on darker skin tones.
- Clinical applications require transparent, explainable outputs.
- Reproducibility is essential: experiments, data, and checkpoints should be versioned.

This repo demonstrates concrete strategies for measuring and reducing bias while keeping the system production ready (ONNX inference, MLflow tracking, DVC).

## Getting started (quick)

### Prerequisites

1. Python 3.8+
2. GPU recommended for training (CUDA)
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Quick inference (example)

```bash
python inference/predict.py --image path/to/face.jpg --checkpoint outputs/checkpoints/best_model.pth
```

Or use the ONNX export for faster inference in production: see `onnx/export.py`.

## System overview

Pipeline:

```
Input Image → Face Detection → Tone-aware Preprocessing → Multi-label Model
→ Predictions + Confidence → Grad-CAM Explainability → Fairness Evaluation → ONNX Export
```

Key modules:

- `preprocessing/` — face detection, alignment, tone-aware augmentation
- `models/` — EfficientNet / ViT implementations and model factory
- `training/` — dataset, dataloader, losses, trainer
- `evaluation/` — metrics, bias analysis, per-group reporting
- `explainability/` — Grad-CAM utilities

## Fairness-first training strategies

1. Balanced sampling: use `WeightedRandomSampler` to upsample underrepresented skin-tone groups during training.

```python
# training/data_loader.py — compute inverse-frequency weights per skin-tone
# sampler = WeightedRandomSampler(weights, num_samples)
```

2. Per-group threshold optimization: compute optimal decision thresholds on validation sets per Fitzpatrick group.

3. Tone-preserving augmentation: prefer geometric and CLAHE-style transforms; avoid aggressive color jitter.

4. Stratified evaluation: track precision/recall/F1 and AUROC per skin-tone group.

## Results & limitations

- Overall AUROC: 0.97
- Pigmentation F1: 0.91
- Acne F1: 0.64
- Wrinkles F1: 0.33 (limited data: 27 samples)

Fairness gap (initial): Dark (V–VI) F1 = 0.48 vs Medium (III–IV) F1 = 0.75 → gap = 0.26. Current work aims to reduce this to <0.15 via balanced sampling and threshold tuning.

### Balanced sampling (concrete example)

```python
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

# assume metadata.csv has a `fitzpatrick_group` column with values like 'I-II','III-IV','V-VI'
meta = pd.read_csv('data/processed/metadata.csv')
group_counts = meta['fitzpatrick_group'].value_counts().to_dict()

# inverse frequency weight per sample
weights = meta['fitzpatrick_group'].map(lambda g: 1.0 / group_counts[g]).values
sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)

# pass `sampler` to DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
```

This keeps the dataset on disk unchanged while balancing exposure during training.

### Per-group threshold optimization

Find thresholds that maximize F1 on validation per skin-tone group (example uses sklearn):

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

def best_threshold_for_f1(y_true, y_scores):
	precisions, recalls, threshs = precision_recall_curve(y_true, y_scores)
	f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
	best = np.nanargmax(f1)
	return threshs[best]

# compute per-group
group_thresholds = {}
for group in meta['fitzpatrick_group'].unique():
	idx = meta['fitzpatrick_group'] == group
	group_thresholds[group] = best_threshold_for_f1(y_val[idx], y_scores_val[idx])

# during inference, apply group_thresholds[detected_group]
```

### Tone-preserving augmentation

Aggressive color transforms can change skin tone. Use geometric transforms and CLAHE for contrast instead. Example augment pipeline using torchvision + a small custom CLAHE step with OpenCV:

```python
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

class CLAHETransform:
	def __call__(self, img: Image.Image) -> Image.Image:
		arr = np.array(img)
		if arr.ndim == 3:
			lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
			l, a, b = cv2.split(lab)
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			l2 = clahe.apply(l)
			lab = cv2.merge((l2,a,b))
			arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
		return Image.fromarray(arr)

train_transforms = transforms.Compose([
	transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
	transforms.RandomHorizontalFlip(),
	CLAHETransform(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

### Grad-CAM (usage snippet)

Using `pytorch-grad-cam` to produce explainability heatmaps:

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms.functional as F

model.eval()
target_layer = model.features[-1]  # depends on architecture
cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

img = Image.open('path/to/face.jpg').convert('RGB')
inp = preprocess(img).unsqueeze(0).cuda()
grayscale_cam = cam(input_tensor=inp)[0]
rgb_img = np.array(img.resize((224,224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
Image.fromarray(visualization).save('outputs/explainability/cam_overlay.png')
```

### MLflow logging (minimal)

```python
import mlflow

mlflow.set_experiment('face-skin-analysis')
with mlflow.start_run():
	mlflow.log_param('model', 'efficientnet-b3')
	mlflow.log_param('balance_skin_tones', True)
	mlflow.log_metric('val_auc', 0.97)
	mlflow.log_artifact('outputs/checkpoints/best_model.pth')
```

### ONNX export & inference (quick)

```python
# export
import torch
dummy = torch.randn(1,3,224,224).cuda()
torch.onnx.export(model, dummy, 'outputs/onnx/model.onnx', opset_version=13, input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}})

# inference with onnxruntime
import onnxruntime as ort
ort_sess = ort.InferenceSession('outputs/onnx/model.onnx')
outputs = ort_sess.run(None, {'input': img_tensor.cpu().numpy()})
```

## How this is organized for reproducibility

- MLflow for experiments (`mlruns/`) — compare runs and artifacts
- DVC for data and pipeline reproducibility (`dvc.yaml`, DVC-tracked data)
- Checkpoints and ONNX models in `outputs/`

## Roadmap

Phase 1 — Fairness first (current): stabilize 3 conditions, reduce fairness gaps.

Phase 2 — Data expansion: add Fitzpatrick17k and other public datasets to improve representation.

Phase 3 — Feature expansion: severity scoring, more conditions, deployment-ready API.

## Quick notes for contributors

- If you add dataset samples, please update `data/processed/metadata.csv` and run `dvc repro`.
- Use MLflow tags for run metadata: `skin_tone_group`, `balance_skin_tones`, `model_arch`.

## Conclusion

High accuracy is not enough for medical AI. This project shows a practical path to building ML systems that are fairer and more explainable while remaining production-ready and reproducible. Contributions and critiques welcome.

---

**Recommended frontmatter for Sanity**

- Title: "Face & Skin Condition Analysis — Fair, Explainable, Production-ready AI"
- Excerpt: "A fairness-first computer vision system for detecting acne, pigmentation, and wrinkles with per-skin-tone evaluation and Grad-CAM explainability."
- Category: `Project Showcase` or `AI/ML`
- Tags: `ComputerVision, FairAI, ExplainableAI, Dermatology, MLOps`

---

_Tags: #ComputerVision #MLOps #FairAI #ExplainableAI #Dermatology #PyTorch #MLflow #DVC #ResponsibleAI_
