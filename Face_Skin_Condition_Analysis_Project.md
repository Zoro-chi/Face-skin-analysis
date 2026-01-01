
# Face & Skin Condition Analysis System
### Fair, Explainable AI for Facial Skin Condition Detection

---

## 1. Project Overview

This project implements a **computer vision system** that detects facial skin conditions—**acne, hyperpigmentation, and wrinkles**—from images while explicitly addressing **model bias across skin tones**, with a strong focus on **darker skin types (Fitzpatrick V–VI)**.

Unlike typical CV demos, this system is built with:
- **Fairness-aware evaluation**
- **Explainability (Grad-CAM)**
- **Confidence-calibrated predictions**
- **Reproducible MLOps practices**
- **Production-ready inference (ONNX)**

The goal is to demonstrate **real-world ML system design**, not just model accuracy.

---

## 2. Key Features

- Multi-label facial skin condition detection
- Fine-tuned CNN and Vision Transformer models
- Skin-tone stratified bias analysis
- Confidence scoring per prediction
- Explainable heatmaps using Grad-CAM
- Experiment tracking with MLflow
- Dataset & model versioning with DVC
- Centralized collaboration via DagsHub
- ONNX-optimized inference pipeline

---

## 3. System Architecture

Input Image  
→ Face Detection & Alignment  
→ Tone-Aware Preprocessing  
→ Multi-Label Skin Condition Model  
→ Predictions + Confidence  
→ Grad-CAM Explainability  
→ Bias & Fairness Evaluation  
→ ONNX Export & Inference

---

## 4. Tech Stack

### Core ML & CV
- PyTorch
- EfficientNet-B3
- Vision Transformer (ViT)
- OpenCV

### Explainability
- Grad-CAM

### MLOps & Reproducibility
- MLflow
- DVC
- DagsHub

### Inference & Optimization
- ONNX
- FastAPI (optional)

---

## 5. Dataset Strategy

### Public Datasets
- DermNet
- Fitzpatrick17k
- Acne04
- CelebA

### Skin Tone Grouping (Fitzpatrick Scale)
- Light: I–II
- Medium: III–IV
- Dark: V–VI

---

## 6. Preprocessing Pipeline

- Face detection (OpenCV / MTCNN)
- Resize to 224x224
- ImageNet normalization
- CLAHE for contrast enhancement
- Tone-aware augmentation

---

## 7. Model Design

### Task Formulation
Multi-label classification:
- Acne
- Pigmentation
- Wrinkles

Sigmoid outputs with Binary Cross-Entropy loss and class weighting.

---

## 8. Training & Evaluation

### Metrics
- Precision
- Recall
- F1-score
- AUROC

Evaluated overall and per skin-tone group.

---

## 9. Bias & Fairness

- Skin-tone stratified metrics
- False Negative Rate analysis
- Group-weighted loss & augmentation

---

## 10. Confidence Scoring

- Monte Carlo Dropout
- Temperature scaling

Each output includes probability and confidence score.

---

## 11. Explainability (Grad-CAM)

Grad-CAM heatmaps are generated per condition and overlaid on facial images.

---

## 12. MLOps: MLflow, DVC & DagsHub

- MLflow tracks experiments, metrics, and artifacts
- DVC versions datasets and models
- DagsHub hosts MLflow UI and DVC storage

Ensures full reproducibility.

---

## 13. Project Structure

skin-analysis-ai/
├── data/
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── explainability/
├── inference/
├── onnx/
├── notebooks/
├── dvc.yaml
├── requirements.txt
└── PROJECT.md

---

## 14. How to Build & Run

### Clone Repository
git clone https://dagshub.com/<username>/skin-analysis-ai.git

### Install Dependencies
pip install -r requirements.txt

### Pull Data
dvc pull

### Preprocess
python preprocessing/run.py

### Train
python training/train.py

### Evaluate
python evaluation/evaluate.py

### Explainability
python explainability/gradcam.py

### Export ONNX
python onnx/export.py

### Inference
python inference/predict.py --image sample.jpg

---

## 15. Ethical Considerations

- Biases are explicitly documented
- Not a medical diagnostic tool
- Research and educational use only

---

## 16. Future Improvements

- Severity grading
- Mobile optimization
- Dermatologist-style reports
- Additional skin conditions

---

## 17. Why This Project Matters

This project demonstrates:
- Advanced CV skills
- Fairness-aware ML
- Explainable AI
- Reproducible MLOps
- Production-ready design
