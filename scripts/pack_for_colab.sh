#!/usr/bin/env bash
set -euo pipefail

# Pack the repository into a Colab-ready zip.
# Includes code, configs, notebook, and processed data.

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/outputs"
# mkdir -p "$OUT_DIR"

BUNDLE_NAME="face-skin-analysis.zip"
BUNDLE_PATH="$ROOT_DIR/$BUNDLE_NAME"

cd "$ROOT_DIR"

echo "Creating bundle: $BUNDLE_PATH"

# Build zip with selected directories and files
zip -r "$BUNDLE_PATH" \
  configs \
  evaluation \
  explainability \
  inference \
  models \
  onnx \
  preprocessing \
  training \
  utils \
  notebooks/colab_train_fixed.ipynb \
  requirements.txt \
  README.md \
  Face_Skin_Condition_Analysis_Project.md \
  data/processed \
  -x "**/__pycache__/**" "**/*.pyc" "**/.DS_Store" "mlruns/**" "outputs/logs/**"

echo "Bundle created at: $BUNDLE_PATH"
echo "Upload this zip to Google Drive, then mount Drive in Colab and unzip:"
echo "\nIn Colab cell:"
echo "from google.colab import drive\ndrive.mount('/content/drive')\n!unzip -q /content/drive/MyDrive/$BUNDLE_NAME -d /content\n%cd /content/Face-skin-analysis\n!pip install -r requirements.txt\n!python training/train.py --config configs/config.yaml --device cuda"
