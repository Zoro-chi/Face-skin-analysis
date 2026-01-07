"""
Gradio App for Face Skin Condition Analysis
Upload an image to detect: acne, pigmentation, and wrinkles
"""

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path

from models.model_factory import create_model
from utils.config_loader import load_config
from explainability.gradcam import GradCAM
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SkinAnalysisApp:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = create_model(self.config)
        checkpoint_path = Path("outputs/checkpoints/best_model.pth")

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            print(f" Model loaded from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        # Load optimized thresholds
        threshold_path = Path("outputs/thresholds.json")
        if threshold_path.exists():
            with open(threshold_path, "r") as f:
                thresholds_data = json.load(f)
                self.thresholds = thresholds_data.get("thresholds", [0.5, 0.5, 0.5])
        else:
            self.thresholds = [0.5, 0.5, 0.5]  # Default thresholds

        print(
            f"Using thresholds: acne={self.thresholds[0]}, pigmentation={self.thresholds[1]}, wrinkles={self.thresholds[2]}"
        )

        # Condition names
        self.conditions = self.config["model"]["conditions"]

        # Image preprocessing
        self.img_size = self.config["preprocessing"]["image_size"]
        self.transform = A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        # Grad-CAM (optional - disable if there are issues)
        try:
            self.gradcam = GradCAM(self.model, self.config)
            self.gradcam_enabled = True
        except Exception as e:
            print(f"⚠️ Grad-CAM disabled due to initialization error: {e}")
            self.gradcam = None
            self.gradcam_enabled = False

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        img_tensor = transformed["image"].unsqueeze(0)

        return img_tensor, image

    def predict(self, image, show_heatmap=True):
        """Run inference on image"""
        try:
            # Preprocess
            img_tensor, original_img = self.preprocess_image(image)
            img_tensor = img_tensor.to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            # Apply thresholds
            predictions = (probs >= np.array(self.thresholds)).astype(int)

            # Format results for Gradio Label component (expects dict with numeric values)
            results = {}
            detailed_results = []
            for i, condition in enumerate(self.conditions):
                confidence = float(probs[i])
                # Label component expects: {label: confidence_score}
                results[
                    f"{condition.capitalize()} ({'✅' if predictions[i] == 1 else '❌'})"
                ] = confidence
                detailed_results.append(
                    {
                        "condition": condition.capitalize(),
                        "detected": bool(predictions[i]),
                        "confidence": confidence,
                        "threshold": self.thresholds[i],
                    }
                )

            # Generate heatmap if requested
            heatmap = None
            if show_heatmap and self.gradcam_enabled:
                # Generate Grad-CAM for detected conditions
                detected_conditions = [
                    i for i, pred in enumerate(predictions) if pred == 1
                ]
                if detected_conditions:
                    try:
                        # Use first detected condition for heatmap
                        target_class = detected_conditions[0]
                        heatmap = self.gradcam.generate_heatmap(
                            img_tensor, target_class=target_class, alpha=0.4
                        )
                        # Convert to PIL for Gradio
                        heatmap = Image.fromarray(heatmap)
                    except Exception as e:
                        print(f"⚠️ Heatmap generation failed: {e}")
                        heatmap = None

            # Create summary text
            summary = "## 🔍 Analysis Results\n\n"
            for item in detailed_results:
                icon = "🔴" if item["detected"] else "🟢"
                summary += f"{icon} **{item['condition']}**: "
                summary += f"{'Detected' if item['detected'] else 'Not Detected'} "
                summary += f"(Confidence: {item['confidence']:.1%}, Threshold: {item['threshold']:.1%})\n\n"

            return results, heatmap, summary

        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)
            return {"Error": error_msg}, None, error_msg


def create_gradio_interface():
    """Create Gradio interface"""

    # Initialize app
    app = SkinAnalysisApp()

    # Create interface
    with gr.Blocks(title="Face Skin Condition Analysis") as demo:
        gr.Markdown(
            """
            # 🔬 Face Skin Condition Analysis
            Upload a face image to detect **acne**, **pigmentation**, and **wrinkles**.
            
            The model uses optimized thresholds for each condition and provides confidence scores.
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil", label="Upload Face Image", height=400
                )
                show_heatmap = gr.Checkbox(
                    label="Show Grad-CAM Heatmap (Explainability)", value=True
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Image", variant="primary", size="lg"
                )

                gr.Markdown(
                    """
                    ### 💡 Tips:
                    - Upload a clear, well-lit face image
                    - Frontal view works best
                    - Model automatically detects faces
                    """
                )

            with gr.Column():
                output_labels = gr.Label(label="Predictions", num_top_classes=3)
                output_heatmap = gr.Image(
                    label="Grad-CAM Heatmap (Shows model focus areas)", height=400
                )
                output_summary = gr.Markdown(label="Detailed Results")

        gr.Markdown(
            """
            ---
            ### 📊 Model Information
            - **Architecture**: EfficientNet-B3
            - **Conditions Detected**: Acne, Pigmentation, Wrinkles
            - **Thresholds**: Optimized per-class (acne=0.15, pigmentation=0.50, wrinkles=0.20)
            - **Performance**: F1=0.81, AUROC=0.99
            """
        )

        # Examples
        gr.Examples(
            examples=(
                [
                    [
                        "data/processed/kaggle_face_skin_diseases/testing/Acne/acne_001.jpg"
                    ],
                ]
                if Path(
                    "data/processed/kaggle_face_skin_diseases/testing/Acne"
                ).exists()
                else []
            ),
            inputs=input_image,
            label="📸 Example Images (if available)",
        )

        # Connect button
        analyze_btn.click(
            fn=app.predict,
            inputs=[input_image, show_heatmap],
            outputs=[output_labels, output_heatmap, output_summary],
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True,
    )
