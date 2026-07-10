import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models

# Define device
DEVICE = torch.device("cpu")

# Model definitions (MLPs)
class MLPAudioClassifier(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=128, num_classes=7, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

class MLPFaceClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

class JointMLPFusionClassifier(nn.Module):
    def __init__(self, input_dim=1319, hidden_dim=256, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

# Combined Face Pipeline for easy ONNX execution
class CombinedFacePipeline(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return feats, logits

def main():
    os.makedirs("onnx_models", exist_ok=True)
    
    # 1. Export Audio MLP
    print("Exporting Audio MLP...")
    audio_ckpt = torch.load("models/audio_baseline/mlp_mfcc_baseline.pt", map_location=DEVICE)
    audio_model = MLPAudioClassifier(input_dim=39, num_classes=7)
    audio_model.load_state_dict(audio_ckpt["model_state_dict"])
    audio_model.eval()
    
    dummy_audio = torch.randn(1, 39)
    torch.onnx.export(
        audio_model,
        dummy_audio,
        "onnx_models/audio_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    # Save statistics
    np.savez("onnx_models/audio_stats.npz", mean=audio_ckpt["mean"], std=audio_ckpt["std"])
    print("Audio model exported.")

    # 2. Export Face combined pipeline (ResNet18 backbone + Face MLP)
    print("Exporting Combined Face model...")
    face_ckpt = torch.load("models/face/resnet18_face_baseline.pt", map_location=DEVICE)
    face_model = MLPFaceClassifier(input_dim=512, num_classes=7)
    face_model.load_state_dict(face_ckpt["model_state_dict"])
    face_model.eval()
    
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Identity()
    resnet18.eval()
    
    combined_face = CombinedFacePipeline(resnet18, face_model)
    combined_face.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        combined_face,
        dummy_image,
        "onnx_models/face_pipeline.onnx",
        input_names=["input_image"],
        output_names=["features", "logits"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=11
    )
    # Save statistics
    np.savez("onnx_models/face_stats.npz", mean=face_ckpt["mean"], std=face_ckpt["std"])
    print("Face combined pipeline exported.")

    # 3. Export Early Fusion MLP
    print("Exporting Early Fusion model...")
    early_ckpt = torch.load("models/fusion/early_fusion_model.pt", map_location=DEVICE)
    early_fusion_model = JointMLPFusionClassifier(input_dim=1319, num_classes=7)
    early_fusion_model.load_state_dict(early_ckpt["model_state_dict"])
    early_fusion_model.eval()
    
    dummy_fusion = torch.randn(1, 1319)
    torch.onnx.export(
        early_fusion_model,
        dummy_fusion,
        "onnx_models/early_fusion.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print("Early Fusion model exported.")
    
    # 4. Save Text Stats
    print("Saving text stats...")
    text_ckpt = torch.load("models/text_baseline/mlp_roberta_baseline.pt", map_location=DEVICE, weights_only=False)
    np.savez("onnx_models/text_stats.npz", mean=text_ckpt["mean"], std=text_ckpt["std"])
    
    # Save Late Fusion weights
    late_ckpt = torch.load("models/fusion/late_fusion_config.pt", map_location=DEVICE, weights_only=False)
    np.savez("onnx_models/late_fusion_weights.npz", weights=np.array(late_ckpt["best_weights"]))
    print("Text stats and Late Fusion weights saved successfully.")
    
    print("\nAll checkpoints exported to ONNX and NumPy files successfully in 'onnx_models/'!")

if __name__ == "__main__":
    main()
