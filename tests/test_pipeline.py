"""
Integration and smoke tests for the Multimodal Emotion Recognition pipeline.
Run with:  venv\\Scripts\\python.exe -m pytest tests/ -v
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import pytest

# ============================================================
# TEST 1: Checkpoint existence
# ============================================================

def test_audio_checkpoint_exists():
    assert Path("models/audio_baseline/mlp_mfcc_baseline.pt").exists(), \
        "Audio baseline checkpoint missing."

def test_text_checkpoint_exists():
    assert Path("models/text_baseline/mlp_roberta_baseline.pt").exists(), \
        "Text baseline checkpoint missing."

def test_face_checkpoint_exists():
    assert Path("models/face/resnet18_face_baseline.pt").exists(), \
        "Face baseline checkpoint missing."

def test_early_fusion_checkpoint_exists():
    assert Path("models/fusion/early_fusion_model.pt").exists(), \
        "Early fusion checkpoint missing."

def test_late_fusion_checkpoint_exists():
    assert Path("models/fusion/late_fusion_config.pt").exists(), \
        "Late fusion config missing."

# ============================================================
# TEST 2: Feature CSVs exist and have correct columns
# ============================================================

def test_mfcc_features_csv():
    p = Path("data/interim/ravdess_mfcc_features.csv")
    assert p.exists()
    df = pd.read_csv(p)
    mfcc_cols = [c for c in df.columns if c.startswith("mfcc_")]
    assert len(mfcc_cols) == 39, f"Expected 39 MFCC cols, got {len(mfcc_cols)}"
    assert "canonical_label" in df.columns
    assert "split" in df.columns

def test_text_features_csv():
    p = Path("data/interim/ravdess_text_features.csv")
    assert p.exists()
    df = pd.read_csv(p)
    roberta_cols = [c for c in df.columns if c.startswith("roberta_")]
    assert len(roberta_cols) == 768, f"Expected 768 RoBERTa cols, got {len(roberta_cols)}"

def test_face_features_csv():
    p = Path("data/interim/ravdess_face_features.csv")
    assert p.exists()
    df = pd.read_csv(p)
    face_cols = [c for c in df.columns if c.startswith("face_")]
    assert len(face_cols) == 512, f"Expected 512 face feature cols, got {len(face_cols)}"

# ============================================================
# TEST 3: Train/Val/Test split sanity check
# ============================================================

def test_split_assignments():
    df = pd.read_csv("data/interim/ravdess_mfcc_features.csv")
    splits = df["split"].unique().tolist()
    for s in ["train", "val", "test"]:
        assert s in splits, f"Split '{s}' not found in feature CSV."

    train_actors = list(range(1, 17))
    val_actors   = list(range(17, 21))
    test_actors  = list(range(21, 25))

    split_df = pd.read_csv("data/interim/ravdess_split_index.csv")
    for a in split_df["actor_id"].unique():
        row = split_df[split_df["actor_id"] == a]["split"].iloc[0]
        if a in train_actors:
            assert row == "train", f"Actor {a} expected in train, got {row}"
        elif a in val_actors:
            assert row == "val", f"Actor {a} expected in val, got {row}"
        elif a in test_actors:
            assert row == "test", f"Actor {a} expected in test, got {row}"

# ============================================================
# TEST 4: Unimodal checkpoint loading and output shape
# ============================================================

DEVICE = torch.device("cpu")

class MLPAudioClassifier(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=128, num_classes=7, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

class MLPFaceClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

class JointMLPFusionClassifier(nn.Module):
    def __init__(self, input_dim=1319, hidden_dim=256, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

def test_audio_model_output_shape():
    ckpt = torch.load("models/audio_baseline/mlp_mfcc_baseline.pt", map_location=DEVICE, weights_only=False)
    model = MLPAudioClassifier(input_dim=39, num_classes=7)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dummy = torch.randn(4, 39)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, 7), f"Expected (4, 7), got {out.shape}"

def test_face_model_output_shape():
    ckpt = torch.load("models/face/resnet18_face_baseline.pt", map_location=DEVICE, weights_only=False)
    model = MLPFaceClassifier(input_dim=512, num_classes=7)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dummy = torch.randn(4, 512)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, 7), f"Expected (4, 7), got {out.shape}"

def test_early_fusion_model_output_shape():
    ckpt = torch.load("models/fusion/early_fusion_model.pt", map_location=DEVICE, weights_only=False)
    input_dim = ckpt["input_dim"]
    model = JointMLPFusionClassifier(input_dim=input_dim, num_classes=7)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dummy = torch.randn(4, input_dim)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, 7), f"Expected (4, 7), got {out.shape}"

# ============================================================
# TEST 5: MFCC preprocessing function smoke test
# ============================================================

def test_mfcc_feature_shape():
    """Extract MFCC from a real WAV file if it exists, else skip."""
    wav_dirs = list(Path("data/interim/audio_wav").rglob("*.wav"))
    if not wav_dirs:
        pytest.skip("No .wav files found – audio extraction may not have run.")
    
    import librosa
    y, sr = librosa.load(str(wav_dirs[0]), sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.concatenate([mfcc.mean(axis=1), delta.std(axis=1), delta2.std(axis=1)])
    assert features.shape == (39,), f"Expected (39,), got {features.shape}"

# ============================================================
# TEST 6: Late Fusion weight loading
# ============================================================

def test_late_fusion_weights():
    ckpt = torch.load("models/fusion/late_fusion_config.pt", map_location=DEVICE, weights_only=False)
    w = ckpt["best_weights"]
    assert len(w) == 3, "Expected 3 weights (audio, text, face)"
    total = sum(float(x) for x in w)
    assert abs(total - 1.0) < 1e-4, f"Weights should sum to ~1.0, got {total}"
