import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Multimodal Fusion: Feature-level (Early) & Decision-level (Late)
# This merges pre-extracted static features from Text, Audio, and Face,
# trains models in seconds, and compares performance.
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Load and merge feature sets
audio_path = Path("data/interim/ravdess_mfcc_features.csv")
text_path = Path("data/interim/ravdess_text_features.csv")
face_path = Path("data/interim/ravdess_face_features.csv")

for p in [audio_path, text_path, face_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing feature file: {p}. Run preprocessing first.")

audio_df = pd.read_csv(audio_path)
text_df = pd.read_csv(text_path)
face_df = pd.read_csv(face_path)

print(f"Loaded feature rows - Audio: {len(audio_df)}, Text: {len(text_df)}, Face: {len(face_df)}")

# Identify unique key metrics for matching
merge_cols = ["filename", "actor_id", "canonical_label", "split"]

# Merge datasets
merged_df = audio_df.merge(text_df, on=merge_cols, how="inner")
merged_df = merged_df.merge(face_df, on=merge_cols, how="inner")

print(f"Merged multimodal dataset: {len(merged_df)} rows")

label_col = "canonical_label"
labels = sorted(merged_df[label_col].unique())
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

merged_df["label_idx"] = merged_df[label_col].map(label2idx)

train_df = merged_df[merged_df["split"] == "train"].reset_index(drop=True)
val_df   = merged_df[merged_df["split"] == "val"].reset_index(drop=True)
test_df  = merged_df[merged_df["split"] == "test"].reset_index(drop=True)

# Separate feature columns
audio_cols = [c for c in merged_df.columns if c.startswith("mfcc_")]
text_cols  = [c for c in merged_df.columns if c.startswith("roberta_")]
face_cols  = [c for c in merged_df.columns if c.startswith("face_")]

print(f"Feature sizes - Audio: {len(audio_cols)}, Text: {len(text_cols)}, Face: {len(face_cols)}")

# ============================================================
# PART 1: Decision-level (Late) Fusion
# Grid search weights for: P_fused = w_a*P_a + w_t*P_t + w_f*P_f
# We reuse the trained unimodal checkpoint models to get probabilities.
# ============================================================

# Setup checkpoints path
audio_ckpt_path = Path("models/audio_baseline/mlp_mfcc_baseline.pt")
text_ckpt_path  = Path("models/text_baseline/mlp_roberta_baseline.pt")
face_ckpt_path  = Path("models/face/resnet18_face_baseline.pt")

for p in [audio_ckpt_path, text_ckpt_path, face_ckpt_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing unimodal checkpoint: {p}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definitions to load state_dicts
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

class MLPTextClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=7, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
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

# Load helper to load baseline probabilities
def get_model_probs(ckpt_path, model_cls, df, feature_cols):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = model_cls(input_dim=len(feature_cols), num_classes=7).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Extract normalizer stats
    mean_val = pd.Series(ckpt["mean"])[feature_cols]
    std_val  = pd.Series(ckpt["std"])[feature_cols]
    
    X = ((df[feature_cols] - mean_val) / std_val).values.astype(np.float32)
    X_tensor = torch.tensor(X).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

# Load val probabilities
a_val_probs = get_model_probs(audio_ckpt_path, MLPAudioClassifier, val_df, audio_cols)
t_val_probs = get_model_probs(text_ckpt_path, MLPTextClassifier, val_df, text_cols)
f_val_probs = get_model_probs(face_ckpt_path, MLPFaceClassifier, val_df, face_cols)

y_val = val_df["label_idx"].values

# Grid search for late fusion weights
print("\nOptimizing Late Fusion weights on validation set...")
best_val_acc = -1
best_weights = None

for w_a in np.linspace(0, 1, 101):
    for w_t in np.linspace(0, 1 - w_a, 101):
        w_f = 1.0 - w_a - w_t
        if w_f < -1e-6: continue
        w_f = max(0.0, w_f)
        
        preds = np.argmax(w_a * a_val_probs + w_t * t_val_probs + w_f * f_val_probs, axis=1)
        acc = accuracy_score(y_val, preds)
        
        if acc > best_val_acc:
            best_val_acc = acc
            best_weights = (w_a, w_t, w_f)

print(f"Optimal Late Fusion weights (Audio, Text, Face): {best_weights} with Val Acc: {best_val_acc:.4f}")

# Evaluate Late Fusion on Test Set
a_test_probs = get_model_probs(audio_ckpt_path, MLPAudioClassifier, test_df, audio_cols)
t_test_probs = get_model_probs(text_ckpt_path, MLPTextClassifier, test_df, text_cols)
f_test_probs = get_model_probs(face_ckpt_path, MLPFaceClassifier, test_df, face_cols)

y_test = test_df["label_idx"].values
w_a, w_t, w_f = best_weights
fused_test_probs = w_a * a_test_probs + w_t * t_test_probs + w_f * f_test_probs
late_fusion_preds = np.argmax(fused_test_probs, axis=1)

# Unimodal test performance (for comparisons)
a_test_preds = np.argmax(a_test_probs, axis=1)
t_test_preds = np.argmax(t_test_probs, axis=1)
f_test_preds = np.argmax(f_test_probs, axis=1)

print("\n" + "#"*60)
print("UNIMODAL & LATE FUSION EVALUATION ON TEST SET")
print("#"*60)
for name, preds in [("Audio Only", a_test_preds), ("Text Only", t_test_preds), ("Face Only", f_test_preds), ("Late Fusion", late_fusion_preds)]:
    acc = accuracy_score(y_test, preds)
    f1_m = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"{name:15s} | Accuracy: {acc:.4f} | Macro F1: {f1_m:.4f}")

# ============================================================
# PART 2: Feature-level (Early) Fusion
# We concatenate normalized features and train a Joint MLP.
# ============================================================

# Normalize features dynamically
mean_a, std_a = train_df[audio_cols].mean(), train_df[audio_cols].std().replace(0, 1e-6)
mean_t, std_t = train_df[text_cols].mean(), train_df[text_cols].std().replace(0, 1e-6)
mean_f, std_f = train_df[face_cols].mean(), train_df[face_cols].std().replace(0, 1e-6)

def build_early_feature(d):
    norm_a = ((d[audio_cols] - mean_a) / std_a).values
    norm_t = ((d[text_cols] - mean_t) / std_t).values
    norm_f = ((d[face_cols] - mean_f) / std_f).values
    X = np.concatenate([norm_a, norm_t, norm_f], axis=1).astype(np.float32)
    y = d["label_idx"].values.astype(np.int64)
    return torch.tensor(X), torch.tensor(y)

X_train_early, y_train_early = build_early_feature(train_df)
X_val_early, y_val_early = build_early_feature(val_df)
X_test_early, y_test_early = build_early_feature(test_df)

class EarlyFusionDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

joint_train_loader = DataLoader(EarlyFusionDataset(X_train_early, y_train_early), batch_size=16, shuffle=True)
joint_val_loader   = DataLoader(EarlyFusionDataset(X_val_early, y_val_early), batch_size=16)
joint_test_loader  = DataLoader(EarlyFusionDataset(X_test_early, y_test_early), batch_size=16)

class JointMLPFusionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=7, dropout=0.3):
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

input_dim_early = X_train_early.shape[1]
print(f"\nTraining Joint MLP fusion model on {input_dim_early}-dim concatenated features...")

joint_model = JointMLPFusionClassifier(input_dim=input_dim_early, num_classes=len(labels)).to(device)
optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-3, weight_decay=1e-3)

class_counts = train_df["label_idx"].value_counts().sort_index()
class_weights = torch.tensor((1.0 / class_counts).values, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(labels)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

EPOCHS = 150
best_val_f1 = -1
best_state = None
patience = 20
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    joint_model.train()
    total_loss = 0
    for X_batch, y_batch in joint_train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = joint_model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    train_loss = total_loss / len(train_df)

    joint_model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in joint_val_loader:
            X_batch = X_batch.to(device)
            logits = joint_model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(y_batch.numpy())
    val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
    val_acc = accuracy_score(val_true, val_preds)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Joint Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in joint_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping Joint Model at epoch {epoch}")
            break

print(f"\nBest val Joint macro-F1: {best_val_f1:.4f}")
joint_model.load_state_dict(best_state)

# Evaluate Early Fusion model on Test Set
joint_model.eval()
early_test_preds, early_test_true = [], []
with torch.no_grad():
    for X_batch, y_batch in joint_test_loader:
        X_batch = X_batch.to(device)
        logits = joint_model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        early_test_preds.extend(preds)
        early_test_true.extend(y_batch.numpy())

early_fusion_acc = accuracy_score(early_test_true, early_test_preds)
early_fusion_f1_macro = f1_score(early_test_true, early_test_preds, average="macro", zero_division=0)
early_fusion_f1_weighted = f1_score(early_test_true, early_test_preds, average="weighted", zero_division=0)

print("\n" + "="*60)
print("FEATURE-LEVEL EARLY FUSION (JOINT MLP) - TEST RESULTS")
print("="*60)
print(f"Test Accuracy:       {early_fusion_acc:.4f}")
print(f"Test Macro-F1:       {early_fusion_f1_macro:.4f}")
print(f"Test Weighted-F1:    {early_fusion_f1_weighted:.4f}")
print("\nPer-class report (Early Fusion):")
print(classification_report(early_test_true, early_test_preds, target_names=[idx2label[i] for i in range(len(labels))], zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(early_test_true, early_test_preds))

late_fusion_acc = accuracy_score(y_test, late_fusion_preds)

print("\n" + "="*60)
print("METRIC COMPARISON: EARLY FUSION VS LATE FUSION VS UNIMODAL")
print("="*60)
print(f"Audio Only Accuracy:     {accuracy_score(y_test, a_test_preds):.4f}")
print(f"Text Only Accuracy:      {accuracy_score(y_test, t_test_preds):.4f}")
print(f"Face Only Accuracy:      {accuracy_score(y_test, f_test_preds):.4f}")
print(f"Late Fusion Accuracy:    {late_fusion_acc:.4f}")
print(f"Early Fusion Accuracy:   {early_fusion_acc:.4f}")
print("="*60)

# Save Fusion checkpoints
Path("models/fusion").mkdir(parents=True, exist_ok=True)

# Save Late Fusion weights
torch.save({
    "best_weights": best_weights,
    "label2idx": label2idx,
    "idx2label": idx2label,
    "test_accuracy": late_fusion_acc,
    "test_macro_f1": f1_score(y_test, late_fusion_preds, average="macro", zero_division=0)
}, "models/fusion/late_fusion_config.pt")
print("Saved Late Fusion weights config to models/fusion/late_fusion_config.pt")

# Save Early Fusion model checkpoint
torch.save({
    "model_state_dict": best_state,
    "label2idx": label2idx,
    "idx2label": idx2label,
    "input_dim": input_dim_early,
    "audio_mean": mean_a.to_dict(), "audio_std": std_a.to_dict(),
    "text_mean": mean_t.to_dict(), "text_std": std_t.to_dict(),
    "face_mean": mean_f.to_dict(), "face_std": std_f.to_dict(),
    "test_accuracy": early_fusion_acc,
    "test_macro_f1": early_fusion_f1_macro,
}, "models/fusion/early_fusion_model.pt")
print("Saved Early Fusion joint MLP model to models/fusion/early_fusion_model.pt")
