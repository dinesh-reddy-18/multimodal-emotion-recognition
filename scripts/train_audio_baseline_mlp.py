import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ============================================================
# Audio baseline v2: MLP over static 39-dim MFCC feature vector.
# Replaces the flawed BiLSTM-over-coefficients approach — MFCC
# coefficients have no sequential relationship to each other,
# so an LSTM across them was learning noise. An MLP is the
# architecturally correct choice for a flat feature vector.
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv("data/interim/ravdess_mfcc_features.csv")

label_col = "canonical_label"
feature_cols = [c for c in df.columns if c.startswith("mfcc_")]
print(f"Found {len(feature_cols)} MFCC feature columns")

labels = sorted(df[label_col].unique())
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}
print(f"Classes ({len(labels)}): {labels}")

df["label_idx"] = df[label_col].map(label2idx)

train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "val"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)
print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

mean = train_df[feature_cols].mean()
std = train_df[feature_cols].std().replace(0, 1e-6)

def to_tensor(d):
    X = ((d[feature_cols] - mean) / std).values.astype(np.float32)
    y = d["label_idx"].values.astype(np.int64)
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = to_tensor(train_df)
X_val, y_val = to_tensor(val_df)
X_test, y_test = to_tensor(test_df)

class MFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(MFCCDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(MFCCDataset(X_val, y_val), batch_size=16)
test_loader = DataLoader(MFCCDataset(X_test, y_test), batch_size=16)

# ---- Model: simple MLP, correct for flat static features ----
class MLPAudioClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=7, dropout=0.4):
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

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTraining on device: {device}")

model = MLPAudioClassifier(input_dim=len(feature_cols), num_classes=len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# Class weights to counter imbalance (esp. neutral has half the samples)
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
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    train_loss = total_loss / len(train_df)

    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(y_batch.numpy())
    val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
    val_acc = accuracy_score(val_true, val_preds)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nBest val macro-F1: {best_val_f1:.4f}")
model.load_state_dict(best_state)

model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_true.extend(y_batch.numpy())

test_acc = accuracy_score(test_true, test_preds)
test_f1_macro = f1_score(test_true, test_preds, average="macro", zero_division=0)
test_f1_weighted = f1_score(test_true, test_preds, average="weighted", zero_division=0)

print("\n" + "="*60)
print("AUDIO BASELINE v2 (MLP on static MFCC) - TEST RESULTS")
print("="*60)
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Test Macro-F1:       {test_f1_macro:.4f}")
print(f"Test Weighted-F1:    {test_f1_weighted:.4f}")
print("\nPer-class report:")
print(classification_report(test_true, test_preds, target_names=[idx2label[i] for i in range(len(labels))], zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(idx2label)
print(confusion_matrix(test_true, test_preds))

Path("models/audio_baseline").mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": best_state,
    "label2idx": label2idx,
    "idx2label": idx2label,
    "feature_cols": feature_cols,
    "mean": mean.to_dict(),
    "std": std.to_dict(),
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1_macro,
}, "models/audio_baseline/mlp_mfcc_baseline.pt")
print("\nSaved model to models/audio_baseline/mlp_mfcc_baseline.pt")
