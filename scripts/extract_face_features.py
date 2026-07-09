import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Extract 512-dim ResNet18 features for facial video crops
# For each video, we average the feature vectors of its 10 frames.
# ============================================================

splits_path = Path("data/interim/ravdess_split_index.csv")
if not splits_path.exists():
    raise FileNotFoundError(f"{splits_path} not found. Run splits scripts first.")

df = pd.read_csv(splits_path)
df_av_speech = df[(df["channel"] == "speech") & (df["modality"] == "full_AV")].copy()

# Setup ResNet18 extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

resnet = models.resnet18(pretrained=True)
# Truncate classifier to get the final pooling features (512-dim)
resnet.fc = nn.Identity() 
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

crops_root = Path("data/interim/face_crops")
feature_records = []
missing_count = 0

print("Extracting facial features...")
with torch.no_grad():
    for idx, row in tqdm(df_av_speech.iterrows(), total=len(df_av_speech)):
        video_stem = Path(row["filepath"]).stem
        actor_folder = f"Actor_{row['actor_id']:02d}"
        video_crop_dir = crops_root / actor_folder / video_stem
        
        if not video_crop_dir.exists():
            missing_count += 1
            continue
            
        frames = sorted(list(video_crop_dir.glob("frame_*.jpg")))
        if len(frames) == 0:
            missing_count += 1
            continue
            
        # Batched inference for frames of this video clip
        batch_tensors = []
        for img_path in frames:
            img = Image.open(img_path).convert("RGB")
            batch_tensors.append(transform(img))
            
        x = torch.stack(batch_tensors).to(device) # (N, 3, 224, 224)
        features = resnet(x) # (N, 512)
        
        # Temporal pooling: average features across the video frames
        pooled_feature = features.mean(dim=0).cpu().numpy() # (512,)
        
        record = {
            "filename": row["filename"],
            "actor_id": row["actor_id"],
            "canonical_label": row["canonical_label"],
            "split": row["split"]
        }
        for i, val in enumerate(pooled_feature):
            record[f"face_{i:03d}"] = val
            
        feature_records.append(record)

features_df = pd.DataFrame(feature_records)
output_path = Path("data/interim/ravdess_face_features.csv")
features_df.to_csv(output_path, index=False)

print(f"\nSuccessfully processed: {len(feature_records)} video clips")
print(f"Skipped/missing: {missing_count} clips")
print(f"Features CSV saved to: {output_path.resolve()}")
