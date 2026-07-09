import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from moviepy import VideoFileClip
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Extract audio track from each full_AV video, then compute
# 39-dimensional MFCC features (12 MFCC + energy + delta + delta-delta)
# This matches the reference paper baseline feature, so we can
# fairly compare against it later.
# ============================================================

SAMPLE_RATE = 22050
N_MFCC = 13  # 12 MFCC + 1 energy-like coefficient

def extract_audio_from_video(video_path: Path, output_wav_path: Path):
    """Extract the audio track from an mp4 and save as wav."""
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(output_wav_path), fps=SAMPLE_RATE, logger=None)
        clip.close()
        return True
    except Exception as e:
        print(f"  FAILED to extract audio from {video_path.name}: {e}")
        return False


def compute_mfcc_features(wav_path: Path):
    """Compute 39-dim MFCC features: 13 base + 13 delta + 13 delta-delta."""
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Average across time frames to get a single fixed-length vector per file
    # (simple baseline approach - later we can keep the full time sequence for BiLSTM)
    mfcc_mean = mfcc.mean(axis=1)
    delta_mean = delta.mean(axis=1)
    delta2_mean = delta2.mean(axis=1)

    full_feature = np.concatenate([mfcc_mean, delta_mean, delta2_mean])  # 39-dim
    return full_feature


# ============================================================
# Process every full_AV speech video
# ============================================================

index_path = Path("data/interim/ravdess_labeled_index.csv")
df = pd.read_csv(index_path)
df_av_speech = df[(df["channel"] == "speech") & (df["modality"] == "full_AV")].copy()

print(f"Found {len(df_av_speech)} full_AV speech video files to process")

audio_output_root = Path("data/interim/audio_wav")
feature_records = []
failed_files = []

for idx, row in df_av_speech.iterrows():
    video_path = Path(row["filepath"])
    video_stem = video_path.stem
    actor_folder = f"Actor_{row['actor_id']:02d}"
    wav_path = audio_output_root / actor_folder / f"{video_stem}.wav"

    success = extract_audio_from_video(video_path, wav_path)
    if not success:
        failed_files.append(video_stem)
        continue

    try:
        mfcc_features = compute_mfcc_features(wav_path)
        record = {
            "filename": row["filename"],
            "actor_id": row["actor_id"],
            "canonical_label": row["canonical_label"],
            "split": row.get("split", None),
        }
        for i, val in enumerate(mfcc_features):
            record[f"mfcc_{i:02d}"] = val
        feature_records.append(record)
        print(f"  {video_stem}: MFCC extracted, shape {mfcc_features.shape}")
    except Exception as e:
        print(f"  FAILED to compute MFCC for {video_stem}: {e}")
        failed_files.append(video_stem)

# ============================================================
# Save results
# ============================================================

features_df = pd.DataFrame(feature_records)
output_path = Path("data/interim/ravdess_mfcc_features.csv")
features_df.to_csv(output_path, index=False)

print(f"\nTotal processed successfully: {len(feature_records)}")
print(f"Total failed: {len(failed_files)}")
if failed_files:
    print("Failed files:", failed_files[:10])

print(f"\nFeature matrix shape: {features_df.shape}")
print(f"Saved to: {output_path.resolve()}")
