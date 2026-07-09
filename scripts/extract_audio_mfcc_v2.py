import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from moviepy import VideoFileClip
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Extract audio track from each full_AV video, then compute
# 39-dimensional features: 13 MFCC (mean-pooled) + 13 delta
# (std-pooled) + 13 delta-delta (std-pooled).
#
# FIX vs. v1: delta/delta2 are rate-of-change signals that
# oscillate in sign across an utterance (especially with
# RAVDESS's silence padding at clip boundaries), so mean-pooling
# them collapsed to ~0 (pure cancellation, not signal). Standard
# deviation captures the magnitude of fluctuation instead, which
# is the informative part of a delta feature. This keeps the
# same 39-dim size as the reference paper's feature for a fair
# comparison, while actually being fixing.
# ============================================================

SAMPLE_RATE = 22050
N_MFCC = 13

def extract_audio_from_video(video_path: Path, output_wav_path: Path):
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
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = mfcc.mean(axis=1)      # static level of each coefficient
    delta_std = delta.std(axis=1)      # magnitude of 1st-order change (fixed: was mean)
    delta2_std = delta2.std(axis=1)    # magnitude of 2nd-order change (fixed: was mean)

    full_feature = np.concatenate([mfcc_mean, delta_std, delta2_std])  # 39-dim
    return full_feature


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

    # Reuse already-extracted wav files if present (faster re-run)
    if not wav_path.exists():
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
        }
        for i, val in enumerate(mfcc_features):
            record[f"mfcc_{i:02d}"] = val
        feature_records.append(record)
        print(f"  {video_stem}: MFCC extracted, shape {mfcc_features.shape}")
    except Exception as e:
        print(f"  FAILED to compute MFCC for {video_stem}: {e}")
        failed_files.append(video_stem)

features_df = pd.DataFrame(feature_records)
output_path = Path("data/interim/ravdess_mfcc_features.csv")
features_df.to_csv(output_path, index=False)

print(f"\nTotal processed successfully: {len(feature_records)}")
print(f"Total failed: {len(failed_files)}")
if failed_files:
    print("Failed files:", failed_files[:10])

print(f"\nFeature matrix shape: {features_df.shape}")
print(f"Saved to: {output_path.resolve()}")
