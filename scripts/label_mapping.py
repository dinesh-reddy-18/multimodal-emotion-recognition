import pandas as pd
from pathlib import Path

# ============================================================
# Explicit label-mapping table (per Phase 0 rule: never silently
# map incompatible labels - document every decision with a reason)
# ============================================================

label_mapping_rows = [
    {"original_label": "neutral",   "source_dataset": "RAVDESS", "canonical_label": "neutral",   "keep_drop": "keep", "reason": "Direct match across RAVDESS and MELD taxonomies."},
    {"original_label": "calm",      "source_dataset": "RAVDESS", "canonical_label": "DROPPED",    "keep_drop": "drop", "reason": "No equivalent class exists in MELD. Merging into neutral would blur a real conceptual distinction (relaxed positive state vs absence of emotion). Dropped per Phase 0 decision to avoid inventing a false label equivalence."},
    {"original_label": "happy",     "source_dataset": "RAVDESS", "canonical_label": "happy",      "keep_drop": "keep", "reason": "Direct match across RAVDESS and MELD taxonomies."},
    {"original_label": "sad",       "source_dataset": "RAVDESS", "canonical_label": "sad",        "keep_drop": "keep", "reason": "Direct match across RAVDESS and MELD taxonomies."},
    {"original_label": "angry",     "source_dataset": "RAVDESS", "canonical_label": "angry",      "keep_drop": "keep", "reason": "Direct match across RAVDESS and MELD taxonomies."},
    {"original_label": "fearful",   "source_dataset": "RAVDESS", "canonical_label": "fearful",    "keep_drop": "keep", "reason": "Direct match across RAVDESS and MELD taxonomies (MELD calls this 'fear')."},
    {"original_label": "disgust",   "source_dataset": "RAVDESS", "canonical_label": "disgust",    "keep_drop": "keep", "reason": "Direct match across RAVDESS and MELD taxonomies."},
    {"original_label": "surprised", "source_dataset": "RAVDESS", "canonical_label": "surprised",  "keep_drop": "keep", "reason": "Direct match across RAVDESS and MELD taxonomies (MELD calls this 'surprise')."},
]

mapping_df = pd.DataFrame(label_mapping_rows)

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)
mapping_path = reports_dir / "label_mapping_table.csv"
mapping_df.to_csv(mapping_path, index=False)
print(f"Saved label mapping table to {mapping_path.resolve()}")
print()
print(mapping_df.to_string(index=False))

# ============================================================
# Apply the mapping to our actual RAVDESS file index
# ============================================================

index_path = Path("data/interim/ravdess_file_index.csv")
if not index_path.exists():
    raise FileNotFoundError(f"{index_path} not found - run scripts/audit_ravdess.py first.")

df = pd.read_csv(index_path)
print(f"\nLoaded {len(df)} rows from {index_path}")

# Build a lookup dict: original_label -> canonical_label (only for RAVDESS rows)
ravdess_map = {
    row["original_label"]: row["canonical_label"]
    for _, row in mapping_df[mapping_df["source_dataset"] == "RAVDESS"].iterrows()
}

df["canonical_label"] = df["emotion"].map(ravdess_map)

# Keep only rows where canonical_label is not DROPPED
before_count = len(df)
df_final = df[df["canonical_label"] != "DROPPED"].copy()
dropped_count = before_count - len(df_final)

print(f"\nRows before mapping: {before_count}")
print(f"Rows dropped (calm): {dropped_count}")
print(f"Rows remaining: {len(df_final)}")

print("\nFinal canonical label distribution (all files, all modalities):")
print(df_final["canonical_label"].value_counts())

# Also show distribution restricted to full_AV + speech only (our actual training-relevant subset)
df_final_av_speech = df_final[(df_final["channel"] == "speech") & (df_final["modality"] == "full_AV")]
print("\nFinal canonical label distribution (full_AV + speech only - this is what we train on):")
print(df_final_av_speech["canonical_label"].value_counts())

output_path = Path("data/interim/ravdess_labeled_index.csv")
df_final.to_csv(output_path, index=False)
print(f"\nSaved {len(df_final)} rows to {output_path.resolve()}")
