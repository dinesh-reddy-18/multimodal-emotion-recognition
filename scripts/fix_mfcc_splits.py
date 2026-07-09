import pandas as pd
from pathlib import Path

# ============================================================
# The MFCC features file was built from ravdess_labeled_index.csv,
# which does not contain split assignments (those live in a
# separate file, ravdess_split_index.csv). This script merges
# them together correctly using filename as the join key.
# ============================================================

mfcc_path = Path("data/interim/ravdess_mfcc_features.csv")
split_path = Path("data/interim/ravdess_split_index.csv")

mfcc_df = pd.read_csv(mfcc_path)
split_df = pd.read_csv(split_path)

print(f"MFCC features: {len(mfcc_df)} rows")
print(f"Split index: {len(split_df)} rows")

# Drop the old (empty) split column before merging
mfcc_df = mfcc_df.drop(columns=["split"], errors="ignore")

# Build a filename -> split lookup (filename is unique per file, actor ID is embedded in it)
split_lookup = split_df[["filename", "split"]].drop_duplicates(subset="filename")

merged_df = mfcc_df.merge(split_lookup, on="filename", how="left")

print(f"\nAfter merge: {len(merged_df)} rows")
print("\nSplit distribution:")
print(merged_df["split"].value_counts(dropna=False))

# Sanity check: make sure nothing came back as NaN
missing = merged_df[merged_df["split"].isna()]
if len(missing) > 0:
    print(f"\nWARNING: {len(missing)} rows still have no split assigned!")
    print(missing[["filename", "actor_id"]].head(10))
else:
    print("\nAll rows successfully matched to a split.")

merged_df.to_csv(mfcc_path, index=False)
print(f"\nSaved corrected file to {mfcc_path.resolve()}")
