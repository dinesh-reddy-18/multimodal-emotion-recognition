import pandas as pd
from pathlib import Path

# ============================================================
# Fixed, documented actor-to-split assignment (subject-independent)
# Decided upfront by actor ID - never changes based on what data
# happens to be downloaded, so it stays reproducible.
# ============================================================

TRAIN_ACTORS = list(range(1, 17))   # actors 1-16  (8 male, 8 female)
VAL_ACTORS   = list(range(17, 21))  # actors 17-20 (2 male, 2 female)
TEST_ACTORS  = list(range(21, 25))  # actors 21-24 (2 male, 2 female)

def get_split(actor_id: int) -> str:
    if actor_id in TRAIN_ACTORS:
        return "train"
    elif actor_id in VAL_ACTORS:
        return "val"
    elif actor_id in TEST_ACTORS:
        return "test"
    else:
        return "unknown"

# ============================================================
# Load the canonical labeled index (from label_mapping.py)
# ============================================================

index_path = Path("data/interim/ravdess_labeled_index.csv")
if not index_path.exists():
    raise FileNotFoundError(f"{index_path} not found - run scripts/label_mapping.py first.")

df = pd.read_csv(index_path)
print(f"Loaded {len(df)} rows from {index_path}")

df["split"] = df["actor_id"].apply(get_split)

# ============================================================
# Report what we actually have vs. the full intended plan
# ============================================================

actors_present = sorted(df["actor_id"].unique())
print(f"\nActors currently present in data: {actors_present}")

missing_train = [a for a in TRAIN_ACTORS if a not in actors_present]
missing_val   = [a for a in VAL_ACTORS if a not in actors_present]
missing_test  = [a for a in TEST_ACTORS if a not in actors_present]

print(f"\nMissing from TRAIN split: {missing_train if missing_train else 'none - complete'}")
print(f"Missing from VAL split:   {missing_val if missing_val else 'none - complete'}")
print(f"Missing from TEST split:  {missing_test if missing_test else 'none - complete'}")

print("\nRow counts per split (based on currently available actors):")
print(df["split"].value_counts())

print("\nCanonical label distribution per split (full_AV + speech only):")
df_av_speech = df[(df["channel"] == "speech") & (df["modality"] == "full_AV")]
print(df_av_speech.groupby("split")["canonical_label"].value_counts())

# ============================================================
# Save split assignment
# ============================================================

output_path = Path("data/interim/ravdess_split_index.csv")
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} rows with split assignments to {output_path.resolve()}")

# Also save the actor-to-split mapping itself as a small reference table
split_plan = pd.DataFrame(
    [{"actor_id": a, "split": "train"} for a in TRAIN_ACTORS] +
    [{"actor_id": a, "split": "val"} for a in VAL_ACTORS] +
    [{"actor_id": a, "split": "test"} for a in TEST_ACTORS]
)
split_plan_path = Path("reports/actor_split_plan.csv")
split_plan.to_csv(split_plan_path, index=False)
print(f"Saved full 24-actor split plan to {split_plan_path.resolve()}")
