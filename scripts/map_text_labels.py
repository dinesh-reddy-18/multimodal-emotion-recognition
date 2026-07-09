import pandas as pd
from pathlib import Path

# ============================================================
# RAVDESS speech clips use exactly 2 fixed sentences per the
# official dataset documentation - no ASR needed, we map
# directly from the statement code already in our index.
# ============================================================

STATEMENT_MAP = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}

index_path = Path("data/interim/ravdess_labeled_index.csv")
df = pd.read_csv(index_path)

df_av_speech = df[(df["channel"] == "speech") & (df["modality"] == "full_AV")].copy()
print(f"Found {len(df_av_speech)} full_AV speech files")

df_av_speech["text"] = df_av_speech["statement"].map(STATEMENT_MAP)

missing_text = df_av_speech[df_av_speech["text"].isna()]
if len(missing_text) > 0:
    print(f"WARNING: {len(missing_text)} rows have unmapped statement codes:")
    print(missing_text[["filename", "statement"]])
else:
    print("All rows successfully mapped to text.")

print("\nText distribution:")
print(df_av_speech["text"].value_counts())

output_path = Path("data/interim/ravdess_text_index.csv")
df_av_speech[["filename", "actor_id", "canonical_label", "statement", "text"]].to_csv(output_path, index=False)
print(f"\nSaved to {output_path.resolve()}")
