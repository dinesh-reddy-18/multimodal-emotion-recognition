import os
from pathlib import Path
import pandas as pd

RAVDESS_ROOT = Path("data/raw/ravdess")
print("Looking for data at:", RAVDESS_ROOT.resolve())
print("Exists:", RAVDESS_ROOT.exists())

actor_folders = sorted([f for f in RAVDESS_ROOT.iterdir() if f.is_dir() and f.name.startswith("Actor_")])
print(f"Found {len(actor_folders)} actor folder(s):")
for f in actor_folders:
    print(" -", f.name)

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
MODALITY_MAP = {"01": "full_AV", "02": "video_only", "03": "audio_only"}
CHANNEL_MAP = {"01": "speech", "02": "song"}
INTENSITY_MAP = {"01": "normal", "02": "strong"}

def parse_ravdess_filename(filename):
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    return {
        "filename": filename,
        "modality": MODALITY_MAP.get(modality, modality),
        "channel": CHANNEL_MAP.get(channel, channel),
        "emotion": EMOTION_MAP.get(emotion, emotion),
        "intensity": INTENSITY_MAP.get(intensity, intensity),
        "statement": statement,
        "repetition": repetition,
        "actor_id": int(actor),
        "gender": "male" if int(actor) % 2 == 1 else "female",
    }

all_records = []
for actor_folder in actor_folders:
    for file in actor_folder.iterdir():
        if file.suffix.lower() in [".mp4", ".wav"]:
            parsed = parse_ravdess_filename(file.name)
            if parsed:
                parsed["filepath"] = str(file)
                parsed["file_size_kb"] = round(file.stat().st_size / 1024, 1)
                all_records.append(parsed)

df = pd.DataFrame(all_records)
print(f"\nTotal files parsed: {len(df)}")
print(df.head(10))

print("\nFiles per actor:")
print(df.groupby("actor_id").size())
print("\nFiles per modality:")
print(df["modality"].value_counts())
print("\nFiles per channel:")
print(df["channel"].value_counts())

df_speech_av = df[(df["channel"] == "speech") & (df["modality"] == "full_AV")]
print(f"\nFull audio-video speech files: {len(df_speech_av)}")
print("\nEmotion distribution (speech, full AV only):")
print(df_speech_av["emotion"].value_counts())

output_path = Path("data/interim/ravdess_file_index.csv")
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} rows to {output_path.resolve()}")
