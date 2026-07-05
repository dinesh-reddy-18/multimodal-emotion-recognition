import cv2
from pathlib import Path
import pandas as pd

# ============================================================
# Extract frames from RAVDESS video files
# We extract a fixed number of evenly-spaced frames per video
# rather than every single frame, to keep data size manageable
# and avoid near-duplicate frames from adjacent video moments.
# ============================================================

FRAMES_PER_VIDEO = 10  # evenly spaced samples across each clip

def extract_frames(video_path: Path, output_dir: Path, n_frames: int = FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames == 0:
        print(f"  WARNING: could not read frames from {video_path.name}")
        cap.release()
        return 0

    # Evenly spaced frame indices across the video
    frame_indices = [int(i * total_frames / n_frames) for i in range(n_frames)]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for idx, frame_no in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, frame = cap.read()
        if success:
            out_path = output_dir / f"frame_{idx:02d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

    cap.release()
    return saved_count


# ============================================================
# Run extraction on our labeled index (full_AV + speech only)
# ============================================================

index_path = Path("data/interim/ravdess_labeled_index.csv")
df = pd.read_csv(index_path)
df_av_speech = df[(df["channel"] == "speech") & (df["modality"] == "full_AV")]

print(f"Found {len(df_av_speech)} full_AV speech video files to process")

frames_output_root = Path("data/interim/face_frames")
total_saved = 0

for _, row in df_av_speech.iterrows():
    video_path = Path(row["filepath"])
    # Organize output as: face_frames/Actor_01/01-01-01-01-01-01-01/frame_00.jpg
    video_stem = video_path.stem
    actor_folder = f"Actor_{row['actor_id']:02d}"
    output_dir = frames_output_root / actor_folder / video_stem

    saved = extract_frames(video_path, output_dir)
    total_saved += saved
    print(f"  {video_path.name}: saved {saved} frames")

print(f"\nTotal frames extracted: {total_saved}")
print(f"Saved to: {frames_output_root.resolve()}")
