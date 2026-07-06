import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

# ============================================================
# Detect and crop faces using MediaPipe's newer Tasks API.
# (The older mp.solutions.face_detection interface was removed
# in recent MediaPipe versions - this replaces it.)
# ============================================================

MODEL_PATH = "models/face/blaze_face_short_range.tflite"
MARGIN_RATIO = 0.2  # 20% padding around detected face box

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceDetectorOptions(base_options=base_options)
detector = mp_vision.FaceDetector.create_from_options(options)


def detect_and_crop_face(image_path: Path, output_path: Path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return False

    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return False

    # Take the first (most confident) detection
    detection = detection_result.detections[0]
    bbox = detection.bounding_box  # pixel coordinates in the new API

    x, y = bbox.origin_x, bbox.origin_y
    box_w, box_h = bbox.width, bbox.height

    margin_x = int(box_w * MARGIN_RATIO)
    margin_y = int(box_h * MARGIN_RATIO)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + box_w + margin_x)
    y2 = min(h, y + box_h + margin_y)

    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), face_crop)
    return True


# ============================================================
# Process every extracted frame
# ============================================================

frames_root = Path("data/interim/face_frames")
faces_root = Path("data/interim/face_crops")

frame_paths = list(frames_root.rglob("*.jpg"))
print(f"Found {len(frame_paths)} extracted frames to process")

detected_count = 0
failed_count = 0
failed_examples = []

for frame_path in frame_paths:
    relative_path = frame_path.relative_to(frames_root)
    output_path = faces_root / relative_path

    success = detect_and_crop_face(frame_path, output_path)
    if success:
        detected_count += 1
    else:
        failed_count += 1
        if len(failed_examples) < 10:
            failed_examples.append(str(relative_path))

print(f"\nFaces detected and cropped: {detected_count}")
print(f"Failed (no face found): {failed_count}")
if failed_examples:
    print("\nExamples of failed frames (no face detected):")
    for ex in failed_examples:
        print(" -", ex)

print(f"\nSaved to: {faces_root.resolve()}")
