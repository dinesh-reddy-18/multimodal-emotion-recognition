# Multimodal Emotion Recognition (Final Year Project)

A complete end-to-end pipeline for **recognizing human emotions** from audio, facial expression, and text using Machine Learning and Deep Learning.

---

## Project Summary

| Item | Detail |
|---|---|
| **Dataset** | RAVDESS (14 actor subset: actors 1-10, 17-18, 21-22) |
| **Modalities** | Audio (MFCC), Facial Expression (ResNet18), Text (RoBERTa) |
| **Emotions** | angry, disgust, fearful, happy, neutral, sad, surprised |
| **Fusion** | Decision-level (Late) + Feature-level (Early) |
| **Backend** | FastAPI |
| **Frontend** | Vanilla HTML/CSS/JS with Chart.js |

---

## Folder Structure

```
multimodal-emotion-recognition/
├── data/
│   ├── raw/ravdess/          # Raw RAVDESS mp4 video clips (Actor_XX folders)
│   └── interim/              # Preprocessed features and index CSVs
├── scripts/                  # All preprocessing and training scripts
├── models/
│   ├── audio_baseline/       # MLP Audio checkpoint (.pt)
│   ├── text_baseline/        # MLP Text/RoBERTa checkpoint (.pt)
│   ├── face/                 # MLP Face checkpoint + BlazeFace tflite
│   └── fusion/               # Late & Early Fusion checkpoints (.pt)
├── frontend/                 # Single-page web app  
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── backend/app/main.py       # FastAPI server
├── tests/test_pipeline.py    # Integration tests
├── reports/                  # Label mappings, actor/split plans
├── notebooks/                # EDA notebook
└── requirements.txt
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+ with `venv`
- CUDA GPU recommended (RAVDESS face features use ResNet18 GPU acceleration)
- FFmpeg installed for audio extraction via moviepy

### 2. Install Dependencies

From the project root:

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 3. RAVDESS Dataset

The dataset should be placed under `data/raw/ravdess/` in actor subfolders:
```
data/raw/ravdess/
    Actor_01/   (60 mp4 clips each)
    Actor_02/
    ...
```

Actors 1–16 = **Train**, Actors 17–20 = **Validation**, Actors 21–24 = **Test** (subject-independent).

---

## Preprocessing Pipeline

Run these scripts in sequence from the project root using `venv\Scripts\python.exe`:

```bash
# 1. Parse filenames and build master index
python scripts/audit_ravdess.py

# 2. Map emotion labels (drop 'calm', keep 7 emotions)
python scripts/label_mapping.py

# 3. Assign speaker-independent train/val/test splits
python scripts/create_splits.py

# 4. Extract MFCC audio features (39-dim: mean + delta-std + delta2-std)
python scripts/extract_audio_mfcc_v2.py
python scripts/fix_mfcc_splits.py

# 5. Extract text sentence index and RoBERTa embeddings
python scripts/map_text_labels.py
python scripts/extract_text_features.py

# 6. Extract video frames for face detection
python scripts/extract_frames.py
python scripts/detect_faces.py          # MediaPipe BlazeFace cropping

# 7. Extract ResNet18 face feature embeddings (512-dim per video)
python scripts/extract_face_features.py
```

---

## Training Pipeline

```bash
# Train unimodal Audio baseline (MLP on 39-dim MFCC)
python scripts/train_audio_baseline_mlp.py

# Train unimodal Text baseline (MLP on 768-dim RoBERTa CLS)
python scripts/train_text_baseline.py

# Train unimodal Face baseline (MLP on 512-dim ResNet18 features)
python scripts/train_face_baseline.py

# Train both multimodal fusion models (Early + Late Fusion)
python scripts/train_fusion.py
```

All models are saved in the corresponding `models/` subfolder.

---

## Evaluation / Test Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Audio MLP (MFCC) | Saved to checkpoint | Macro F1 in report |
| Text MLP (RoBERTa) | ~16% (RAVDESS limitation*) | ~0.14 |
| Face MLP (ResNet18) | Saved to checkpoint | Macro F1 in report |
| Late Fusion | Best of above | Saved to config |
| Early Fusion | Best joint model | Saved to checkpoint |

> **Note on Text Modality**: RAVDESS contains only 2 unique sentences, therefore RoBERTa CLS embeddings carry no per-sample emotional signal. Near-chance text performance is **expected and documented** — not a bug.

---

## Running Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
```

All 14 integration tests should pass, verifying:
- Checkpoint existence
- Feature CSV validity
- Split assignment correctness (no data leakage)
- Unimodal model output shapes
- MFCC preprocessing shape
- Late fusion weight validity

---

## Running the Application

### 1. Start the FastAPI Server

```bash
venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Open the Frontend

Navigate to `http://localhost:8000` in a browser.

The app will:
- Load all model checkpoints on startup
- Show a green "Models Online" indicator when ready
- Accept MP4 video, WAV audio, or typed text
- Return per-modality predictions + fused result with probability bar chart

---

## Architecture

### Audio Branch
- **Features**: 39-dim MFCC (13 base + 13 delta-std + 13 delta2-std), mean-pooled per utterance
- **Model**: 2-layer MLP with BatchNorm + Dropout + class-weighted CrossEntropy loss
- **Normalization**: Train-set mean/std, stored in checkpoint

### Text Branch
- **Features**: 768-dim RoBERTa-base `[CLS]` pooled embedding  
- **Model**: 1-layer MLP with BatchNorm + Dropout
- **Note**: Near-chance performance is expected (RAVDESS 2-sentence limitation)

### Face/Image Branch
- **Features**: 512-dim ResNet18 (ImageNet pretrained) spatial features, temporally average-pooled across 10 evenly-spaced video frames per clip
- **Model**: 2-layer MLP with BatchNorm + Dropout

### Multimodal Fusion
- **Late Fusion**: Weighted probability sum P = w_a·P_audio + w_t·P_text + w_f·P_face. Optimal weights grid-searched on validation set. Graceful fallback if a modality is missing.
- **Early Fusion**: 1319-dim concatenation (39 + 768 + 512) fed to a joint 2-layer MLP.

---

## Label Mapping

| RAVDESS Label | Canonical Label | Decision |
|---|---|---|
| neutral | neutral | keep |
| calm | — | **dropped** — no equivalent in MELD taxonomy |
| happy | happy | keep |
| sad | sad | keep |
| angry | angry | keep |
| fearful | fearful | keep |
| disgust | disgust | keep |
| surprised | surprised | keep |

Final label set: **7 classes** (after dropping `calm`)

---

## Limitations

1. **Partial RAVDESS dataset**: Only 14 of 24 actors are present. Missing actors 11-16, 19-20, 23-24 mean training data is reduced.
2. **Text modality ceiling**: Only 2 RAVDESS sentences exist. Text branch serves as a legitimate multimodal component during demo but contributes minimal emotion signal.
3. **No real-time webcam/microphone**: Demo requires pre-recorded file uploads.
4. **Static feature pooling**: MFCC and ResNet18 features are temporally averaged, losing prosody and facial dynamics.

## Future Work

1. Use full 24-actor RAVDESS dataset
2. Replace MFCC with end-to-end CNN/Wav2Vec audio model
3. Use CNN-LSTM for video temporal modeling
4. Add attention-based fusion
5. Implement live webcam and microphone streaming
6. Fine-tune RoBERTa on emotional speech transcripts (MELD, IEMOCAP)

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `No module named 'torch'` | Use `venv\Scripts\python.exe`, not system Python |
| `libomp.dll missing` | Install VC++ redistributable or use `conda` environment |
| `File not found: ravdess_mfcc_features.csv` | Run preprocessing scripts in order |
| `Faces not detected` | Confirm `models/face/blaze_face_short_range.tflite` exists |
| Server not responding | Check port 8000 is free: `netstat -an \| findstr 8000` |