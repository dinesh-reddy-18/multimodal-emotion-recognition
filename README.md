# TriSense-EmotionNet

**Adaptive Multimodal Emotion Recognition Using Facial, Speech, and Textual Cues with Confidence-Aware Gated Fusion**

Final-year B.Tech major project.

## Overview

TriSense-EmotionNet predicts emotion from three modalities — facial expression, speech/voice tone, and text — independently, then fuses them using a confidence-aware gated fusion mechanism that dynamically weights each modality's reliability. The system is designed to remain robust when one or more modalities are missing or noisy.

**This project performs emotion recognition, not clinical diagnosis.** It does not claim to detect depression, anxiety, or any mental-health condition.

## Status

🚧 Under active development — Phase 1 (environment setup) in progress.

## Tech Stack

- **Deep Learning:** PyTorch, torchvision, torchaudio, Hugging Face Transformers
- **Face:** OpenCV, MediaPipe, EfficientNet-B0
- **Audio:** Librosa (MFCC baseline), Wav2Vec2/HuBERT (advanced)
- **Text:** RoBERTa
- **Backend:** FastAPI
- **Frontend:** React + Vite
- **Experiment tracking:** MLflow
- **Hyperparameter tuning:** Optuna

## Project Structure

See `configs/`, `data/`, `notebooks/`, `src/`, `models/`, `backend/`, `frontend/`, `tests/`, `reports/`, and `scripts/` for the full repository layout.

## Setup

See project documentation (Phase 1 setup notes) for full environment setup instructions, including PyTorch CUDA installation matched to your GPU.

## Ethical Notes

- No clinical/medical diagnostic claims are made by this system.
- Facial and vocal data are treated as sensitive; no persistent storage of raw biometric data occurs without explicit consent in any demo/deployment.
- Models are trained primarily on acted/scripted emotional datasets and may not generalize to all real-world, spontaneous, or cross-cultural expressions — this is disclosed as a limitation, not hidden.

## License

Academic project — license TBD.