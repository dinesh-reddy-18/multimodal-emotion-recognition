import os
import shutil
import cv2
import librosa
import numpy as np
import pandas as pd
import warnings
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

app = FastAPI(title="Multimodal Emotion Recognition API")

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define labels
labels = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

# Helper functions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Load stats
audio_stats = np.load("onnx_models/audio_stats.npz", allow_pickle=True)
audio_mean = audio_stats["mean"].item()
audio_std = audio_stats["std"].item()

face_stats = np.load("onnx_models/face_stats.npz", allow_pickle=True)
face_mean = face_stats["mean"].item()
face_std = face_stats["std"].item()

text_stats = np.load("onnx_models/text_stats.npz", allow_pickle=True)
text_mean = text_stats["mean"].item()
text_std = text_stats["std"].item()

late_weights_data = np.load("onnx_models/late_fusion_weights.npz", allow_pickle=True)
late_fusion_weights = late_weights_data["weights"]

# Configure ONNX runtime options to prevent thread-deadlocking in resource-constrained environments (like Render)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Load ONNX sessions on CPU
print("Loading ONNX sessions for FastAPI backend...")
audio_session = ort.InferenceSession("onnx_models/audio_model.onnx", sess_options=sess_options, providers=["CPUExecutionProvider"])
face_session = ort.InferenceSession("onnx_models/face_pipeline.onnx", sess_options=sess_options, providers=["CPUExecutionProvider"])
early_fusion_session = ort.InferenceSession("onnx_models/early_fusion.onnx", sess_options=sess_options, providers=["CPUExecutionProvider"])
text_session = ort.InferenceSession("onnx_models/text_pipeline.onnx", sess_options=sess_options, providers=["CPUExecutionProvider"])
print("ONNX sessions loaded successfully.")

# Load Tokenizer locally
print("Loading tokenizer for FastAPI backend...")
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
print("Tokenizer loaded successfully.")

# MediaPipe BlazeFace Detector
detector = None
try:
    model_path = "models/face/blaze_face_short_range.tflite"
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(base_options=base_options)
    detector = mp_vision.FaceDetector.create_from_options(options)
    print("MediaPipe Face Detector initialized.")
except Exception as e:
    print(f"Error loading MediaPipe detector: {e}")

# ============================================================
# PREPROCESSING HELPERS
# ============================================================

def convert_to_wav_ffmpeg(input_path: str, output_path: str) -> bool:
    """Convert any audio or video file to standard 22050Hz mono 16-bit WAV using lightweight ffmpeg CLI."""
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ar", "22050",
        "-ac", "1",
        "-codec:a", "pcm_s16le",
        output_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return res.returncode == 0
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        return False

def ensure_wav(input_path: str) -> str:
    """Ensure the audio file is in standard WAV format using lightweight ffmpeg CLI."""
    if input_path.lower().endswith(".wav"):
        return input_path
    
    temp_wav = input_path + "_converted.wav"
    success = convert_to_wav_ffmpeg(input_path, temp_wav)
    if success:
        return temp_wav
    return input_path

def preprocess_audio(wav_path: str) -> np.ndarray:
    """Compute static 39-dim MFCC values."""
    y, sr = librosa.load(wav_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    mfcc_mean = mfcc.mean(axis=1)
    delta_std = delta.std(axis=1)
    delta2_std = delta2.std(axis=1)
    
    full_feature = np.concatenate([mfcc_mean, delta_std, delta2_std])
    return full_feature

def detect_and_crop_face(image_bgr):
    """Detect crop of the primary face using MediaPipe detector."""
    if detector is None:
        return None
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    if not detection_result.detections:
        return None
        
    bbox = detection_result.detections[0].bounding_box
    x, y = bbox.origin_x, bbox.origin_y
    box_w, box_h = bbox.width, bbox.height
    
    margin_x = int(box_w * 0.2)
    margin_y = int(box_h * 0.2)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + box_w + margin_x)
    y2 = min(h, y + box_h + margin_y)
    
    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    return face_crop

def preprocess_face_numpy(face_bgr):
    """Resize, transpose, and normalize crop for ResNet18."""
    resized = cv2.resize(face_bgr, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std
    chw = normalized.transpose(2, 0, 1)
    return chw

def preprocess_video(video_path: str):
    """Extract frames, crop faces, and extract ResNet18 embeddings using ONNX."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read frames from video")
        
    n_frames = 15
    frame_indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    
    frame_tensors = []
    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        success, img = cap.read()
        if success:
            cropped = detect_and_crop_face(img)
            if cropped is None:
                cropped = img
            chw = preprocess_face_numpy(cropped)
            frame_tensors.append(chw)
                
    cap.release()
    
    if len(frame_tensors) == 0:
        raise ValueError("No faces detected in video file.")
        
    x = np.stack(frame_tensors).astype(np.float32)
    
    # Run the face pipeline ONNX session
    features_out, logits_out = face_session.run(None, {"input_image": x})
    pooled_feat = features_out.mean(axis=0)
    
    return features_out, logits_out, pooled_feat

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/models/status")
def get_models_status():
    """Return status of all loaded components."""
    return {
        "status": "online",
        "audio_loaded": audio_session is not None,
        "text_loaded": text_session is not None,
        "face_loaded": face_session is not None,
        "early_fusion_loaded": early_fusion_session is not None,
        "late_fusion_loaded": late_fusion_weights is not None,
        "labels": labels
    }

@app.post("/api/predict")
async def predict_emotion(
    text: str = Form(None),
    audio: UploadFile = File(None),
    video: UploadFile = File(None)
):
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    active_modalities = {}
    probs_dict = {}
    feats_dict = {}
    temp_files = []
    
    # 1. PROCESS TEXT MODALITY
    if text and text.strip():
        try:
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
            
            logits, raw_text_feat = text_session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            probs = softmax(logits[0])
            
            # Map predictions from GoEmotions output index mapping
            label_map = {
                'anger': 'angry',
                'disgust': 'disgust',
                'fear': 'fearful',
                'joy': 'happy',
                'neutral': 'neutral',
                'sadness': 'sad',
                'surprise': 'surprised'
            }
            mapped_probs = np.zeros(7)
            hf_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            for i, score in enumerate(probs):
                mapped_probs[label2idx[label_map[hf_labels[i]]]] = score
                
            probs_dict["text"] = mapped_probs.tolist()
            active_modalities["text"] = {
                "label": labels[int(mapped_probs.argmax())],
                "confidence": float(mapped_probs.max()),
                "probs": mapped_probs.tolist()
            }
            
            # Normalize embedding for early fusion
            mean_vals = np.array([text_mean[f"roberta_{i}"] for i in range(768)], dtype=np.float32)
            std_vals = np.array([text_std[f"roberta_{i}"] for i in range(768)], dtype=np.float32)
            norm_feat = (raw_text_feat[0] - mean_vals) / std_vals
            feats_dict["text"] = norm_feat
        except Exception as e:
            print(f"Error serving text analysis: {e}")
            
    # 2. PROCESS AUDIO MODALITY
    temp_audio_file = None
    if audio:
        try:
            file_ext = Path(audio.filename).suffix
            temp_audio_file = temp_dir / f"uploaded_audio{file_ext}"
            with open(temp_audio_file, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            temp_files.append(temp_audio_file)
            
            # Convert audio to wav if necessary using moviepy converter
            wav_path = ensure_wav(str(temp_audio_file))
            if wav_path != str(temp_audio_file):
                temp_files.append(Path(wav_path))
                
            raw_audio_feat = preprocess_audio(wav_path)
            mean_vals = np.array([audio_mean[f"mfcc_{i:02d}"] for i in range(39)], dtype=np.float32)
            std_vals = np.array([audio_std[f"mfcc_{i:02d}"] for i in range(39)], dtype=np.float32)
            norm_feat = (raw_audio_feat - mean_vals) / std_vals
            
            x_aud = norm_feat.astype(np.float32).reshape(1, 39)
            logits = audio_session.run(None, {"input": x_aud})[0]
            probs = softmax(logits)[0]
            
            probs_dict["audio"] = probs.tolist()
            feats_dict["audio"] = norm_feat
            active_modalities["audio"] = {
                "label": labels[int(probs.argmax())],
                "confidence": float(probs.max()),
                "probs": probs.tolist()
            }
        except Exception as e:
            print(f"Error serving audio analysis: {e}")
            
    # 3. PROCESS VIDEO MODALITY
    temp_video_file = None
    if video:
        try:
            file_ext = Path(video.filename).suffix
            temp_video_file = temp_dir / f"uploaded_video{file_ext}"
            with open(temp_video_file, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            temp_files.append(temp_video_file)
            
            # A. Process Video Frames (Face crops classification)
            try:
                features_out, logits_out, pooled_face_feat = preprocess_video(str(temp_video_file))
                
                # Normalize features for early fusion representation
                mean_vals = np.array([face_mean[f"face_{i:03d}"] for i in range(512)], dtype=np.float32)
                std_vals = np.array([face_std[f"face_{i:03d}"] for i in range(512)], dtype=np.float32)
                norm_face_feat = (pooled_face_feat - mean_vals) / std_vals
                
                # Aggregate probabilities temporally
                all_probs = [softmax(l) for l in logits_out]
                face_probs = np.mean(all_probs, axis=0)
                
                probs_dict["face"] = face_probs.tolist()
                feats_dict["face"] = norm_face_feat
                active_modalities["face_video"] = {
                    "label": labels[int(face_probs.argmax())],
                    "confidence": float(face_probs.max()),
                    "probs": face_probs.tolist()
                }
            except Exception as e:
                print(f"Error processing video frames: {e}")
                
            # B. Extract Audio track from video if not already processed a separate audio
            if "audio" not in probs_dict:
                try:
                    extracted_wav = str(temp_dir / "temp_video_audio.wav")
                    success = convert_to_wav_ffmpeg(str(temp_video_file), extracted_wav)
                    if success and os.path.exists(extracted_wav):
                        temp_files.append(Path(extracted_wav))
                        
                        raw_audio_feat = preprocess_audio(extracted_wav)
                        mean_vals = np.array([audio_mean[f"mfcc_{i:02d}"] for i in range(39)], dtype=np.float32)
                        std_vals = np.array([audio_std[f"mfcc_{i:02d}"] for i in range(39)], dtype=np.float32)
                        norm_feat = (raw_audio_feat - mean_vals) / std_vals
                        
                        x_aud = norm_feat.astype(np.float32).reshape(1, 39)
                        logits = audio_session.run(None, {"input": x_aud})[0]
                        probs = softmax(logits)[0]
                        
                        probs_dict["audio"] = probs.tolist()
                        feats_dict["audio"] = norm_feat
                        active_modalities["audio"] = {
                            "label": labels[int(probs.argmax())],
                            "confidence": float(probs.max()),
                            "probs": probs.tolist()
                        }
                except Exception as e:
                    print(f"Error extracting audio from video: {e}")
        except Exception as e:
            print(f"Error serving video input: {e}")
            
    # Clean up temp files to prevent disk leak
    for path in temp_files:
        if path.exists():
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error removing temporary file {path}: {e}")
                
    if len(active_modalities) == 0:
        raise HTTPException(status_code=400, detail="No active inputs found or processed successfully. Provide text, audio file, or webcam recording.")
        
    # ============================================================
    # MULTIMODAL FUSION
    # ============================================================
    fusion_results = {}
    
    # 1. Late Fusion Calculation
    w_a, w_t, w_f = late_fusion_weights
    curr_weights = []
    curr_probs = []
    
    if "audio" in probs_dict:
        curr_weights.append(w_a)
        curr_probs.append(np.array(probs_dict["audio"]))
    if "text" in probs_dict:
        curr_weights.append(w_t)
        curr_probs.append(np.array(probs_dict["text"]))
    if "face" in probs_dict:
        curr_weights.append(w_f)
        curr_probs.append(np.array(probs_dict["face"]))
        
    curr_weights = np.array(curr_weights)
    weight_sum = curr_weights.sum()
    if weight_sum > 0:
        curr_weights = curr_weights / weight_sum  # normalize
        fused_probs = np.zeros(7)
        for w, p in zip(curr_weights, curr_probs):
            fused_probs += w * p
            
        fusion_results["late_fusion"] = {
            "label": labels[int(fused_probs.argmax())],
            "confidence": float(fused_probs.max()),
            "probs": fused_probs.tolist()
        }
        
    # 2. Early Fusion Calculation
    try:
        feat_a = feats_dict.get("audio", np.zeros(39, dtype=np.float32))
        feat_t = feats_dict.get("text", np.zeros(768, dtype=np.float32))
        feat_f = feats_dict.get("face", np.zeros(512, dtype=np.float32))
        
        joint_feat = np.concatenate([feat_a, feat_t, feat_f]).astype(np.float32).reshape(1, 1319)
        logits = early_fusion_session.run(None, {"input": joint_feat})[0]
        probs = softmax(logits)[0]
        
        fusion_results["early_fusion"] = {
            "label": labels[int(probs.argmax())],
            "confidence": float(probs.max()),
            "probs": probs.tolist()
        }
    except Exception as e:
        print(f"Error serving Early Fusion: {e}")
        
    # Final recommendation
    primary_fusion = "late_fusion" if "late_fusion" in fusion_results else "early_fusion"
    if primary_fusion in fusion_results:
        final_prediction = {
            "label": fusion_results[primary_fusion]["label"],
            "confidence": fusion_results[primary_fusion]["confidence"]
        }
    else:
        first_mod = list(active_modalities.keys())[0]
        final_prediction = {
            "label": active_modalities[first_mod]["label"],
            "confidence": active_modalities[first_mod]["confidence"]
        }
        
    return {
        "final_prediction": final_prediction,
        "modalities": active_modalities,
        "fusion": fusion_results
    }

# Mount static files folder to serve the frontend web-app
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
