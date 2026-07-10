import os
import shutil
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForSequenceClassification
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Multimodal Emotion Recognition API")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"FastAPI Server loading models on device: {DEVICE}")

# Define the model classes exactly as they were trained
class MLPAudioClassifier(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=128, num_classes=7, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

class MLPTextClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=7, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x): return self.net(x)

class MLPFaceClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

class JointMLPFusionClassifier(nn.Module):
    def __init__(self, input_dim=1319, hidden_dim=256, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    def forward(self, x): return self.net(x)

# Global variables to hold models and statistics
audio_model = None
audio_mean = None
audio_std = None

text_model = None
text_mean = None
text_std = None
roberta_tokenizer = None
roberta_model = None

face_model = None
face_mean = None
face_std = None
resnet18 = None

early_fusion_model = None
late_fusion_weights = None

text_pretrained_tokenizer = None
text_pretrained_model = None

labels = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

# MediaPipe face detector setup
detector = None

@app.on_event("startup")
def load_all_components():
    global audio_model, audio_mean, audio_std
    global text_model, text_mean, text_std, roberta_tokenizer, roberta_model
    global face_model, face_mean, face_std, resnet18, detector
    global early_fusion_model, late_fusion_weights
    
    # 1. MediaPipe BlazeFace Detector
    try:
        model_path = "models/face/blaze_face_short_range.tflite"
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(base_options=base_options)
        detector = mp_vision.FaceDetector.create_from_options(options)
        print("MediaPipe Face Detector initialized.")
    except Exception as e:
        print(f"Error loading MediaPipe detector: {e}")

    # 2. Audio Baseline
    try:
        audio_ckpt = torch.load("models/audio_baseline/mlp_mfcc_baseline.pt", map_location=DEVICE, weights_only=False)
        audio_model = MLPAudioClassifier(input_dim=39, num_classes=7).to(DEVICE)
        audio_model.load_state_dict(audio_ckpt["model_state_dict"])
        audio_model.eval()
        audio_mean = pd.Series(audio_ckpt["mean"])
        audio_std = pd.Series(audio_ckpt["std"])
        print("Audio MLP model loaded successfully.")
    except Exception as e:
        print(f"Error loading Audio model: {e}")

    # 3. Text Baseline
    try:
        global text_pretrained_tokenizer, text_pretrained_model
        text_ckpt = torch.load("models/text_baseline/mlp_roberta_baseline.pt", map_location=DEVICE, weights_only=False)
        text_model = MLPTextClassifier(input_dim=768, num_classes=7).to(DEVICE)
        text_model.load_state_dict(text_ckpt["model_state_dict"])
        text_model.eval()
        text_mean = pd.Series(text_ckpt["mean"])
        text_std = pd.Series(text_ckpt["std"])
        
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta_model = RobertaModel.from_pretrained("roberta-base").to(DEVICE)
        roberta_model.eval()
        
        # Load high-accuracy pre-trained GoEmotions model
        text_pretrained_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        text_pretrained_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base").to(DEVICE)
        text_pretrained_model.eval()
        print("Text MLP, pre-trained RoBERTa, and GoEmotions models loaded successfully.")
    except Exception as e:
        print(f"Error loading Text model: {e}")

    # 4. Face/Image Baseline
    try:
        face_ckpt = torch.load("models/face/resnet18_face_baseline.pt", map_location=DEVICE, weights_only=False)
        face_model = MLPFaceClassifier(input_dim=512, num_classes=7).to(DEVICE)
        face_model.load_state_dict(face_ckpt["model_state_dict"])
        face_model.eval()
        face_mean = pd.Series(face_ckpt["mean"])
        face_std = pd.Series(face_ckpt["std"])
        
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Identity()
        resnet18 = resnet18.to(DEVICE)
        resnet18.eval()
        print("Face MLP and pre-trained ResNet18 loaded successfully.")
    except Exception as e:
        print(f"Error loading Face model: {e}")

    # 5. Fusion Models
    try:
        early_ckpt = torch.load("models/fusion/early_fusion_model.pt", map_location=DEVICE, weights_only=False)
        early_fusion_model = JointMLPFusionClassifier(input_dim=1319, num_classes=7).to(DEVICE)
        early_fusion_model.load_state_dict(early_ckpt["model_state_dict"])
        early_fusion_model.eval()
        print("Early Fusion joint MLP model loaded successfully.")
    except Exception as e:
        print(f"Error loading Early Fusion model: {e}")
        
    try:
        late_ckpt = torch.load("models/fusion/late_fusion_config.pt", map_location=DEVICE, weights_only=False)
        late_fusion_weights = late_ckpt["best_weights"]
        print(f"Late Fusion loaded. Optimal weights: {late_fusion_weights}")
    except Exception as e:
        print(f"Error loading Late Fusion config: {e}")
        # Default fallback equal weights
        late_fusion_weights = (0.33, 0.33, 0.34)

# ============================================================
# PREPROCESSING HELPER FUNCTIONS
# ============================================================

def preprocess_text(text: str) -> np.ndarray:
    """Tokenize and extract RoBERTa [CLS] pooled embedding."""
    inputs = roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        # Class CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return cls_embedding

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

def preprocess_video(video_path: str):
    """Extract frames, crop faces, and extract ResNet18 embeddings for all valid frames."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read frames from video")
        
    n_frames = 15
    frame_indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    
    face_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_tensors = []
    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        success, img = cap.read()
        if success:
            cropped = detect_and_crop_face(img)
            if cropped is not None:
                # Convert BGR to RGB
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cropped_rgb)
                frame_tensors.append(face_transform(pil_img))
                
    cap.release()
    
    if len(frame_tensors) == 0:
        raise ValueError("No faces detected in video file.")
        
    x = torch.stack(frame_tensors).to(DEVICE)
    with torch.no_grad():
        features = resnet18(x).cpu().numpy()
        
    pooled = features.mean(axis=0)
    return features, pooled

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/models/status")
def get_models_status():
    """Return loading flag status of all unimodal baselines."""
    return {
        "audio_loaded": audio_model is not None,
        "text_loaded": text_model is not None,
        "face_loaded": face_model is not None,
        "early_fusion_loaded": early_fusion_model is not None,
        "late_fusion_loaded": late_fusion_weights is not None,
        "optimal_weights": late_fusion_weights,
        "labels": labels
    }

@app.post("/api/predict")
async def predict_emotion(
    text: str = Form(None),
    audio: UploadFile = File(None),
    video: UploadFile = File(None)
):
    # Setup temporary directory for uploaded files
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    # Track which modalities are processed and active
    active_modalities = {}
    probs_dict = {}
    feats_dict = {}
    
    # 1. PROCESS TEXT MODALITY
    if text and text.strip():
        if text_model is None or text_pretrained_model is None:
            raise HTTPException(status_code=500, detail="Text model is not initialized.")
        try:
            # A. Get RoBERTa CLS embeddings for early fusion feature representation
            raw_text_feat = preprocess_text(text)
            cols = [f"roberta_{i}" for i in range(768)]
            norm_feat = ((raw_text_feat - text_mean[cols]) / text_std[cols]).values.astype(np.float32)
            
            # B. Get high-accuracy predictions from pre-trained GoEmotions model
            pretrained_inputs = text_pretrained_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                pretrained_logits = text_pretrained_model(**pretrained_inputs).logits
                probs = torch.softmax(pretrained_logits, dim=1).cpu().numpy()[0]
                
            probs_dict["text"] = probs.tolist()
            feats_dict["text"] = norm_feat
            active_modalities["text"] = {
                "label": labels[int(probs.argmax())],
                "confidence": float(probs.max()),
                "probs": probs.tolist()
            }
        except Exception as e:
            print(f"Error serving text processing: {e}")
            
    # 2. PROCESS AUDIO MODALITY
    temp_audio_file = None
    if audio:
        if audio_model is None:
            raise HTTPException(status_code=500, detail="Audio model is not initialized.")
        try:
            file_ext = Path(audio.filename).suffix
            temp_audio_file = temp_dir / f"uploaded_audio{file_ext}"
            with open(temp_audio_file, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
                
            raw_audio_feat = preprocess_audio(str(temp_audio_file))
            
            # Normalize
            cols = [f"mfcc_{i:02d}" for i in range(39)]
            norm_feat = ((raw_audio_feat - audio_mean[cols]) / audio_std[cols]).values.astype(np.float32)
            
            # Predict
            x = torch.tensor(norm_feat).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = audio_model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            probs_dict["audio"] = probs.tolist()
            feats_dict["audio"] = norm_feat
            active_modalities["audio"] = {
                "label": labels[int(probs.argmax())],
                "confidence": float(probs.max()),
                "probs": probs.tolist()
            }
        except Exception as e:
            print(f"Error serving audio prediction: {e}")
            
    # 3. PROCESS VIDEO MODALITY (Contains Face Image Sequences & Video Audio)
    temp_video_file = None
    if video:
        try:
            file_ext = Path(video.filename).suffix
            temp_video_file = temp_dir / f"uploaded_video{file_ext}"
            with open(temp_video_file, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
                
            # A. Process Video Frames (Face crops baseline prediction)
            if face_model is not None:
                try:
                    raw_face_features, pooled_face_feat = preprocess_video(str(temp_video_file))
                    cols = [f"face_{i:03d}" for i in range(512)]
                    
                    # Normalize pooled features for early fusion
                    norm_face_feat = ((pooled_face_feat - face_mean[cols]) / face_std[cols]).values.astype(np.float32)
                    
                    # Compute prediction probabilities per valid frame and average them
                    all_probs = []
                    for frame_feat in raw_face_features:
                        norm_frame_feat = ((frame_feat - face_mean[cols]) / face_std[cols]).values.astype(np.float32)
                        x_face = torch.tensor(norm_frame_feat).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            logits = face_model(x_face)
                            frame_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        all_probs.append(frame_probs)
                        
                    # Aggregate predictions temporally: mean probability aggregation
                    probs = np.mean(all_probs, axis=0)
                    
                    probs_dict["face"] = probs.tolist()
                    feats_dict["face"] = norm_face_feat
                    active_modalities["face_video"] = {
                        "label": labels[int(probs.argmax())],
                        "confidence": float(probs.max()),
                        "probs": probs.tolist()
                    }
                except Exception as e:
                    print(f"Error processing video frames: {e}")
                    
            # B. Extract Audio track from video to process as Audio Modality
            if audio_model is not None:
                try:
                    from moviepy import VideoFileClip
                    extracted_wav = temp_dir / "temp_video_audio.wav"
                    clip = VideoFileClip(str(temp_video_file))
                    clip.audio.write_audiofile(str(extracted_wav), fps=22050, logger=None)
                    clip.close()
                    
                    raw_audio_feat = preprocess_audio(str(extracted_wav))
                    cols = [f"mfcc_{i:02d}" for i in range(39)]
                    norm_feat = ((raw_audio_feat - audio_mean[cols]) / audio_std[cols]).values.astype(np.float32)
                    
                    x_aud = torch.tensor(norm_feat).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = audio_model(x_aud)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        
                    probs_dict["audio"] = probs.tolist()
                    feats_dict["audio"] = norm_feat
                    active_modalities["audio"] = {
                        "label": labels[int(probs.argmax())],
                        "confidence": float(probs.max()),
                        "probs": probs.tolist()
                    }
                    
                    # Delete temp extracted wav
                    if extracted_wav.exists():
                        os.remove(extracted_wav)
                except Exception as e:
                    print(f"Error extracting audio from video: {e}")
                    
        except Exception as e:
            print(f"Error serving video input: {e}")

    # Remove temporary uploads to prevent disk bloat
    for temp_f in [temp_audio_file, temp_video_file]:
        if temp_f and temp_f.exists():
            os.remove(temp_f)
            
    if len(active_modalities) == 0:
        raise HTTPException(status_code=400, detail="No active inputs found or processed successfully. Provide text, audio wav, or valid video mp4.")

    # ============================================================
    # MULTIMODAL FUSION DECISION
    # ============================================================
    fusion_results = {}
    
    # 1. Late Fusion Calculation with Graceful fallbacks
    w_a, w_t, w_f = late_fusion_weights
    
    curr_weights = []
    curr_probs = []
    
    # Accumulate available probabilities and apply weight re-normalization
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
        curr_weights = curr_weights / weight_sum # renormalize
        fused_probs = np.zeros(7)
        for w, p in zip(curr_weights, curr_probs):
            fused_probs += w * p
            
        fusion_results["late_fusion"] = {
            "label": labels[int(fused_probs.argmax())],
            "confidence": float(fused_probs.max()),
            "probs": fused_probs.tolist()
        }

    # 2. Early Fusion Calculation with Fallbacks (fills missing features with zeros)
    if early_fusion_model is not None:
        try:
            # Reconstruct the 1319 dimensions
            feat_a = feats_dict.get("audio", np.zeros(39, dtype=np.float32))
            feat_t = feats_dict.get("text", np.zeros(768, dtype=np.float32))
            feat_f = feats_dict.get("face", np.zeros(512, dtype=np.float32))
            
            joint_feat = np.concatenate([feat_a, feat_t, feat_f]).astype(np.float32)
            x_fuse = torch.tensor(joint_feat).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = early_fusion_model(x_fuse)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            fusion_results["early_fusion"] = {
                "label": labels[int(probs.argmax())],
                "confidence": float(probs.max()),
                "probs": probs.tolist()
            }
        except Exception as e:
            print(f"Error serving Early Fusion prediction: {e}")

    # Determine final recommendation
    primary_fusion = "late_fusion" if "late_fusion" in fusion_results else "early_fusion"
    if primary_fusion in fusion_results:
        final_prediction = {
            "label": fusion_results[primary_fusion]["label"],
            "confidence": fusion_results[primary_fusion]["confidence"],
        }
    else:
        # Fallback to key modality if fusion failed
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
