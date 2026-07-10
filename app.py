import os
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import tempfile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server safety
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from pathlib import Path
from moviepy import VideoFileClip
from torchvision import models, transforms
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForSequenceClassification
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

warnings.filterwarnings("ignore")

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Gradio App running on device: {DEVICE}")

# Define the MLP model classes exactly as trained
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

# Setup Global variables to hold models and statistics
audio_model = None
audio_mean = None
audio_std = None

text_model = None
text_mean = None
text_std = None
roberta_tokenizer = None
roberta_model = None
text_pretrained_tokenizer = None
text_pretrained_model = None

face_model = None
face_mean = None
face_std = None
resnet18 = None

early_fusion_model = None
late_fusion_weights = None
detector = None

labels = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

# Load all components (called once on startup)
def load_all_components():
    global audio_model, audio_mean, audio_std
    global text_model, text_mean, text_std, roberta_tokenizer, roberta_model
    global text_pretrained_tokenizer, text_pretrained_model
    global face_model, face_mean, face_std, resnet18, detector
    global early_fusion_model, late_fusion_weights
    
    print("Loading ML models and pipeline checkpoints...")
    
    # 1. MediaPipe BlazeFace Detector
    try:
        model_path = "models/face/blaze_face_short_range.tflite"
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(base_options=base_options)
        detector = mp_vision.FaceDetector.create_from_options(options)
        print("MediaPipe Face Detector loaded successfully.")
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
        print("Audio MLP baseline loaded successfully.")
    except Exception as e:
        print(f"Error loading Audio model: {e}")

    # 3. Text Baseline & Pre-trained Emotion Models
    try:
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
        print(f"Error loading Text models: {e}")

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
        print("Face MLP and ResNet18 model loaded successfully.")
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
        print(f"Late Fusion config loaded. Weights: {late_fusion_weights}")
    except Exception as e:
        print(f"Error loading Late Fusion config: {e}")
        late_fusion_weights = (0.33, 0.33, 0.34)

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_text(text: str) -> np.ndarray:
    """Tokenize and extract RoBERTa [CLS] pooled embedding."""
    inputs = roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
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

def generate_distribution_plot(probs_dict, labels_list):
    """Generate Matplotlib bar chart for visual feedback."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#111827')
    ax.set_facecolor('#1f2937')
    
    x = np.arange(len(labels_list))
    width = 0.2
    
    colors = {
        'text': '#3b82f6',   # Blue
        'audio': '#8b5cf6',  # Purple
        'face': '#ec4899',   # Pink
        'fused': '#10b981'   # Green
    }
    
    label_mapping = {
        'text': 'Text Modality',
        'audio': 'Audio Modality',
        'face': 'Face Modality',
        'fused': 'Fused (Late Fusion)'
    }
    
    active_keys = [k for k in ['text', 'audio', 'face', 'fused'] if k in probs_dict]
    
    for idx, key in enumerate(active_keys):
        offset = (idx - (len(active_keys) - 1) / 2) * width
        ax.bar(x + offset, probs_dict[key], width, label=label_mapping[key], color=colors[key])
        
    ax.set_ylabel('Confidence', color='#f3f4f6', fontsize=10)
    ax.set_title('Modality-wise Emotion Confidence Levels', color='#f3f4f6', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in labels_list], color='#f3f4f6', fontsize=9)
    ax.tick_params(colors='#f3f4f6')
    ax.legend(facecolor='#111827', labelcolor='#f3f4f6', edgecolor='none', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, color='rgba(255, 255, 255, 0.05)', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    return fig

# Initialize checkpoints
load_all_components()

# ============================================================
# CORE PREDICTION INTERFACE
# ============================================================

def analyze_multimodal_emotion(text_input, audio_input_path, video_input_path):
    # Setup track variables
    active_modalities = {}
    probs_dict = {}
    feats_dict = {}
    
    temp_files = []
    
    # 1. PROCESS TEXT MODALITY
    if text_input and text_input.strip():
        try:
            # Feature extraction for early fusion
            raw_text_feat = preprocess_text(text_input)
            cols = [f"roberta_{i}" for i in range(768)]
            norm_feat = ((raw_text_feat - text_mean[cols]) / text_std[cols]).values.astype(np.float32)
            
            # Predict using GoEmotions classifier
            pretrained_inputs = text_pretrained_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                pretrained_logits = text_pretrained_model(**pretrained_inputs).logits
                probs = torch.softmax(pretrained_logits, dim=1).cpu().numpy()[0]
                
            probs_dict["text"] = probs
            feats_dict["text"] = norm_feat
            active_modalities["text"] = {labels[i]: float(probs[i]) for i in range(7)}
        except Exception as e:
            print(f"Error serving text analysis: {e}")
            
    # 2. PROCESS AUDIO MODALITY (Direct Microphone or File Upload)
    if audio_input_path:
        try:
            raw_audio_feat = preprocess_audio(audio_input_path)
            cols = [f"mfcc_{i:02d}" for i in range(39)]
            norm_feat = ((raw_audio_feat - audio_mean[cols]) / audio_std[cols]).values.astype(np.float32)
            
            x_aud = torch.tensor(norm_feat).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = audio_model(x_aud)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            probs_dict["audio"] = probs
            feats_dict["audio"] = norm_feat
            active_modalities["audio"] = {labels[i]: float(probs[i]) for i in range(7)}
        except Exception as e:
            print(f"Error serving audio analysis: {e}")
            
    # 3. PROCESS VIDEO MODALITY (Contains Face frames & Optional Video Audio track)
    if video_input_path:
        try:
            # A. Process Video Frames (Face crops classification)
            try:
                raw_face_features, pooled_face_feat = preprocess_video(video_input_path)
                cols = [f"face_{i:03d}" for i in range(512)]
                
                # Normalize pooled features for early fusion representation
                norm_face_feat = ((pooled_face_feat - face_mean[cols]) / face_std[cols]).values.astype(np.float32)
                
                # Compute predictions per frame individually and aggregate temporally
                all_probs = []
                for frame_feat in raw_face_features:
                    norm_frame_feat = ((frame_feat - face_mean[cols]) / face_std[cols]).values.astype(np.float32)
                    x_face = torch.tensor(norm_frame_feat).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = face_model(x_face)
                        frame_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    all_probs.append(frame_probs)
                    
                face_probs = np.mean(all_probs, axis=0)
                
                probs_dict["face"] = face_probs
                feats_dict["face"] = norm_face_feat
                active_modalities["face"] = {labels[i]: float(face_probs[i]) for i in range(7)}
            except Exception as e:
                print(f"Error processing video face frames: {e}")
                
            # B. Extract Audio track from video if not already processed a separate audio
            if "audio" not in probs_dict:
                try:
                    temp_dir = tempfile.gettempdir()
                    extracted_wav = os.path.join(temp_dir, "extracted_video_audio.wav")
                    
                    clip = VideoFileClip(video_input_path)
                    if clip.audio is not None:
                        clip.audio.write_audiofile(extracted_wav, fps=22050, logger=None)
                        clip.close()
                        temp_files.append(extracted_wav)
                        
                        # Process extracted audio
                        raw_audio_feat = preprocess_audio(extracted_wav)
                        cols = [f"mfcc_{i:02d}" for i in range(39)]
                        norm_feat = ((raw_audio_feat - audio_mean[cols]) / audio_std[cols]).values.astype(np.float32)
                        
                        x_aud = torch.tensor(norm_feat).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            logits = audio_model(x_aud)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            
                        probs_dict["audio"] = probs
                        feats_dict["audio"] = norm_feat
                        active_modalities["audio"] = {labels[i]: float(probs[i]) for i in range(7)}
                    else:
                        clip.close()
                except Exception as e:
                    print(f"Error extracting audio from video: {e}")
                    
        except Exception as e:
            print(f"Error serving video analysis: {e}")
            
    # 4. MULTIMODAL FUSION
    fusion_results = {}
    
    # Late Fusion weights
    w_a, w_t, w_f = late_fusion_weights
    curr_weights = []
    curr_probs = []
    
    if "audio" in probs_dict:
        curr_weights.append(w_a)
        curr_probs.append(probs_dict["audio"])
    if "text" in probs_dict:
        curr_weights.append(w_t)
        curr_probs.append(probs_dict["text"])
    if "face" in probs_dict:
        curr_weights.append(w_f)
        curr_probs.append(probs_dict["face"])
        
    curr_weights = np.array(curr_weights)
    weight_sum = curr_weights.sum()
    
    if weight_sum > 0:
        curr_weights = curr_weights / weight_sum  # Renormalize active weights
        fused_probs = np.zeros(7)
        for w, p in zip(curr_weights, curr_probs):
            fused_probs += w * p
            
        probs_dict["fused"] = fused_probs
        fusion_results["late_fusion"] = {labels[i]: float(fused_probs[i]) for i in range(7)}
    
    # Early Fusion
    if early_fusion_model is not None:
        try:
            feat_a = feats_dict.get("audio", np.zeros(39, dtype=np.float32))
            feat_t = feats_dict.get("text", np.zeros(768, dtype=np.float32))
            feat_f = feats_dict.get("face", np.zeros(512, dtype=np.float32))
            
            joint_feat = np.concatenate([feat_a, feat_t, feat_f]).astype(np.float32)
            x_fuse = torch.tensor(joint_feat).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = early_fusion_model(x_fuse)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            fusion_results["early_fusion"] = {labels[i]: float(probs[i]) for i in range(7)}
        except Exception as e:
            print(f"Error serving Early Fusion: {e}")
            
    # Clean up temporary audio files to prevent memory leak / disk bloat
    for path in temp_files:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error removing temporary file {path}: {e}")

    # Fallback checks
    if len(probs_dict) == 0:
        raise gr.Error("Please provide at least one valid input (Text, Audio, or Video) to perform prediction.")
        
    # Generate Matplotlib chart
    chart_fig = generate_distribution_plot(probs_dict, labels)
    
    # Map predictions to output components
    fused_out = fusion_results.get("late_fusion", list(active_modalities.values())[0])
    early_out = fusion_results.get("early_fusion", {})
    text_out = active_modalities.get("text", {})
    audio_out = active_modalities.get("audio", {})
    face_out = active_modalities.get("face", {})
    
    return fused_out, early_out, text_out, audio_out, face_out, chart_fig

# ============================================================
# GRADIO INTERFACE LAYOUT
# ============================================================

# Dark premium CSS styling
custom_css = """
body {
    background-color: #0b0f19;
    color: #f3f4f6;
    font-family: 'Outfit', 'Inter', sans-serif;
}
.gradio-container {
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    max-width: 1100px !important;
    margin: 2rem auto !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5) !important;
}
h1, h2, h3 {
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.025em !important;
}
button.primary {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate"), css=custom_css, title="Multimodal Emotion Recognition") as demo:
    gr.HTML("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Multimodal Emotion Recognition System
            </h1>
            <p style='color: #94a3b8; font-size: 1.1rem;'>
                Analyze human emotions from voice pitch, facial expressions, and text context using Deep Learning Fusion.
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3>📥 Input Modalities</h3>")
            
            text_input = gr.Textbox(
                label="Text Input Context",
                placeholder="Type your emotional sentence here (e.g. 'I am feeling so sad today')...",
                lines=2
            )
            
            audio_input = gr.Audio(
                label="Voice Input (Microphone / WAV Upload)",
                type="filepath",
                sources=["microphone", "upload"]
            )
            
            video_input = gr.Video(
                label="Facial Expression (Webcam / MP4 Video Upload)",
                sources=["webcam", "upload"]
            )
            
            submit_btn = gr.Button("Analyze Modalities & Fused Output", variant="primary")
            
        with gr.Column(scale=1):
            gr.HTML("<h3>📊 Emotion Classifier Results</h3>")
            
            fused_label = gr.Label(num_top_classes=3, label="Final Recommendation (Late Fusion)")
            early_label = gr.Label(num_top_classes=3, label="Early Fusion Prediction (Joint MLP)")
            
            with gr.Accordion("Individual Modality Details", open=False):
                text_label = gr.Label(num_top_classes=3, label="Text Modality Prediction")
                audio_label = gr.Label(num_top_classes=3, label="Audio Modality Prediction")
                face_label = gr.Label(num_top_classes=3, label="Face Modality Prediction")
                
            chart_plot = gr.Plot(label="Modality Confidence Comparison")
            
    # Connect submit click to inference handler
    submit_btn.click(
        fn=analyze_multimodal_emotion,
        inputs=[text_input, audio_input, video_input],
        outputs=[fused_label, early_label, text_label, audio_label, face_label, chart_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, share=True)
