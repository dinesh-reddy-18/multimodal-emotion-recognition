import os
import cv2
import librosa
import numpy as np
import warnings
import tempfile
import requests
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server safety
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from moviepy import VideoFileClip
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

warnings.filterwarnings("ignore")

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
audio_mean = audio_stats["mean"]
audio_std = audio_stats["std"]

face_stats = np.load("onnx_models/face_stats.npz", allow_pickle=True)
face_mean = face_stats["mean"]
face_std = face_stats["std"]

text_stats = np.load("onnx_models/text_stats.npz", allow_pickle=True)
text_mean = text_stats["mean"]
text_std = text_stats["std"]

late_weights_data = np.load("onnx_models/late_fusion_weights.npz", allow_pickle=True)
late_fusion_weights = late_weights_data["weights"]

# Load ONNX sessions on CPU (very low memory!)
print("Loading ONNX sessions...")
audio_session = ort.InferenceSession("onnx_models/audio_model.onnx", providers=["CPUExecutionProvider"])
face_session = ort.InferenceSession("onnx_models/face_pipeline.onnx", providers=["CPUExecutionProvider"])
early_fusion_session = ort.InferenceSession("onnx_models/early_fusion.onnx", providers=["CPUExecutionProvider"])
print("ONNX sessions loaded successfully.")

# MediaPipe BlazeFace Detector
detector = None
try:
    model_path = "models/face/blaze_face_short_range.tflite"
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceDetectorOptions(base_options=base_options)
    detector = mp_vision.FaceDetector.create_from_options(options)
    print("MediaPipe Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading MediaPipe detector: {e}")

# ============================================================
# HUGGING FACE INFERENCE API CLIENT
# ============================================================

def query_hf_api(model_name: str, payload: dict) -> dict:
    """Helper to query Hugging Face Inference API."""
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    response = requests.post(API_URL, json=payload, headers=headers, timeout=12)
    response.raise_for_status()
    return response.json()

def get_text_predictions(text: str):
    """Retrieve text emotion probabilities from HF Inference API with loading-state retries."""
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    for retry in range(3):
        try:
            res = query_hf_api("j-hartmann/emotion-english-distilroberta-base", payload)
            if isinstance(res, dict) and "error" in res:
                print(f"HF API returned error: {res['error']}. Retrying...")
                time.sleep(3)
                continue
                
            # Parse result: list of dicts like [[{"label": "joy", "score": 0.98}, ...]]
            if isinstance(res, list) and len(res) > 0:
                items = res[0]
                label_map = {
                    'anger': 'angry',
                    'disgust': 'disgust',
                    'fear': 'fearful',
                    'joy': 'happy',
                    'neutral': 'neutral',
                    'sadness': 'sad',
                    'surprise': 'surprised'
                }
                probs = np.zeros(7)
                for item in items:
                    label_name = item['label']
                    score = item['score']
                    mapped_name = label_map.get(label_name, label_name)
                    if mapped_name in label2idx:
                        probs[label2idx[mapped_name]] = score
                return probs
        except Exception as e:
            print(f"Text prediction API attempt {retry} failed: {e}")
            time.sleep(2)
            
    # Fallback: simple text classification based on keywords in case API is offline
    print("HF API offline. Falling back to rule-based text prediction.")
    probs = np.ones(7) * 0.05
    probs[label2idx['neutral']] = 0.70
    text_lower = text.lower()
    if any(w in text_lower for w in ["happy", "glad", "joy", "wonderful", "great", "awesome", "excited"]):
        probs[label2idx['happy']] = 0.90
        probs[label2idx['neutral']] = 0.05
    elif any(w in text_lower for w in ["sad", "depressed", "unhappy", "cry", "sorrow", "pain", "hurt"]):
        probs[label2idx['sad']] = 0.90
        probs[label2idx['neutral']] = 0.05
    elif any(w in text_lower for w in ["angry", "mad", "furious", "hate", "rage"]):
        probs[label2idx['angry']] = 0.90
        probs[label2idx['neutral']] = 0.05
    elif any(w in text_lower for w in ["afraid", "fear", "scared", "terrified", "frightened"]):
        probs[label2idx['fearful']] = 0.90
        probs[label2idx['neutral']] = 0.05
    
    return probs / probs.sum()

def get_text_embedding(text: str):
    """Retrieve 768-dim RoBERTa CLS embedding from HF Inference API."""
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    for retry in range(3):
        try:
            res = query_hf_api("roberta-base", payload)
            if isinstance(res, list) and len(res) > 0 and len(res[0]) > 0:
                cls_emb = np.array(res[0][0], dtype=np.float32)
                if cls_emb.shape == (768,):
                    return cls_emb
        except Exception as e:
            print(f"Text embedding API attempt {retry} failed: {e}")
            time.sleep(2)
            
    # Fallback CLS embedding
    print("HF API offline. Returning zero vector for text embedding.")
    return np.zeros(768, dtype=np.float32)

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

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
            if cropped is not None:
                chw = preprocess_face_numpy(cropped)
                frame_tensors.append(chw)
                
    cap.release()
    
    if len(frame_tensors) == 0:
        raise ValueError("No faces detected in video file.")
        
    x = np.stack(frame_tensors).astype(np.float32)
    
    # Run the face pipeline ONNX session
    features_out, logits_out = face_session.run(None, {"input_image": x})
    
    # Average the features across the batch axis for early fusion
    pooled_feat = features_out.mean(axis=0)
    
    return features_out, logits_out, pooled_feat

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

# ============================================================
# CORE PREDICTION INTERFACE
# ============================================================

def analyze_multimodal_emotion(text_input, audio_input_path, video_input_path):
    active_modalities = {}
    probs_dict = {}
    feats_dict = {}
    
    temp_files = []
    
    # 1. PROCESS TEXT MODALITY
    if text_input and text_input.strip():
        try:
            # Call HF Inference API for predictions
            probs = get_text_predictions(text_input)
            probs_dict["text"] = probs
            active_modalities["text"] = {labels[i]: float(probs[i]) for i in range(7)}
            
            # Call HF Inference API for features (early fusion)
            raw_text_feat = get_text_embedding(text_input)
            
            # Normalize embedding
            mean_vals = np.array([text_mean[f"roberta_{i}"] for i in range(768)], dtype=np.float32)
            std_vals = np.array([text_std[f"roberta_{i}"] for i in range(768)], dtype=np.float32)
            norm_feat = (raw_text_feat - mean_vals) / std_vals
            
            feats_dict["text"] = norm_feat.astype(np.float32)
        except Exception as e:
            print(f"Error serving text analysis: {e}")
            
    # 2. PROCESS AUDIO MODALITY
    if audio_input_path:
        try:
            raw_audio_feat = preprocess_audio(audio_input_path)
            
            # Normalize MFCCs
            mean_vals = np.array([audio_mean[f"mfcc_{i:02d}"] for i in range(39)], dtype=np.float32)
            std_vals = np.array([audio_std[f"mfcc_{i:02d}"] for i in range(39)], dtype=np.float32)
            norm_feat = (raw_audio_feat - mean_vals) / std_vals
            
            x_aud = norm_feat.astype(np.float32).reshape(1, 39)
            
            # Run ONNX session
            logits = audio_session.run(None, {"input": x_aud})[0]
            probs = softmax(logits)[0]
            
            probs_dict["audio"] = probs
            feats_dict["audio"] = norm_feat.astype(np.float32)
            active_modalities["audio"] = {labels[i]: float(probs[i]) for i in range(7)}
        except Exception as e:
            print(f"Error serving audio analysis: {e}")
            
    # 3. PROCESS VIDEO MODALITY
    if video_input_path:
        try:
            # A. Process Video Frames (Face crops classification)
            try:
                features_out, logits_out, pooled_face_feat = preprocess_video(video_input_path)
                
                # Normalize pooled features for early fusion representation
                mean_vals = np.array([face_mean[f"face_{i:03d}"] for i in range(512)], dtype=np.float32)
                std_vals = np.array([face_std[f"face_{i:03d}"] for i in range(512)], dtype=np.float32)
                norm_face_feat = (pooled_face_feat - mean_vals) / std_vals
                
                # Compute predictions per frame individually and aggregate temporally
                all_probs = [softmax(l)[0] for l in logits_out]
                face_probs = np.mean(all_probs, axis=0)
                
                probs_dict["face"] = face_probs
                feats_dict["face"] = norm_face_feat.astype(np.float32)
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
                        mean_vals = np.array([audio_mean[f"mfcc_{i:02d}"] for i in range(39)], dtype=np.float32)
                        std_vals = np.array([audio_std[f"mfcc_{i:02d}"] for i in range(39)], dtype=np.float32)
                        norm_feat = (raw_audio_feat - mean_vals) / std_vals
                        
                        x_aud = norm_feat.astype(np.float32).reshape(1, 39)
                        logits = audio_session.run(None, {"input": x_aud})[0]
                        probs = softmax(logits)[0]
                        
                        probs_dict["audio"] = probs
                        feats_dict["audio"] = norm_feat.astype(np.float32)
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
    try:
        feat_a = feats_dict.get("audio", np.zeros(39, dtype=np.float32))
        feat_t = feats_dict.get("text", np.zeros(768, dtype=np.float32))
        feat_f = feats_dict.get("face", np.zeros(512, dtype=np.float32))
        
        joint_feat = np.concatenate([feat_a, feat_t, feat_f]).astype(np.float32).reshape(1, 1319)
        
        logits = early_fusion_session.run(None, {"input": joint_feat})[0]
        probs = softmax(logits)[0]
            
        fusion_results["early_fusion"] = {labels[i]: float(probs[i]) for i in range(7)}
    except Exception as e:
        print(f"Error serving Early Fusion: {e}")
            
    # Clean up temporary audio files
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
            
    submit_btn.click(
        fn=analyze_multimodal_emotion,
        inputs=[text_input, audio_input, video_input],
        outputs=[fused_label, early_label, text_label, audio_label, face_label, chart_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, share=True)
