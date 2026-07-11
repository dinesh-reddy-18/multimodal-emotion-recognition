# ==========================================
# STAGE 1: Model Compilation Builder
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install PyTorch, transformers, onnxscript, and onnx for ONNX compilation
RUN pip install --no-cache-dir torch transformers onnxscript onnx

# Copy compilation script
COPY scratch/export_text_onnx.py /build/scratch/export_text_onnx.py

# Run the compilation script to generate text_pipeline.onnx & data
RUN python /build/scratch/export_text_onnx.py

# ==========================================
# STAGE 2: Production Final Runner Image
# ==========================================
FROM python:3.10-slim

WORKDIR /code

# Install system dependencies needed for OpenCV, MediaPipe, and MoviePy
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (requirements.txt does NOT contain torch!)
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all project files
COPY . /code

# Copy the compiled text ONNX models from the builder stage
COPY --from=builder /build/onnx_models/text_pipeline.onnx /code/onnx_models/text_pipeline.onnx
COPY --from=builder /build/onnx_models/text_pipeline.onnx.data /code/onnx_models/text_pipeline.onnx.data

# Expose port 7860 (Hugging Face / Render default)
EXPOSE 7860

# Run FastAPI using uvicorn on port 7860
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
