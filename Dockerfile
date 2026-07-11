FROM python:3.10-slim

WORKDIR /code

# Install system dependencies needed for OpenCV, MediaPipe, and MoviePy
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all project files
COPY . /code

# Compile the pre-trained text model to ONNX format locally
RUN python scratch/export_text_onnx.py

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run FastAPI using uvicorn on port 7860
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
