FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Set environment variables
ENV MODEL_PATH=/models/zimage
ENV HF_HOME=/models/hf_cache

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Z-Image Turbo model from HuggingFace
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir='/models/zimage')"

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
