FROM runpod/pytorch:2.8.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Set environment variables
ENV MODEL_PATH=/models/zimage
ENV HF_HOME=/models/hf_cache
ENV PYTHONUNBUFFERED=1


# Verify PyTorch version (need 2.5+ for enable_gqa in scaled_dot_product_attention)
RUN python -c "import torch; v=torch.__version__; print(f'PyTorch: {v}'); assert tuple(map(int, v.split('+')[0].split('.')[:2])) >= (2,5), f'Need PyTorch 2.5+, got {v}'"

# Install git (needed for pip install from github)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install diffusers from source (required for ZImagePipeline)
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# Verify diffusers version and ZImagePipeline availability
RUN python -c "import diffusers; print(f'diffusers version: {diffusers.__version__}')"
RUN python -c "from diffusers import ZImagePipeline; print('ZImagePipeline imported successfully!')"

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Z-Image Turbo model from HuggingFace
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir='/models/zimage')"

# List model files to verify download
RUN ls -la /models/zimage/

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
