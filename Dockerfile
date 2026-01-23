FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2204

WORKDIR /app

# Set environment variables
ENV MODEL_PATH=/models/zimage
ENV HF_HOME=/models/hf_cache
ENV PYTHONUNBUFFERED=1

# Check initial PyTorch version
RUN python -c "import torch; print(f'Initial PyTorch: {torch.__version__}')"

# Install git (needed for pip install from github)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install remaining dependencies FIRST (before diffusers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install diffusers from source WITHOUT letting it change torch
# --no-deps prevents diffusers from downgrading torch
RUN pip install --no-cache-dir --no-deps git+https://github.com/huggingface/diffusers.git

# Install diffusers dependencies that we don't already have (excluding torch)
RUN pip install --no-cache-dir regex requests filelock numpy Pillow

# CRITICAL: Verify PyTorch version AFTER all installs
RUN python -c "import torch; v=torch.__version__; print(f'Final PyTorch: {v}'); assert tuple(map(int, v.split('+')[0].split('.')[:2])) >= (2,5), f'Need PyTorch 2.5+, got {v}'"

# Verify diffusers and ZImagePipeline
RUN python -c "import diffusers; print(f'diffusers version: {diffusers.__version__}')"
RUN python -c "from diffusers import ZImagePipeline; print('ZImagePipeline imported successfully!')"

# Download Z-Image Turbo model from HuggingFace
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir='/models/zimage')"

# List model files to verify download
RUN ls -la /models/zimage/

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
