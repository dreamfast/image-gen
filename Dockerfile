# Use RunPod's optimized PyTorch image (pre-cached, worked with full model)
# PyTorch 2.8.0 + CUDA 12.81 - 4-bit model should use less VRAM
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2204

WORKDIR /app

# Set environment variables
ENV MODEL_PATH=unsloth/Z-Image-Turbo-unsloth-bnb-4bit
ENV LORA_PATH=/models/loras
ENV HF_HOME=/models/hf_cache
ENV PYTHONUNBUFFERED=1

# Check initial PyTorch version
RUN python -c "import torch; print(f'Initial PyTorch: {torch.__version__}')"

# Install git and wget (needed for downloads)
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Install remaining dependencies FIRST (before diffusers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install bitsandbytes for 4-bit quantization
RUN pip install --no-cache-dir bitsandbytes

# Install diffusers from source WITHOUT letting it change torch
# --no-deps prevents diffusers from downgrading torch
RUN pip install --no-cache-dir --no-deps git+https://github.com/huggingface/diffusers.git

# Install diffusers dependencies that we don't already have (excluding torch)
# Note: peft is in requirements.txt, accelerate handles bitsandbytes integration
RUN pip install --no-cache-dir regex requests filelock numpy Pillow

# CRITICAL: Verify PyTorch version AFTER all installs (need 2.5+ for diffusers GQA support)
RUN python -c "import torch; v=torch.__version__; print(f'Final PyTorch: {v}'); major_minor = tuple(map(int, v.split('+')[0].split('.')[:2])); assert major_minor >= (2,5), f'Need PyTorch 2.5+, got {v}'"

# Verify diffusers and bitsandbytes
RUN python -c "import diffusers; print(f'diffusers version: {diffusers.__version__}')"
RUN python -c "import bitsandbytes; print('bitsandbytes imported successfully')"

# Pre-download the 4-bit quantized model files (without loading - no GPU during build)
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('unsloth/Z-Image-Turbo-unsloth-bnb-4bit', \
    cache_dir='/models/hf_cache')"

# Create LoRA directory
RUN mkdir -p /models/loras

# Download LoRAs from BunnyCDN
# Note: We rename them to simpler names on download to avoid path issues
RUN wget -O /models/loras/better_nudes.safetensors "https://romancify-build.b-cdn.net/b3tternud3s_v2.safetensors" && \
    wget -O /models/loras/photo_enhance.safetensors "https://romancify-build.b-cdn.net/OB%E6%8B%8D%E7%AB%8B%E5%BE%97%E4%BA%BA%E5%83%8F%E6%91%84%E5%BD%B1Instant%20camera%20portrait%20photography%20V4.0.safetensors" && \
    wget -O /models/loras/nice_asians.safetensors "https://romancify-build.b-cdn.net/NIceAsians_Zimage.safetensors"

# Verify LoRA downloads
RUN ls -la /models/loras/

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
