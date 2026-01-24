FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /app

# Set environment variables
ENV MODEL_PATH=/models/zimage
ENV GGUF_PATH=/models/z-image-turbo-Q6_K.gguf
ENV LORA_PATH=/models/loras
ENV HF_HOME=/models/hf_cache
ENV PYTHONUNBUFFERED=1
# Enable optimized CUDA kernels for GGUF (torch27-cu128 is supported)
ENV DIFFUSERS_GGUF_CUDA_KERNELS=true

# Check initial PyTorch version
RUN python -c "import torch; print(f'Initial PyTorch: {torch.__version__}')"

# Install git and wget (needed for downloads)
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Install remaining dependencies FIRST (before diffusers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gguf package for GGUF model loading
RUN pip install --no-cache-dir gguf

# Install kernels for optimized CUDA GGUF inference (torch27-cu128 supported)
RUN pip install --no-cache-dir kernels

# Install diffusers from source WITHOUT letting it change torch
# --no-deps prevents diffusers from downgrading torch
RUN pip install --no-cache-dir --no-deps git+https://github.com/huggingface/diffusers.git

# Install diffusers dependencies that we don't already have (excluding torch)
RUN pip install --no-cache-dir regex requests filelock numpy Pillow peft

# CRITICAL: Verify PyTorch version AFTER all installs
RUN python -c "import torch; v=torch.__version__; print(f'Final PyTorch: {v}'); assert tuple(map(int, v.split('+')[0].split('.')[:2])) >= (2,5), f'Need PyTorch 2.5+, got {v}'"

# Verify diffusers, ZImagePipeline, and GGUF support
RUN python -c "import diffusers; print(f'diffusers version: {diffusers.__version__}')"
RUN python -c "from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig; print('GGUF imports successful!')"

# Download Z-Image Turbo components (text encoder, VAE, tokenizer, scheduler)
# We exclude only the transformer weights since we're using GGUF for that
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir='/models/zimage', \
    ignore_patterns=['transformer/*'])"

# Download GGUF model from Hugging Face
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('unsloth/Z-Image-Turbo-GGUF', 'z-image-turbo-Q6_K.gguf', \
    local_dir='/models', local_dir_use_symlinks=False)"

# Verify GGUF download
RUN ls -la /models/*.gguf

# Create LoRA directory
RUN mkdir -p /models/loras

# Download LoRAs from BunnyCDN
# Note: We rename them to simpler names on download to avoid path issues
RUN wget -O /models/loras/better_nudes.safetensors "https://romancify-build.b-cdn.net/b3tternud3s_v2.safetensors" && \
    wget -O /models/loras/photo_enhance.safetensors "https://romancify-build.b-cdn.net/OB%E6%8B%8D%E7%AB%8B%E5%BE%97%E4%BA%BA%E5%83%8F%E6%91%84%E5%BD%B1Instant%20camera%20portrait%20photography%20V4.0.safetensors" && \
    wget -O /models/loras/nice_asians.safetensors "https://romancify-build.b-cdn.net/NIceAsians_Zimage.safetensors"

# Verify LoRA downloads
RUN ls -la /models/loras/

# List model files to verify download
RUN ls -la /models/zimage/

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
