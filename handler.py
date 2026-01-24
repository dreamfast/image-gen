"""
Z-Image Turbo RunPod Serverless Handler (BNB 4-bit + LoRAs)

Loads Z-Image Turbo 4-bit quantized model with LoRAs at container start,
then handles inference requests returning base64-encoded images.
"""

import base64
import io
import os
import sys
import traceback

print("=" * 60)
print("Z-Image Turbo Handler Starting (BNB 4-bit + LoRAs)...")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Import torch first and check CUDA
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
except Exception as e:
    print(f"ERROR importing torch: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import runpod
try:
    import runpod
    print("RunPod SDK loaded")
except Exception as e:
    print(f"ERROR importing runpod: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import diffusers and transformers for BNB config
try:
    from diffusers import DiffusionPipeline
    from transformers import BitsAndBytesConfig
    print("DiffusionPipeline and BitsAndBytesConfig imported successfully")
except Exception as e:
    print(f"ERROR importing diffusers/transformers: {e}")
    traceback.print_exc()
    sys.exit(1)

# Model paths (set during Docker build)
MODEL_PATH = os.environ.get("MODEL_PATH", "unsloth/Z-Image-Turbo-unsloth-bnb-4bit")
LORA_PATH = os.environ.get("LORA_PATH", "/models/loras")
HF_CACHE = os.environ.get("HF_HOME", "/models/hf_cache")

print(f"Model: {MODEL_PATH}")
print(f"LoRA path: {LORA_PATH}")
print(f"HF cache: {HF_CACHE}")

if os.path.exists(LORA_PATH):
    print(f"LoRA files: {os.listdir(LORA_PATH)}")

# LoRA configuration
LORAS = [
    {"name": "better_nudes", "file": "better_nudes.safetensors", "strength": 0.8},
    {"name": "photo_enhance", "file": "photo_enhance.safetensors", "strength": 0.8},
    {"name": "nice_asians", "file": "nice_asians.safetensors", "strength": 0.2},
]

# Default generation settings
DEFAULTS = {
    "width": 512,
    "height": 512,
    "steps": 9,  # 9 steps = 8 DiT forwards (NFEs)
    "quality": 75,  # WebP quality (1-100)
}


def load_pipeline():
    """Load the Z-Image Turbo 4-bit pipeline with LoRAs."""
    print("Loading Z-Image Turbo 4-bit pipeline...")

    # Clear any existing CUDA memory
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load the 4-bit quantized model with memory optimizations
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        low_cpu_mem_usage=True,  # Load weights sequentially to reduce peak memory
    )

    # Move to GPU
    print("Moving pipeline to CUDA...")
    pipe.to("cuda")

    # Load LoRAs with their respective strengths
    print("Loading LoRAs...")
    adapter_names = []
    adapter_weights = []

    for lora in LORAS:
        lora_file = os.path.join(LORA_PATH, lora["file"])
        if os.path.exists(lora_file):
            try:
                print(f"  Loading {lora['name']} (strength: {lora['strength']})...")
                pipe.load_lora_weights(lora_file, adapter_name=lora["name"])
                adapter_names.append(lora["name"])
                adapter_weights.append(lora["strength"])
                print(f"    LoRA {lora['name']} loaded successfully")
            except Exception as e:
                print(f"    WARNING: Failed to load LoRA {lora['name']}: {e}")
        else:
            print(f"    WARNING: LoRA file not found: {lora_file}")

    # Set all adapters with their weights
    if adapter_names:
        print(f"Setting adapters: {adapter_names} with weights: {adapter_weights}")
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        print(f"  {len(adapter_names)} LoRAs active")
    else:
        print("  No LoRAs loaded")

    # Memory optimizations
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing(1)
        print("  Enabled attention slicing")

    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
        print("  Enabled VAE slicing")

    # Enable model CPU offload if needed (uncomment if hitting OOM)
    # if hasattr(pipe, 'enable_model_cpu_offload'):
    #     pipe.enable_model_cpu_offload()
    #     print("  Enabled model CPU offload")

    # Clear any cached memory
    torch.cuda.empty_cache()

    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Pipeline loaded successfully!")
    return pipe


# Load pipeline at container start (stays warm)
try:
    pipe = load_pipeline()
    print("=" * 60)
    print("Handler ready to accept requests!")
    print("=" * 60)
except Exception as e:
    print(f"FATAL ERROR loading pipeline: {e}")
    traceback.print_exc()
    sys.exit(1)


def handler(job):
    """
    Handle image generation request.

    Input:
        prompt (str): Image description
        width (int, optional): Image width (default: 512)
        height (int, optional): Image height (default: 512)
        steps (int, optional): Inference steps (default: 9, which = 8 NFEs)
        seed (int, optional): Random seed (default: random)
        quality (int, optional): WebP quality 1-100 (default: 85)

    Output:
        image (str): Base64-encoded WebP image
        format (str): "webp"
        width (int): Image width
        height (int): Image height
        seed (int|null): Random seed used
        gpu (str): GPU name used

    Note: Z-Image Turbo uses guidance_scale=0.0 (no CFG)
    """
    job_input = job["input"]

    # Extract parameters
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required parameter: prompt"}

    width = job_input.get("width", DEFAULTS["width"])
    height = job_input.get("height", DEFAULTS["height"])
    steps = job_input.get("steps", DEFAULTS["steps"])
    seed = job_input.get("seed")
    quality = job_input.get("quality", DEFAULTS["quality"])

    # Cap size to prevent OOM (max ~1MP on 24GB)
    max_pixels = 1024 * 1024
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
        print(f"Resized to {width}x{height} to prevent OOM")

    print(f"Generating image: {width}x{height}, steps={steps}, seed={seed}, quality={quality}")
    print(f"CUDA memory before generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Prompt: {prompt[:100] if len(prompt) > 100 else prompt}")

    # Create generator with seed
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))
    else:
        generator = None

    try:
        # Generate image with inference mode for memory efficiency
        # Add timeout to prevent stuck workers
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Image generation timed out after 30 seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout

        with torch.inference_mode():
            # Z-Image Turbo uses guidance_scale=0.0 (no CFG)
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            )

        image = result.images[0]

        # Convert to base64 WebP (optimized for web)
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=quality, method=6)
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Cancel timeout
        signal.alarm(0)

        # Clean up to prevent memory buildup
        del result
        torch.cuda.empty_cache()

        print(f"Image generated successfully: {len(image_b64)} bytes")
        print(f"CUDA memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return {
            "image": image_b64,
            "format": "webp",
            "width": width,
            "height": height,
            "seed": seed,
            "gpu": torch.cuda.get_device_name(0),
        }

    except TimeoutError as e:
        signal.alarm(0)
        torch.cuda.empty_cache()
        print(f"TIMEOUT ERROR: {e}")
        return {"error": "Image generation timed out after 30 seconds"}

    except torch.cuda.OutOfMemoryError as e:
        signal.alarm(0)
        torch.cuda.empty_cache()
        print(f"OOM ERROR: {e}")
        return {"error": f"Out of memory - try smaller image size (current: {width}x{height})"}

    except Exception as e:
        signal.alarm(0)
        torch.cuda.empty_cache()
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
