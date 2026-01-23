"""
Z-Image Turbo RunPod Serverless Handler

Loads Z-Image Turbo model at container start, then handles
inference requests returning base64-encoded images.
"""

import base64
import io
import os
import sys
import traceback

print("=" * 60)
print("Z-Image Turbo Handler Starting...")
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
except Exception as e:
    print(f"ERROR importing torch: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import runpod
try:
    import runpod
    print(f"RunPod SDK loaded")
except Exception as e:
    print(f"ERROR importing runpod: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import diffusers - this is the likely failure point
try:
    from diffusers import ZImagePipeline
    print(f"ZImagePipeline imported successfully")
except ImportError as e:
    print(f"ERROR: ZImagePipeline not found in diffusers!")
    print(f"This usually means diffusers was not installed from source.")
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"ERROR importing ZImagePipeline: {e}")
    traceback.print_exc()
    sys.exit(1)

# Model path (set during Docker build)
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/zimage")
print(f"Model path: {MODEL_PATH}")
print(f"Model path exists: {os.path.exists(MODEL_PATH)}")
if os.path.exists(MODEL_PATH):
    print(f"Model contents: {os.listdir(MODEL_PATH)[:10]}...")

# Default generation settings
DEFAULTS = {
    "width": 512,
    "height": 512,
    "steps": 9,  # 9 steps = 8 DiT forwards (NFEs)
    "quality": 85,  # WebP quality (1-100)
}


def load_pipeline():
    """Load the Z-Image Turbo pipeline with memory optimizations."""
    print("Loading Z-Image Turbo pipeline...")
    print(f"  torch_dtype: float16")
    print(f"  low_cpu_mem_usage: True")

    pipe = ZImagePipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,  # float16 uses less memory than bfloat16
        low_cpu_mem_usage=True,
    )

    # Load directly to GPU (like ComfyUI does)
    print("Moving pipeline to CUDA...")
    pipe.to("cuda")

    # Memory optimizations
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing(1)
        print("  Enabled attention slicing")

    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
        print("  Enabled VAE slicing")

    # Clear any cached memory
    torch.cuda.empty_cache()

    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
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

    Note: negative_prompt is ignored - Z-Image Turbo uses guidance_scale=0.0
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
        with torch.inference_mode():
            # Z-Image Turbo requires guidance_scale=0.0 (no CFG)
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
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        print(f"OOM ERROR: {e}")
        return {"error": f"Out of memory - try smaller image size (current: {width}x{height})"}

    except Exception as e:
        torch.cuda.empty_cache()
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
