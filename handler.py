"""
Z-Image Turbo RunPod Serverless Handler

Loads Z-Image Turbo model at container start, then handles
inference requests returning base64-encoded images.
"""

import base64
import io
import os
import torch
import runpod
from diffusers import ZImagePipeline

# Model path (set during Docker build)
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/zimage")

# Default generation settings
DEFAULTS = {
    "width": 512,
    "height": 512,
    "steps": 8,  # Actually 8 DiT forwards (steps=9 in API)
    "negative_prompt": "blurry ugly bad",
}


def load_pipeline():
    """Load the Z-Image Turbo pipeline."""
    print("Loading Z-Image Turbo pipeline...")

    pipe = ZImagePipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")

    # Optional: compile for faster inference (uncomment if torch.compile works)
    # pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

    print("Pipeline loaded successfully!")
    return pipe


# Load pipeline at container start (stays warm)
pipe = load_pipeline()


def handler(job):
    """
    Handle image generation request.

    Input:
        prompt (str): Image description
        negative_prompt (str, optional): Negative prompt (default: "blurry ugly bad")
        width (int, optional): Image width (default: 512)
        height (int, optional): Image height (default: 512)
        steps (int, optional): Inference steps (default: 8)
        seed (int, optional): Random seed (default: random)

    Output:
        image (str): Base64-encoded PNG image
    """
    job_input = job["input"]

    # Extract parameters
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required parameter: prompt"}

    negative_prompt = job_input.get("negative_prompt", DEFAULTS["negative_prompt"])
    width = job_input.get("width", DEFAULTS["width"])
    height = job_input.get("height", DEFAULTS["height"])
    steps = job_input.get("steps", DEFAULTS["steps"])
    seed = job_input.get("seed")

    # Create generator with seed
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = None

    try:
        # Generate image
        # Note: Z-Image Turbo uses guidance_scale=0.0
        # Note: steps+1 because "9 steps = 8 DiT forwards"
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps + 1,
            guidance_scale=0.0,
            generator=generator,
        )

        image = result.images[0]

        # Convert to base64 PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "image": image_b64,
            "width": width,
            "height": height,
            "seed": seed,
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
