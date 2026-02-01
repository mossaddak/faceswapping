'''
cd /home/mossaddak/Projects/facewaps/charecter && source venv/bin/activate && rm -f run.log && nohup python index.py > run.log 2>&1 &
'''


import torch

from diffusers import StableDiffusionXLPipeline

# ---------------- CONFIG ----------------
# Auto-select device; fall back to CPU if CUDA unavailable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 on GPU for memory saving, float32 on CPU
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ----------------------------------------

# Load SDXL (simpler approach without IP-Adapter)
print(f"Loading StableDiffusionXL on {DEVICE}...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=TORCH_DTYPE
).to(DEVICE)

# enable_xformers is optional and may not be installed; ignore if not available
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

print("Pipeline loaded successfully!")

# Prompt
prompt = """
3D cartoon boy, pixar style, cute child,
sitting on an airplane seat,
soft lighting, cinematic,
high quality, smooth skin,
big expressive eyes
"""

negative_prompt = """
realistic, photo, ugly, deformed,
extra fingers, bad anatomy,
low quality
"""

# Generate using SDXL directly (no IP-Adapter)
print("Generating image...")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=768,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("output.png")
print("✅ Image saved as output.png")
