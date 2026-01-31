import os
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

# ---------------- CONFIG ----------------
IMAGE_PATH = "me.jpeg"
IP_ADAPTER_PATH = "models/ip-adapter-faceid_sdxl.bin"
# Auto-select device; fall back to CPU if CUDA unavailable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 on GPU for memory saving, float32 on CPU
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ----------------------------------------

# Load InsightFace
app = FaceAnalysis(name="buffalo_l")
# insightface ctx_id: 0 for GPU, -1 for CPU
ctx_id = 0 if DEVICE == "cuda" and torch.cuda.is_available() else -1
app.prepare(ctx_id=ctx_id, det_size=(640, 640))

img = cv2.imread(IMAGE_PATH)
faces = app.get(img)

if len(faces) == 0:
    raise Exception("No face detected")

face_embedding = faces[0].embedding

# Load SDXL
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=TORCH_DTYPE
).to(DEVICE)

# enable_xformers is optional and may not be installed; ignore if not available
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

# Load IP-Adapter
if not os.path.exists(IP_ADAPTER_PATH):
    raise FileNotFoundError(f"IP-Adapter model not found at {IP_ADAPTER_PATH}.\nPlease download or set the correct path.")

ip_adapter = IPAdapterFaceID(
    pipe,
    IP_ADAPTER_PATH,
    device=DEVICE
)

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

# Generate
image = ip_adapter.generate(
    face_embedding=face_embedding,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=768,
    num_inference_steps=30,
    guidance_scale=7.5
)

image.save("output.png")
print("✅ Image saved as output.png")
