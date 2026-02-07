'''
https://github.com/deepinsight/insightface/tree/master/examples/in_swapper

Download inswapper_128.onnx - > https://huggingface.co/ashleykleynhans/a1111-models/resolve/main/insightface/inswapper_128.onnx
pip install -U insightface
'''

import cv2

import insightface

from insightface.app import FaceAnalysis

# Paths
SOURCE_IMG = "source.jpeg"  # face want to swap FROM
TARGET_IMG = "target.jpeg"  # face want to swap TO
OUTPUT_IMG = "output.jpg"
MODEL_PATH = "models/inswapper_128.onnx"

# Load face analysis model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Load swapper model
swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False)

# Read images
img_source = cv2.imread(SOURCE_IMG)
img_target = cv2.imread(TARGET_IMG)

# Detect faces
source_faces = app.get(img_source)
target_faces = app.get(img_target)

if len(source_faces) == 0:
    raise Exception("No face detected in source image")

if len(target_faces) == 0:
    raise Exception("No face detected in target image")

# Use the first detected face
source_face = source_faces[0]

# Swap face(s)
result = img_target.copy()
for face in target_faces:
    result = swapper.get(result, face, source_face, paste_back=True)

cv2.imwrite(OUTPUT_IMG, result)
print("Face swap completed:", OUTPUT_IMG)
