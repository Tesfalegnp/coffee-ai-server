import io
import os
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# =====================================================
# APP CONFIG
# =====================================================
app = FastAPI(title="CoffeeGuard AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODEL PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VERIFY_MODEL_PATH = os.path.join(BASE_DIR, "coffee_leaf_strong_model.h5")
RUST_MODEL_PATH = os.path.join(BASE_DIR, "coffee_rust_strong_model.h5")

# =====================================================
# LOAD MODELS
# =====================================================
print("Loading AI models...")

verify_model = tf.keras.models.load_model(VERIFY_MODEL_PATH)
rust_model = tf.keras.models.load_model(RUST_MODEL_PATH)

print("CoffeeGuard AI Models Loaded Successfully")

# =====================================================
# IMAGE PREPROCESS (MATCH LOCAL EXACTLY)
# =====================================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)

    # SAME NORMALIZATION AS FLUTTER
    img_array = (img_array / 127.5) - 1.0

    return np.expand_dims(img_array, axis=0)

# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {"status": "CoffeeGuard Server is Running"}

# =====================================================
# HEALTH
# =====================================================
@app.get("/health")
def health():
    return {"success": True, "server": "online"}

# =====================================================
# PREDICT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        # =================================================
        # STAGE 1: COFFEE LEAF DETECTION
        # FIXED MEANING: 1 = coffee leaf probability
        # =================================================
        verify_pred = verify_model.predict(input_tensor, verbose=0)

        verify_score = float(verify_pred[0][0])  # IS COFFEE LEAF

        # ❌ NOT COFFEE LEAF
        if verify_score < 0.5:
            return {
                "success": False,
                "message": "Please provide a clear image of a coffee leaf.",
                "leafConfidence": round((1 - verify_score) * 100, 2)
            }

        # =================================================
        # STAGE 2: DISEASE DETECTION
        # =================================================
        rust_pred = rust_model.predict(input_tensor, verbose=0)

        rust_score = float(rust_pred[0][0])

        if rust_score > 0.5:
            disease = "Rust Disease"
            disease_conf = rust_score
        else:
            disease = "Healthy Leaf"
            disease_conf = 1.0 - rust_score

        return {
            "success": True,
            "disease": disease,
            "confidence": round(disease_conf * 100, 2),
            "leafConfidence": round(verify_score * 100, 2)
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }