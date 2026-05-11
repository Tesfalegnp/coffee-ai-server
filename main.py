import io
import os
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ==========================================
# APP CONFIG
# ==========================================
app = FastAPI(title="CoffeeGuard AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MODEL PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VERIFY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "coffee_leaf_strong_model.h5"
)

RUST_MODEL_PATH = os.path.join(
    BASE_DIR,
    "coffee_rust_strong_model.h5"
)

# ==========================================
# LOAD MODELS
# ==========================================
print("Loading AI models...")

verify_model = tf.keras.models.load_model(
    VERIFY_MODEL_PATH
)

rust_model = tf.keras.models.load_model(
    RUST_MODEL_PATH
)

print("CoffeeGuard AI Models Loaded Successfully")

# ==========================================
# IMAGE PREPROCESS
# ==========================================
def preprocess_image(image_bytes):
    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    img = img.resize((224, 224))

    img_array = np.array(
        img,
        dtype=np.float32
    )

    # Normalize to [-1,1]
    img_array = (img_array / 127.5) - 1.0

    # Shape => [1,224,224,3]
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

# ==========================================
# ROOT ROUTE
# ==========================================
@app.get("/")
def root():
    return {
        "status": "CoffeeGuard Server is Running"
    }

# ==========================================
# HEALTH CHECK
# ==========================================
@app.get("/health")
def health():
    return {
        "success": True,
        "server": "online"
    }

# ==========================================
# PREDICT ROUTE
# ==========================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        # Read image bytes
        contents = await file.read()

        # Preprocess
        input_tensor = preprocess_image(
            contents
        )

        # ==================================
        # STAGE 1:
        # VERIFY COFFEE LEAF
        # ==================================
        verify_pred = verify_model.predict(
            input_tensor,
            verbose=0
        )

        verify_prob = float(
            verify_pred[0][0]
        )

        # FIXED LOGIC:
        # If below threshold = not coffee leaf
        if verify_prob < 0.5:
            return {
                "success": False,
                "message": "Please provide a clear image of a coffee leaf.",
                "confidence": round(
                    (1 - verify_prob) * 100,
                    2
                )
            }

        # ==================================
        # STAGE 2:
        # RUST DETECTION
        # ==================================
        rust_pred = rust_model.predict(
            input_tensor,
            verbose=0
        )

        rust_prob = float(
            rust_pred[0][0]
        )

        if rust_prob > 0.5:
            disease = "Rust Disease"
            confidence = rust_prob
        else:
            disease = "Healthy Leaf"
            confidence = 1.0 - rust_prob

        return {
            "success": True,
            "disease": disease,
            "confidence": round(
                confidence * 100,
                2
            ),
            "leafConfidence": round(
                verify_prob * 100,
                2
            )
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }