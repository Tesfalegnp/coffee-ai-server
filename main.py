import io
import os
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

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

VERIFY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "coffee_leaf_strong_model.h5"
)

RUST_MODEL_PATH = os.path.join(
    BASE_DIR,
    "coffee_rust_strong_model.h5"
)

# =====================================================
# LOAD MODELS
# =====================================================
print("Loading AI models...")

verify_model = tf.keras.models.load_model(
    VERIFY_MODEL_PATH
)

rust_model = tf.keras.models.load_model(
    RUST_MODEL_PATH
)

print("CoffeeGuard AI Models Loaded Successfully")

# =====================================================
# IMAGE PREPROCESS
# MUST MATCH FLUTTER LOCAL LOGIC
# =====================================================
def preprocess_image(image_bytes):
    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Fix phone rotation using EXIF
    img = ImageOps.exif_transpose(img)

    # SAME resize style as mobile app
    img = img.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    img_array = np.array(
        img,
        dtype=np.float32
    )

    # SAME normalization as Flutter:
    # (pixel / 127.5) - 1
    img_array = (img_array / 127.5) - 1.0

    # shape => [1,224,224,3]
    img_array = np.expand_dims(
        img_array,
        axis=0
    ).astype(np.float32)

    return img_array


# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {
        "status": "CoffeeGuard Server is Running"
    }


# =====================================================
# HEALTH
# =====================================================
@app.get("/health")
def health():
    return {
        "success": True,
        "server": "online"
    }


# =====================================================
# PREDICT
# =====================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()

        input_tensor = preprocess_image(
            contents
        )

        # ===============================================
        # STAGE 1 : COFFEE LEAF VERIFY
        # LOCAL APP RULE:
        # verifyProb >= 0.5 => NOT coffee leaf
        # verifyProb < 0.5  => Coffee leaf
        # ===============================================
        verify_pred = verify_model.predict(
            input_tensor,
            verbose=0
        )

        verify_prob = float(
            verify_pred[0][0]
        )

        if verify_prob >= 0.5:
            return {
                "success": False,
                "message": "Please provide a clear image of a coffee leaf.",
                "confidence": round(
                    verify_prob * 100,
                    2
                ),
                "rawVerify": round(
                    verify_prob,
                    6
                )
            }

        # ===============================================
        # STAGE 2 : RUST DETECTION
        # rustProb > 0.5 => Rust
        # else => Healthy
        # ===============================================
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
                (1.0 - verify_prob) * 100,
                2
            ),
            "rawVerify": round(
                verify_prob,
                6
            ),
            "rawRust": round(
                rust_prob,
                6
            )
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }