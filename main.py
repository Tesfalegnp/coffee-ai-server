import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tflite_runtime.interpreter as tflite

# =========================
# APP INIT
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VERIFY_MODEL_PATH = os.path.join(BASE_DIR, "coffee_leaf_verification.tflite")
RUST_MODEL_PATH = os.path.join(BASE_DIR, "coffee_rust_model.tflite")

# =========================
# LOAD TFLITE MODELS
# =========================
verify_interpreter = tflite.Interpreter(model_path=VERIFY_MODEL_PATH)
rust_interpreter = tflite.Interpreter(model_path=RUST_MODEL_PATH)

verify_interpreter.allocate_tensors()
rust_interpreter.allocate_tensors()

verify_input = verify_interpreter.get_input_details()
verify_output = verify_interpreter.get_output_details()

rust_input = rust_interpreter.get_input_details()
rust_output = rust_interpreter.get_output_details()

print("☕ Coffee AI Models Loaded Successfully")

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"status": "Coffee AI Server Running"}

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)

    # normalize [-1, 1]
    image = (image / 127.5) - 1

    image = np.expand_dims(image, axis=0)  # (1,224,224,3)
    return image

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # read image
    image = Image.open(file.file).convert("RGB")

    input_data = preprocess_image(image)

    # =========================
    # STEP 1: VERIFY COFFEE LEAF
    # =========================
    verify_interpreter.set_tensor(
        verify_input[0]["index"],
        input_data
    )
    verify_interpreter.invoke()

    verify_prob = verify_interpreter.get_tensor(
        verify_output[0]["index"]
    )[0][0]

    # NOT a coffee leaf
    if verify_prob >= 0.5:
        return {
            "success": False,
            "message": "Not a coffee leaf. Please upload a valid coffee leaf image.",
            "confidence": round(float(verify_prob * 100), 2)
        }

    # =========================
    # STEP 2: RUST DETECTION
    # =========================
    rust_interpreter.set_tensor(
        rust_input[0]["index"],
        input_data
    )
    rust_interpreter.invoke()

    rust_prob = rust_interpreter.get_tensor(
        rust_output[0]["index"]
    )[0][0]

    # =========================
    # RESULT LOGIC
    # =========================
    if rust_prob > 0.5:
        disease = "Rust Disease"
        confidence = rust_prob * 100
    else:
        disease = "Healthy Leaf"
        confidence = (1 - rust_prob) * 100

    return {
        "success": True,
        "disease": disease,
        "confidence": round(float(confidence), 2)
    }