import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI()

# ===============================
# CORS (ALLOW FLUTTER APP)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# LOAD TFLITE MODELS
# ===============================
verify_interpreter = tflite.Interpreter(
    model_path="coffee_leaf_verification.tflite"
)

rust_interpreter = tflite.Interpreter(
    model_path="coffee_rust_model.tflite"
)

verify_interpreter.allocate_tensors()
rust_interpreter.allocate_tensors()

verify_input = verify_interpreter.get_input_details()
verify_output = verify_interpreter.get_output_details()

rust_input = rust_interpreter.get_input_details()
rust_output = rust_interpreter.get_output_details()

print("✅ Coffee AI Models Loaded")


# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)

    # normalize [-1, 1]
    image = (image / 127.5) - 1
    image = np.expand_dims(image, axis=0)

    return image


# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def root():
    return {"status": "Coffee AI Server Running 🚀"}


# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")
    input_data = preprocess(image)

    # =========================
    # STAGE 1: VERIFY COFFEE LEAF
    # =========================
    verify_interpreter.set_tensor(
        verify_input[0]['index'],
        input_data
    )
    verify_interpreter.invoke()

    verify_prob = verify_interpreter.get_tensor(
        verify_output[0]['index']
    )[0][0]

    if verify_prob >= 0.5:
        return {
            "success": False,
            "message": "Not a coffee leaf. Please upload a clear coffee leaf image."
        }

    # =========================
    # STAGE 2: DISEASE DETECTION
    # =========================
    rust_interpreter.set_tensor(
        rust_input[0]['index'],
        input_data
    )
    rust_interpreter.invoke()

    rust_prob = rust_interpreter.get_tensor(
        rust_output[0]['index']
    )[0][0]

    if rust_prob > 0.5:
        disease = "Rust Disease"
        confidence = float(rust_prob * 100)
    else:
        disease = "Healthy Leaf"
        confidence = float((1 - rust_prob) * 100)

    return {
        "success": True,
        "disease": disease,
        "confidence": round(confidence, 2)
    }