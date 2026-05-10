import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
# Change this line
import tensorflow as tf 

app = FastAPI(title="CoffeeGuard AI Server")

# Update these lines to use the new import
verify_interpreter = tf.lite.Interpreter(model_path="coffee_leaf_verification.tflite")
verify_interpreter.allocate_tensors()

rust_interpreter = tf.lite.Interpreter(model_path="coffee_rust_model.tflite")
rust_interpreter.allocate_tensors()

# ... rest of the code remains the same

def preprocess_image(image_bytes):
    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize to 224x224 (Matches your Flutter code)
    img = img.resize((224, 224))
    # Convert to numpy array and normalize to [-1, 1]
    input_data = np.array(img, dtype=np.float32)
    input_data = (input_data / 127.5) - 1.0
    # Add batch dimension [1, 224, 224, 3]
    return np.expand_dims(input_data, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)

        # Stage 1: Verification (Is it a coffee leaf?)
        v_input_details = verify_interpreter.get_input_details()
        v_output_details = verify_interpreter.get_output_details()
        
        verify_interpreter.set_tensor(v_input_details[0]['index'], input_tensor)
        verify_interpreter.invoke()
        verify_prob = verify_interpreter.get_tensor(v_output_details[0]['index'])[0][0]

        if verify_prob >= 0.5:
            return {
                "success": False,
                "message": "Please provide a clear image of a coffee leaf.",
                "confidence": round(float(verify_prob) * 100, 2)
            }

        # Stage 2: Rust Detection
        r_input_details = rust_interpreter.get_input_details()
        r_output_details = rust_interpreter.get_output_details()

        rust_interpreter.set_tensor(r_input_details[0]['index'], input_tensor)
        rust_interpreter.invoke()
        rust_prob = rust_interpreter.get_tensor(r_output_details[0]['index'])[0][0]

        if rust_prob > 0.5:
            disease = "Rust Disease"
            conf = rust_prob
        else:
            disease = "Healthy Leaf"
            conf = 1.0 - rust_prob

        return {
            "success": True,
            "disease": disease,
            "confidence": round(float(conf) * 100, 2)
        }

    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/")
def health_check():
    return {"status": "CoffeeGuard Server is Running"}