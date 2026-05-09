from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()

    # TODO: Replace with real model
    return {
        "success": True,
        "disease": "Coffee Leaf Rust",
        "diseaseConfidence": 0.94,
        "recommendation": "Use approved fungicide."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)