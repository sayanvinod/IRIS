# Bring in lightweight dependencies
from fastapi import FastAPI
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

def load_model():
    model_path = r"C:\Users\sayan\OneDrive\Cursor-Files\runs\classify\train4\weights\best.pt"
    model = YOLO(model_path)
    return model

model = load_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/get_predictions')
async def get_predictions(file:UploadFile):
    try:
        print("Received file:", file.filename)
        image = await file.read()
        image = Image.open(io.BytesIO(image))
        print("Image loaded")
        result = model(image)
        print("Model inference done")
        names = result[0].names
        probability = result[0].probs.data.numpy()
        prediction = np.argmax(probability)
        response = {
            'Prediction': names[prediction],
            'Confidence': float(probability[prediction])
        }
        print("Response:", response)
        return response
    except Exception as e:
        print("Error during prediction:", e)
        return {"error": str(e)}