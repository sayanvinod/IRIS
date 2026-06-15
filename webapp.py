from fastapi import FastAPI, HTTPException, File, UploadFile
import numpy as np
from ultralytics import YOLO
from PIL import Image as PILImage
import uvicorn
import io
from fastapi.middleware.cors import CORSMiddleware

origins = ['*']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

model_path = r"C:\Users\sayan\OneDrive\Cursor-Files\runs\classify\train4\weights\best.pt"
model = YOLO(model_path)

@app.post('/classify')
async def classify(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    # Open image
    image = PILImage.open(io.BytesIO(contents))
    # Run model on image
    output = model(image)[0]
    pred = output.probs.data.tolist()
    pred_index = int(np.argmax(pred))
    return {
        'prediction': output.names[pred_index],
        'score': pred[pred_index],
        'filename': file.filename
    }
