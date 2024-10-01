from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load the best performing model
model = load_model('./models/model_weights/best_model.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256)) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
