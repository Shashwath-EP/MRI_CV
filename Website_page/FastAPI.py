# backend.py
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# Load models
nested_unet = load_model('./models/model_weights/nested_unet.h5')
attention_unet = load_model('./models/model_weights/attention_unet.h5')

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = img.reshape(1, 256, 256, 1)
    return img

@app.post("/predict/")
async def predict(image: UploadFile = File(...), model_name: str = "nested_unet"):
    image_bytes = await image.read()
    img = preprocess_image(image_bytes)

    if model_name == "nested_unet":
        prediction = nested_unet.predict(img)
    else:
        prediction = attention_unet.predict(img)

    pred_mask = (prediction[0] > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(pred_mask.squeeze(), mode='L')

    img_byte_arr = io.BytesIO()
    mask_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return {"prediction": img_byte_arr}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
