import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# Title
st.title("Brain MRI Metastasis Segmentation")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    
    # Convert the image to grayscale and resize
    image = np.array(image.convert('L'))
    image = cv2.resize(image, (256, 256))
    
    # Send the image to backend for prediction
    response = requests.post("http://localhost:8000/predict/", files={"file": uploaded_file.getvalue()})
    result = response.json()
    
    st.write("Metastasis Segmentation Prediction: ", result["prediction"])
