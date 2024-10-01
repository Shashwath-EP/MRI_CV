import os
import cv2
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split

# Apply CLAHE to enhance contrast
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Function to load and preprocess images from all subfolders
def load_and_preprocess_images(root_folder):
    images = []
    masks = []
    
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".tif") and "_mask" not in file:  # Exclude mask files initially
                img_path = os.path.join(subdir, file)
                img = imread(img_path)
                
                if len(img.shape) == 3:  # Convert color images to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                img = apply_clahe(img)
                
                # Find corresponding mask file
                mask_file = file.replace('.tif', '_mask.tif')
                mask_path = os.path.join(subdir, mask_file)
                
                if os.path.exists(mask_path):
                    mask = imread(mask_path)
                    
                    if len(mask.shape) == 3:  # Convert mask to grayscale
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    
                    images.append(img)
                    masks.append(mask)
    
    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0
    
    return images, masks

# Split the data into training and testing sets
def split_data(images, masks, test_size=0.2):
    return train_test_split(images, masks, test_size=test_size, random_state=42)
