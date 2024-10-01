# Brain Metastasis Segmentation Using Nested U-Net and Attention U-Net

## Table of Contents
1. [Project Description](#project-description)
2. [Architectures](#architectures)
    - [Nested U-Net (U-Net++)](#nested-u-net-u-net)
    - [Attention U-Net](#attention-u-net)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Web Application](#web-application)
6. [Installation](#Installation)
7. [Usage](#usage)
8. [Challenges and Solutions](#challenges-and-solutions)
9. [Future Work](#future-work)
10. [License](#license)

---

## Project Description
This project uses advanced deep-learning models to segment brain metastases from MRI images. Brain metastasis detection is crucial for early treatment but presents challenges such as small lesion sizes, low contrast, and imbalanced data. We implement **Nested U-Net (U-Net++)** and **Attention U-Net** for precise segmentation, using **CLAHE** preprocessing and augmentation to improve performance. The models are evaluated using the **DICE Score**, and a web application is developed using **FastAPI** and **Streamlit** to allow real-time metastasis segmentation for clinical use.
brain_mri_segmentation/
│
├── data/
│   └── images/ (Place your dataset here)
│
├── models/
│   └── model_weights/ (Saved model weights here)
│
├── app.py (Streamlit application)
├── backend.py (FastAPI backend)
├── train.py (Model training script)
├── utils.py (Data preprocessing and utility functions)
├── README.md
└── requirements.txt


## Architectures

### Nested U-Net (U-Net++)
Nested U-Net introduces dense skip connections to improve feature propagation and reduce the semantic gap between the encoder and decoder. This architecture is particularly beneficial for detecting small, irregularly shaped metastases by refining segmentation at multiple scales.

### Attention U-Net
Attention U-Net integrates an attention mechanism to focus on the most relevant regions of the image, enhancing the model’s ability to segment metastases while ignoring irrelevant areas. This mechanism is crucial for handling the low-contrast, small metastases common in brain MRIs.

## Data Preprocessing
To enhance the quality of the input data, we applied **CLAHE (Contrast Limited Adaptive Histogram Equalization)**, improving the visibility of brain metastases in MRI scans. This step is followed by dataset **normalization** and **augmentation** to manage the challenges of imbalanced data and varying lesion sizes.

Steps include:
- **CLAHE Enhancement**: Increases the contrast of MRI images to highlight metastasis regions.
- **Normalization**: Ensures uniform pixel intensity distribution.
- **Data Augmentation**: Includes random rotation, flipping, and scaling to improve model generalization.

## Model Training and Evaluation
Both **Nested U-Net** and **Attention U-Net** were trained on a preprocessed dataset of brain MRI images with metastasis annotations. The models are evaluated using the **DICE Score**, a popular metric for segmentation accuracy. Training and evaluation steps are provided below:

1. **Model Training**: The models are trained for 50 epochs using a combination of cross-entropy loss and DICE loss to optimize segmentation accuracy.
2. **Evaluation**: After training, the models are evaluated using the DICE Score to measure the overlap between the predicted segmentation and ground truth.

## Web Application
A user-friendly web application is developed to deploy the best-performing model. Users can upload brain MRI images and get real-time segmentation results for brain metastasis.

- **Backend**: Powered by **FastAPI**, which serves the segmentation model as an API.
- **Frontend**: Built with **Streamlit**, providing an intuitive interface for uploading images and displaying segmentation results.

## Installation

### Prerequisites
- Python 3.8+
- Key libraries: TensorFlow/PyTorch, FastAPI, Streamlit, OpenCV, scikit-image, and others (listed in `requirements.txt`).

## Usage
- Upload a brain MRI image through the **Streamlit UI**.
- The segmentation results will be displayed, indicating the detected metastasis regions.
- DICE score evaluation is provided for each segmented image to measure model accuracy.

## Challenges and Solutions

### Challenges
- **Small Lesions**: Metastases are often small, making them difficult to detect.
- **Imbalanced Data**: The majority of MRI pixels represent healthy brain tissue, creating an imbalance.
- **Low Contrast**: Brain metastases may not be easily distinguishable from surrounding tissue.

### Solutions
- **Data Augmentation**: Random transformations like cropping and rotating to enhance model generalization.
- **CLAHE**: Used to increase image contrast, improving visibility of metastases.
- **Advanced Architectures**: Both Nested U-Net and Attention U-Net capture fine details and improve segmentation in difficult cases.

## Future Work
- Extend the model to handle multi-modal MRI data (e.g., T1, T2, FLAIR).
- Explore transformer-based models or other state-of-the-art segmentation architectures for further performance improvements.
- Optimize model inference time for real-time clinical use.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
