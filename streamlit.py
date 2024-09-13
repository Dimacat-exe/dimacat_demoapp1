import streamlit as st
from PIL import Image
import numpy as np
import requests
from ultralytics import YOLO

def download_model(url, save_as):
    """Download the model from a URL."""
    response = requests.get(url)
    with open(save_as, "wb") as file:
        file.write(response.content)

MODEL_URL = "https://github.com/Dimacat-exe/dimacat_demoapp1/releases/download/v1-segmentation/yolov8-segmentation-v1.pt"
MODEL_PATH = "yolov8-segmentation.pt"
download_model(MODEL_URL, MODEL_PATH)
model = YOLO(MODEL_PATH)

# Set up the Streamlit app
st.set_page_config(page_title='Find Cats', initial_sidebar_state='expanded')
st.title('Find Cats')
st.write('Upload your images here')

# Image uploader
uploaded_files = st.file_uploader(
    'Choose up to 200 images:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
)
if uploaded_files:
    col1, col2 = st.columns(2)
    for index, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        with col1:
            st.image(image, caption=f'Image {index + 1}', use_column_width=True)
        with col2:
            with st.spinner(f'Detecting cats in image {index + 1}...'):
                results = model(image_array)
                if len(results) > 0 and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    mask_overlay = np.zeros_like(image_array)
                    for mask in masks:
                        resized_mask = np.zeros(image_array.shape[:2], dtype=bool)
                        resized_mask[:mask.shape[0], :mask.shape[1]] = mask.astype(bool)
                        mask_overlay[resized_mask] = [0, 255, 255]  # Cyan color
                    annotated_image = Image.fromarray(mask_overlay)
                else:
                    annotated_image = image
            st.image(annotated_image, caption=f'Cats in image {index + 1}', use_column_width=True)
