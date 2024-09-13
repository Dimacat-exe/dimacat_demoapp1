import streamlit as st
from PIL import Image
import os
import subprocess
import glob
import requests

def download_model(url, save_as):
    """Download the model from a URL if it doesn't exist."""
    if not os.path.exists(save_as):
        response = requests.get(url)
        with open(save_as, "wb") as file:
            file.write(response.content)
MODEL_URL = "https://github.com/Dimacat-exe/dimacat_demoapp1/releases/download/v1-segmentation/yolov8-segmentation-v1.pt"
MODEL_PATH = "yolov8-segmentation.pt"
download_model(MODEL_URL, MODEL_PATH)

# Set up the Streamlit app
st.set_page_config(page_title='Find Cats', initial_sidebar_state='expanded')
st.title('Find Cats')
st.write('Upload your images here')

# Image uploader
uploaded_files = st.file_uploader(
    'Choose up to 200 images:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
)
if uploaded_files:
    # Save uploaded files to a temporary directory
    upload_dir = "uploaded_images"
    os.makedirs(upload_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    OUTPUT_PATH = 'runs/segment/predict'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    command = f"yolo task=segment mode=predict model={MODEL_PATH} conf=0.25 source={upload_dir} save=true"
    subprocess.run(command, shell=True)
    predicted_images = glob.glob(f'{OUTPUT_PATH}/*.jpg')[:3]
    for image_path in predicted_images:
        st.image(Image.open(image_path), caption=os.path.basename(image_path), use_column_width=True)
else:
    st.info("Please upload some images to start detecting cats.")
