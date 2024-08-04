import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLOv10
import requests
import some_visualization_library as sv  # Replace with the actual library you're using for visualization
import random

# Function to download the model
def download_model(url, save_as):
    resp = requests.get(url)
    with open(save_as, "wb") as f:
        f.write(resp.content)

# Model URL and save path
URL = "https://github.com/Dimacat-exe/dimacat_demoapp1/releases/download/model3/catdt.pt"
SAVE_AS = "catdt.pt"
download_model(URL, SAVE_AS)
model = YOLOv10(SAVE_AS)

# Set up Streamlit page
st.set_page_config(
    page_title='Find Cats',
    initial_sidebar_state='expanded',
)
st.title('Find Cats')
st.write('Upload your images here')

# File uploader for images
uploaded_files = st.file_uploader(
    'Choose up to 200 images:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
)

# Detect cats in uploaded images
if uploaded_files:
    col1, col2 = st.columns(2)
    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(img, caption=f'Image {i + 1}', use_column_width=True)
        with col2:
            with st.spinner(f'Detecting cats in image {i + 1}...'):
                results = model(img, conf=0.15)[0]
                detections = sv.Detections.from_ultralytics(results)
                bounding_box_annotator = sv.BoundingBoxAnnotator()
                label_annotator = sv.LabelAnnotator()
                img_out = bounding_box_annotator.annotate(
                    scene=np.array(img), detections=detections)
                img_out = label_annotator.annotate(
                    scene=img_out, detections=detections)
                img_out = Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
            st.image(img_out, caption=f'Cats in image {i + 1}', use_column_width=True)
