import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO
import requests
import tempfile
import os

URL = "https://github.com/Dimacat-exe/dimacat_demoapp1/releases/download/model2/best.pt"
SAVE_AS = "best.pt"
resp = requests.get(URL)
with open(SAVE_AS, "wb") as f: 
    f.write(resp.content)
model = YOLO(SAVE_AS)
st.set_page_config(
    page_title='Find Cats',
    initial_sidebar_state='expanded',
)
st.title('Find Cats')
st.write('Upload your images and videos here')
uploaded_files = st.file_uploader(
    'Choose up to 50 image files:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
uploaded_video = st.file_uploader(
    'Choose a video file:', type=['mp4', 'avi'])
if uploaded_files:
    col1, col2 = st.columns(2)
    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        with col1:
            st.image(uploaded_file,
                     caption=f'Image {i+1}',
                     use_column_width=True)
        with col2:
            with st.spinner(f'Detecting cats {i+1}...'):
                results = model(img, conf=0.15)
                img_out = results[0].plot()
            st.image(img_out,
                     caption=f'Cats in image {i+1}',
                     use_column_width=True)
if uploaded_video:
    st.write('Analyzing the video...')
    try:
        video_bytes = uploaded_video.read()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        video = cv2.VideoCapture(tmp_path)
        frame_count = 0
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = 'output_video.mp4'
        out_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        while True:
            ret, frame = video.read()
            if not ret:
                break
            with st.spinner(f'Detecting cats in frame {frame_count}...'):
                results = model(frame, conf=0.15)
                frame_out = results[0].plot()
                frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                out_video.write(frame_out)
                frame_count += 1
        video.release()
        out_video.release()
        st.write('Done!')
        st.video(output_file)
        os.remove(tmp_path)
    except Exception as e:
        st.write(f"Error processing the video: {e}")
