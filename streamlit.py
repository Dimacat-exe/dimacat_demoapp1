import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLOv10
import requests

# Set up model, YOLO, Streamlit
URL = "https://github.com/Dimacat-exe/dimacat_demoapp1/releases/download/model3/catdt.pt"
SAVE_AS = "catdt.pt"
resp = requests.get(URL)
with open(SAVE_AS, "wb") as f:
    f.write(resp.content)
model = YOLOv10(SAVE_AS)
st.set_page_config(
    page_title='Find Cats',
    initial_sidebar_state='expanded',
)
st.title('Find Cats')
st.write('Upload your images and videos here')

# Upload pictures or videos
uploaded_files = st.file_uploader(
    'Choose max 200 pictures:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
)
uploaded_videos = st.file_uploader(
    'Choose max 20 videos:', type=['mp4', 'avi'], accept_multiple_files=True
)

# Detect cat in pictures
if uploaded_files:
    col1, col2 = st.columns(2)
    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file).convert("RGB") 
        with col1:
            st.image(img, caption=f'Image {i + 1}', use_column_width=True)
        with col2:
            with st.spinner(f'Detecting cats in image {i + 1}...'):
                results = model(img, conf=0.15)  # Detect cats
                img_out = results[0].plot()  # Plot results
                img_out = Image.fromarray(img_out)  # Convert back to PIL Image
            st.image(img_out, caption=f'Cats in image {i + 1}', use_column_width=True)

# Detect cat in videos
if uploaded_videos:
    for uploaded_video in uploaded_videos:
        st.write(f'Analyzing video: {uploaded_video.name}...')
        try:
            video_bytes = uploaded_video.read()
            with open('temp_video.mp4', 'wb') as temp_video:
                temp_video.write(video_bytes)
            video = cv2.VideoCapture('temp_video.mp4')
            frame_count = 0
            fps = video.get(cv2.CAP_PROP_FPS) if video.get(cv2.CAP_PROP_FPS) > 0 else 30
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = f'output_video_{uploaded_video.name}.mp4'
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
            st.write('Done processing video: {}'.format(uploaded_video.name))
            st.video(output_file)  
        except Exception as e:
            st.write(f"Error processing video {uploaded_video.name}: {e}")
