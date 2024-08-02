import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLOv10
import requests
import tempfile
import os

# Download model
def download_model(url, save_as):
    resp = requests.get(url)
    with open(save_as, "wb") as f:
        f.write(resp.content)

URL = "https://github.com/Dimacat-exe/dimacat_demoapp1/releases/download/model3/catdt.pt"
SAVE_AS = "catdt.pt"
download_model(URL, SAVE_AS)
model = YOLOv10(SAVE_AS)

# Set up Streamlit
st.set_page_config(
    page_title='Find Cats',
    initial_sidebar_state='expanded',
)
st.title('Find Cats')
st.write('Upload your images and videos here')

# File uploaders for images and videos
uploaded_files = st.file_uploader(
    'Choose up to 200 images:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
)
uploaded_videos = st.file_uploader(
    'Choose up to 20 videos:', type=['mp4', 'avi'], accept_multiple_files=True
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
                results = model(img, conf=0.15)
                img_out = results[0].plot()
                img_out = cv2.cvtColor(np.array(img_out), cv2.COLOR_BGR2RGB)
                img_out = Image.fromarray(img_out)
            st.image(img_out, caption=f'Cats in image {i + 1}', use_column_width=True)

# Detect cats in uploaded videos
if uploaded_videos:
    for uploaded_video in uploaded_videos:
        st.write(f'Analyzing video: {uploaded_video.name}...')
        try:
            video_bytes = uploaded_video.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_bytes)
                temp_video_path = temp_video.name
            video = cv2.VideoCapture(temp_video_path)
            if not video.isOpened():
                st.error(f"Error: Could not open video {uploaded_video.name}")
                continue
            frame_count = 0
            fps = video.get(cv2.CAP_PROP_FPS) if video.get(cv2.CAP_PROP_FPS) > 0 else 30
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = f'output_video_{uploaded_video.name}'
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as out_temp_video:
                out_video_path = out_temp_video.name
            out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
            buffer_frames = []
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                buffer_frames.append(frame)
                if len(buffer_frames) >= 10:
                    for buffered_frame in buffer_frames:
                        with st.spinner(f'Detecting cats in frame {frame_count}...'):
                            results = model(buffered_frame, conf=0.15)
                            for info in results:
                                parameters = info.boxes
                                for box in parameters:
                                    x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
                                    confidence = box.conf[0].numpy().astype('int') * 100
                                    class_detected_name = "Cat"  # Assuming the class name is "Cat"

                                    cv2.rectangle(buffered_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(buffered_frame, f'{class_detected_name}', (x1 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                            frame_out = cv2.cvtColor(buffered_frame, cv2.COLOR_RGB2BGR)
                            out_video.write(frame_out)
                            frame_count += 1
                    buffer_frames = []
            for buffered_frame in buffer_frames:
                with st.spinner(f'Detecting cats in frame {frame_count}...'):
                    results = model(buffered_frame, conf=0.15)
                    for info in results:
                        parameters = info.boxes
                        for box in parameters:
                            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
                            confidence = box.conf[0].numpy().astype('int') * 100
                            class_detected_name = "Cat"  # Assuming the class name is "Cat"

                            cv2.rectangle(buffered_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(buffered_frame, f'{class_detected_name}', (x1 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                    frame_out = cv2.cvtColor(buffered_frame, cv2.COLOR_RGB2BGR)
                    out_video.write(frame_out)
                    frame_count += 1
            video.release()
            out_video.release()
            st.write(f'Done processing video: {uploaded_video.name}')
            st.video(out_video_path)
            os.remove(temp_video_path)
            os.remove(out_video_path)

        except Exception as e:
            st.write(f"Error processing video {uploaded_video.name}: {e}")
