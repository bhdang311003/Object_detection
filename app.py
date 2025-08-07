import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("🔍 Phát hiện đối tượng với YOLOv8")

uploaded_file = st.file_uploader("📁 Tải ảnh lên", type=["jpg", "jpeg", "png"])

model = YOLO("yolov8n.pt") 

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Ảnh gốc", width=600)

    if st.button("🔍 Phát hiện đối tượng"):
        results = model(np.array(image))
        result_img = results[0].plot()
        st.image(result_img, caption="🎯 Kết quả phát hiện", width=600)
