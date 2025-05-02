import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import time

st.set_page_config(layout="wide")
model = load_model("emotion.h5",compile=False)
class_labels = ['angry', 'happy', 'neutral', 'sad']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
output_folder = "camera_detects"
os.makedirs(output_folder, exist_ok=True)
st.markdown("<h1 style='text-align: center;margin-top:-2rem'>Real-Time Emotion Detection </h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
#CAMERA SECTION
with col1:
    st.markdown(
        "<div style='padding-left: 240px;margin-top:1.5rem;'><h3>Real-Time Camera</h3></div>",
        unsafe_allow_html=True
    )
    cam_col1, cam_col2 = st.columns([1, 1])
    with cam_col1:
        start_button = st.button("Start Camera", use_container_width=True)
    with cam_col2:
        stop_button = st.button("Stop Camera", use_container_width=True)

    FRAME_WINDOW = st.image([])

    if start_button:
        cap = cv2.VideoCapture(0)
        run = True

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_normalized = roi_resized / 255.0
                roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

                prediction = model.predict(roi_reshaped, verbose=0)
                confidence = np.max(prediction)
                predicted_emotion = class_labels[np.argmax(prediction)]
                emotion_label = f"{predicted_emotion} ({confidence*100:.2f}%)"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

                #SAVE IMAGE
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{output_folder}/{predicted_emotion}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            if stop_button:
                run = False

        cap.release()
        st.success("Camera stopped.")
# IMAGE UPLOAD SECTION
with col2:
    st.markdown(
        "<div style='padding-left: 280px;margin-top:1.5rem;'><h3>Upload Image</h3></div>",
        unsafe_allow_html=True
    )
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png", "avif", "webp"], label_visibility="collapsed")

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert('RGB')
        img_array = np.array(img)
        resized_original = cv2.resize(img_array, (200, 200)) 

        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

        col2a, col2b = st.columns(2)

        with col2a:
            st.markdown("**Original Image**")
            st.image(resized_original, use_container_width=True)

        if len(faces) == 0:
            st.warning("No face detected in the uploaded image.")
        else:
            pred_img = img_array.copy()
            for (x, y, w, h) in faces:
                roi_gray = gray_img[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_normalized = roi_resized / 255.0
                roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

                prediction = model.predict(roi_reshaped, verbose=0)
                confidence = np.max(prediction)
                emotion_label = class_labels[np.argmax(prediction)]
                label = f"{emotion_label} ({confidence*100:.1f}%)"
                cv2.rectangle(pred_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(pred_img, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
                cv2.putText(pred_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            pred_img_resized = cv2.resize(pred_img, (200, 200))
            with col2b:
                st.markdown("**Predicted Image**")
                st.image(pred_img_resized, use_container_width=True)

