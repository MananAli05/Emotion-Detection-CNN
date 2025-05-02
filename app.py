import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model

model = load_model("emotion.h5")
class_labels = ['angry', 'happy', 'neutral', 'sad']
#Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
st.set_page_config(page_title="Facial Emotion Detector", layout="centered")
st.title("Facial Emotion Detection App")

# Folder to save captured images
captured_dir = "captured_images"
os.makedirs(captured_dir, exist_ok=True)

def detect_emotion_from_image(image_np):
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = image_gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
        prediction = model.predict(roi_reshaped)
        emotion = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_np, f"{emotion} ({confidence:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image_np

col1, col2 = st.columns(2)

with col1:
    if st.button("Open Webcam", use_container_width=True):
        st.info("Opening webcam... Please look at the camera. Auto-capturing in 3 seconds...")

        cap = cv2.VideoCapture(0)
        time.sleep(3)  # Wait for 3 seconds

        ret, frame = cap.read()
        cap.release()

        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save captured image
            filename = f"captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(captured_dir, filename)
            cv2.imwrite(filepath, frame)
            st.success(f"Image saved as {filename}")

            st.image(rgb_frame, caption="Captured Image", use_container_width=True)


            result_img = detect_emotion_from_image(rgb_frame)
            st.image(result_img, caption="Emotion Prediction", use_container_width=True)
        else:
            st.error(" Failed to capture image from webcam.")

with col2:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)
        result_img = detect_emotion_from_image(image_np)
        st.image(result_img, caption="Emotion Prediction", use_container_width=True)
