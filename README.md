# Emotion Detection Using Webcam And Image Upload 

I developed a Real-Time Facial Emotion Detection System using Deep Learning and Computer Vision. The goal of this project is to recognize human emotions such as happiness, sadness, anger, and neutrality by analyzing 
facial expressions from a live camera feed or an uploaded image. First, I trained the model on the FER-2013 dataset,which originally contains seven emotion classes. I removed three of these classes and trained the model using only four classes: Angry, Sad, Neutral, and Happy. The trained model was then saved in an .h5 file (named emotion.h5) for emotion detection. This model was trained on a dataset of facial images labeled with different emotions. 
Captured images are stored in the `camera_detects` folder.

## How to Run

## Clone the Repository
```bash
git clone https://github.com/MananAli05/Emotion-Detection-CNN.git
cd Emotion-Detection-CNN

1. Install requirements:
```bash
pip install -r requirements.txt

streamlit run app.py

![Happy Emotion](assets/happy.jpg)
