import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your custom model
MODEL_PATH = 'model_optimal.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the image/frame for your model
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (48, 48))  # Resize to the input size your model expects
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize if required by your model
    return image

# Function to detect emotions
def detect_emotions(frame):
    processed_image = preprocess_image(frame)
    predictions = model.predict(processed_image)
    emotion = np.argmax(predictions)  # Assuming your model outputs a softmax vector
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # Modify based on your model's output
    return emotion_labels[emotion]

# Streamlit app
st.title("Emotion Detection App")

# Upload Image Section
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    frame = np.array(image.convert('RGB'))
    emotion = detect_emotions(frame)
    st.write(f"Detected Emotion: {emotion}")

# Real-Time Webcam Section
st.subheader("Real-Time Emotion Detection")
run = st.checkbox('Use Webcam')
FRAME_WINDOW = st.image([])

camera = None
if run:
    camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to capture image")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emotion = detect_emotions(frame)
    st.write(f"Detected Emotion: {emotion}")
    FRAME_WINDOW.image(frame)

if not run and camera is not None:
    camera.release()
    st.write('Webcam stopped.')
