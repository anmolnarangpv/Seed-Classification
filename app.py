import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('vgg16.h5')

# Define the image size (should match the training size)
IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Make batch of 1
    return image

st.title('Weed Detection AI')
st.write("Upload an image of the crop field, and the AI will predict if there are weeds.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png","tiff"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = preprocess_image(image)
    prediction = model.predict(image)
    labels = ['broadleaf', 'grass', 'soil', 'soybean']  # Update with your actual labels
    st.write(f"Prediction: {labels[np.argmax(prediction)]}")

# To run the app, save this script as `app.py` and run the following command in your terminal:
# streamlit run app.py
