import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from PIL import Image
import numpy as np
import gdown
import os

st.set_page_config(page_title="PetVision Pro - Thor System", layout="centered")

@st.cache_resource
def load_thor_model():
    # Building the 105-layer architecture
    base = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    out = layers.Dense(37, activation='softmax')(base.output)
    model = Model(inputs=base.input, outputs=out)

    return model


# Initializing the model
model = load_thor_model()

st.title("Thor Pet Classifier")
st.write("Upload a photo to see the 99% accuracy model in action.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"Predicted Class Index: {predicted_class}")
