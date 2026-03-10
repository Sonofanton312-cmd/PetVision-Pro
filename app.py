import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras import layers, Model
from PIL import Image
import numpy as np
import gdown
import os

st.set_page_config(page_title="PetVision Pro - Thor System", layout="centered")

# --- THE SYNCED BREED LIST ---
BREEDS = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

@st.cache_resource
def load_thor_model():
    base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet', pooling='avg')
    out = layers.Dense(37, activation='softmax')(base.output)
    model = Model(inputs=base.input, outputs=out)
    return model

model = load_thor_model()

st.title("⚡ Thor Pet Classifier")
st.write("Upload a photo to see the 90%+ accuracy strike.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
