import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="PetVision Pro", page_icon="🐾", layout="centered")

# Custom CSS for the "Classy" look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .prediction-text { font-size: 18px; font-weight: bold; color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for Hardware & Project Info
with st.sidebar:
    st.header("Project Details")
    st.info("📍 **Location:** SSGMCE")
    st.info("💻 **Hardware:** NVIDIA RTX 3050 (Trained)")
    st.info("📊 **Model:** MobileNetV2")
    st.write("---")
    st.write("This AI recognizes 37 distinct breeds from the Oxford-IIIT Pet Dataset.")

st.title("🐾 PetVision Pro")
st.write("High-precision breed identification powered by Deep Learning.")
st.write("---")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_my_model():
    # REMOVED D: DRIVE PATH - Now looks in the same folder as this script
    model_path = 'pet_classifier_PRO_v1.h5'
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# --- 3. HARD-CODED LABELS ---
# These are the 37 breeds in alphabetical order (exactly how Keras sees folders)
unique_labels = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 
    'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 
    'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 
    'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 
    'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 
    'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 
    'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 
    'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
]

# --- 4. UI LAYOUT ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload a Pet Photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Photo', use_container_width=True)

with col2:
    st.subheader("AI Analysis")
    if uploaded_file:
        with st.spinner('Calculating probabilities...'):
            # Preprocessing (Match your training 128x128)
            img_resized = image.resize((128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make Prediction
            preds = model.predict(img_array)[0]
            
            # Get Top 3 results
            top_3_indices = preds.argsort()[-3:][::-1]
            
            for i in top_3_indices:
                label = unique_labels[i].replace('_', ' ').title()
                confidence = preds[i] * 100
                
                st.write(f"**{label}**")
                st.progress(int(confidence))
                st.write(f"{confidence:.2f}%")
                st.write("") 

            top_label = unique_labels[top_3_indices[0]].replace('_', ' ').title()
            if preds[top_3_indices[0]] > 0.80:
                st.success(f"High Confidence: **{top_label}**")
            else:
                st.warning(f"Likely: **{top_label}**")
    else:
        st.write("Please upload an image to see results.")