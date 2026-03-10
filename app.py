import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# --- 1. CONFIG ---
st.set_page_config(page_title="PetVision Pro", page_icon="🐾")

# --- 2. THE SKELETON-WEIGHTS INJECTION ---
@st.cache_resource
def load_fixed_model():
    model_path = 'pet_classifier_PRO_v1.h5'
    
    # Create the 'Engine' (Architecture) from scratch
    # This matches the 128x128 input you trained on
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights=None, pooling='avg')
    
    # Add the 'Top' (The 37 breed categories)
    predictions = tf.keras.layers.Dense(37, activation='softmax')(base_model.output)
    full_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    # Load your 9MB of 'Intelligence' into this new engine
    # We use by_name=True to ensure every synapse matches the right layer
    full_model.load_weights(model_path, by_name=True)
    return full_model

model = load_fixed_model()

# --- 3. LABELS ---
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

# --- 4. UI ---
st.title("🐾 PetVision Pro")
st.info("System Status: Architecture Rebuilt & Weights Synced")

uploaded_file = st.file_uploader("Upload Pet Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    with st.spinner('AI Analysis in Progress...'):
        # Preprocessing
        img_resized = image.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array)
        
        # Predict using the newly built model
        preds = model.predict(img_preprocessed)[0]
        
        top_3 = preds.argsort()[-3:][::-1]
        st.subheader("Results:")
        for i in top_3:
            label = unique_labels[i].replace('_', ' ').title()
            st.write(f"**{label}**: {preds[i]*100:.2f}%")
            st.progress(float(preds[i]))
