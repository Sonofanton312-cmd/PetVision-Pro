import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import backend as K

# --- 1. CONFIG ---
st.set_page_config(page_title="PetVision Pro", page_icon="🐾")

# --- 2. MODEL LOADING (RE-ENGINEERED) ---
@st.cache_resource
def load_my_model():
    K.clear_session() # Clears existing models from memory
    model_path = 'pet_classifier_PRO_v1.h5'
    
    class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            if 'groups' in kwargs: kwargs.pop('groups')
            super().__init__(**kwargs)

    return tf.keras.models.load_model(
        model_path, 
        custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
        compile=False
    )

model = load_my_model()

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
uploaded_file = st.file_uploader("Upload Pet Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    with st.spinner('Analyzing...'):
        # FIXED PREPROCESSING
        img_resized = image.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        
        # Ensure it is a single float32 array
        img_array = np.array([img_array], dtype=np.float32) 
        img_preprocessed = preprocess_input(img_array)
        
        # Make Prediction - explicitly using the single tensor
        preds = model(img_preprocessed, training=False).numpy()[0]
        
        top_3 = preds.argsort()[-3:][::-1]
        for i in top_3:
            label = unique_labels[i].replace('_', ' ').title()
            st.write(f"**{label}**: {preds[i]*100:.1f}%")
            st.progress(float(preds[i]))
