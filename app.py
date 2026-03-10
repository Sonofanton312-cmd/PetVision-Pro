import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. CONFIG ---
st.set_page_config(page_title="PetVision Pro", page_icon="🐾")

# --- 2. MODEL LOADING (NO CACHE TO PREVENT GRAPH ERRORS) ---
def load_my_model():
    model_path = 'pet_classifier_PRO_v1.h5'
    
    # Custom class for the version fix
    class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            if 'groups' in kwargs: kwargs.pop('groups')
            super().__init__(**kwargs)

    return tf.keras.models.load_model(
        model_path, 
        custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
        compile=False
    )

# Load once per run
if 'model' not in st.session_state:
    st.session_state.model = load_my_model()

model = st.session_state.model

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
        # 1. Resize
        img_resized = image.resize((128, 128))
        # 2. Convert to array
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        # 3. Add Batch Dimension (None, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # 4. MobileNet Preprocessing
        img_preprocessed = preprocess_input(img_array)
        
        # 5. Predict using the standard method
        preds = model.predict(img_preprocessed)[0]
        
        top_3 = preds.argsort()[-3:][::-1]
        st.subheader("Results:")
        for i in top_3:
            label = unique_labels[i].replace('_', ' ').title()
            st.write(f"**{label}**: {preds[i]*100:.1f}%")
            st.progress(float(preds[i]))
