import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# --- 1. CONFIG ---
st.set_page_config(page_title="PetVision Pro", page_icon="🐾")

# --- 2. THE HYBRID LOAD (SKELETON + WEIGHTS) ---
@st.cache_resource
def load_full_model():
    model_path = 'pet_classifier_PRO_v1.h5'
    
    # 1. Create a fresh 'Skeleton' of MobileNetV2
    # This matches the architecture you used on your RTX 3050
    base = MobileNetV2(input_shape=(128, 128, 3), weights=None, classes=37)
    
    # 2. Try to load your 9MB of brain power into this skeleton
    try:
        base.load_weights(model_path)
    except:
        # If load_weights fails, try loading the whole file with a version-bypass
        class FixedDepthwise(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, **kwargs):
                if 'groups' in kwargs: kwargs.pop('groups')
                super().__init__(**kwargs)
        
        return tf.keras.models.load_model(
            model_path, 
            custom_objects={'DepthwiseConv2D': FixedDepthwise},
            compile=False
        )
    return base

model = load_full_model()

# --- 3. LABELS (Exactly as your 3050 trained them) ---
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
st.info("System Status: Hybrid Brain Active")

uploaded_file = st.file_uploader("Upload Pet Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    with st.spinner('AI is Thinking...'):
        img = image.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array)
        
        preds = model.predict(img_preprocessed)[0]
        top_3 = preds.argsort()[-3:][::-1]
        
        for i in top_3:
            label = unique_labels[i].replace('_', ' ').title()
            st.write(f"**{label}**: {preds[i]*100:.1f}%")
            st.progress(float(preds[i]))
