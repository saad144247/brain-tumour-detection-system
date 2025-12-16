import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import os

# --- CORRECT CONSTANTS FOR PNEUMONIA ---
# Apne Hugging Face Repo ID ka istemaal karein
HF_REPO_ID = "saad1BM/pneumonia-detection-system"
MODEL_FILENAME = "pneumonia_detection_model.keras" 
MODEL_PATH = MODEL_FILENAME  # Local path for caching

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


# --- Function to load model from Hugging Face ---
@st.cache_resource
def load_pneumonia_model():
    """Model ko Hugging Face se download aur load karta hai."""
    
    # Model file ko Hugging Face se download karein (Streamlit Cloud par directly cache hota hai)
    try:
        with st.spinner(f"Downloading model file ({MODEL_FILENAME}) from Hugging Face..."):
            # hf_hub_download se model ka local path mil jayega
            model_download_path = hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename=MODEL_FILENAME,
                # Simple download ke liye cache_dir zaroori nahi, lekin performance ke liye rakha hai
                cache_dir=os.path.join(os.getcwd(), ".hf_cache") 
            )
        
        st.success("Model downloaded successfully!")
        
        # Model ko load karein
        model = load_model(model_download_path)
        return model
    
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        st.info("Please make sure the repo is Public and model file name is correct.")
        return None

# Model load karein
model = load_pneumonia_model()


# --- Streamlit UI ---
st.set_page_config(page_title="Pneumonia Detection System", layout="centered")

st.title("🫁 Pneumonia Detection System (AI Powered)")
st.caption("Upload a chest X-ray image to predict Normal or Pneumonia.")
st.markdown("---")

uploaded_file = st.file_uploader("Choose a Chest X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Image Preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    
    # 2. Prediction Button
    if st.button("Detect Pneumonia"):
        if model is None:
            st.error("Model is not loaded. Cannot perform detection.")
            st.stop()
            
        st.subheader("📊 Prediction Result")
        
        with st.spinner('Analyzing X-ray image...'):
            # Preprocessing
            img = image.resize(IMAGE_SIZE)
            img_array = np.array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0) # Batch dimension add karna
            
            # Prediction
            predictions = model.predict(img_array)
            # Softmax for classification confidence (required for Keras Functional API output)
            score = tf.nn.softmax(predictions[0]) 
            
            # Result Display
            predicted_class_index = np.argmax(score)
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = np.max(score) * 100

            # --- Display Logic ---
            if predicted_class == 'PNEUMONIA':
                st.error(f"### ⚠️ Prediction: {predicted_class}")
                st.markdown(f"**Confidence:** **{confidence:.2f}%**")
                
            else:
                st.success(f"### ✅ Prediction: {predicted_class}")
                st.markdown(f"**Confidence:** **{confidence:.2f}%**")

            # Optional: Probability chart
            st.markdown("---")
            st.bar_chart({
                "Normal": score[0].numpy(),
                "Pneumonia": score[1].numpy()
            })