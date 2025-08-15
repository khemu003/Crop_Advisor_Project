import streamlit as st
import requests
from PIL import Image
import io
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="Crop Disease Advisor", page_icon="🌱", layout="wide")

# Title and description
st.title("🌾 AI-Powered Sustainable Agriculture Advisor")
st.markdown("Upload a crop leaf image to detect diseases and get recommendations.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        logger.error(f"Image display error: {str(e)}")
        st.stop()

    # Send image to FastAPI endpoint
    with st.spinner("Analyzing image..."):
        try:
            # Prepare file for API
            image_data = io.BytesIO()
            image.save(image_data, format=image.format or "JPEG")
            image_data.seek(0)
            files = {"file": (uploaded_file.name, image_data, uploaded_file.type or "image/jpeg")}
            
            # Log request
            logger.debug(f"Sending request to API with file: {uploaded_file.name}")
            response = requests.post("http://127.0.0.1:8000/api/v1/predict", files=files, timeout=10)

            # Check response
            logger.debug(f"API response status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                st.write(f"**Predicted Disease**: {result['predicted_class']}")
                st.write(f"**Confidence**: {result['confidence']:.2%}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                logger.error(f"API error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {str(e)}")
            logger.error(f"API connection error: {str(e)}")
        except ValueError as e:
            st.error(f"Error parsing API response: {str(e)}")
            logger.error(f"Response parsing error: {str(e)}")

# Sidebar for additional info
st.sidebar.header("About")
st.sidebar.markdown(
    """
    This app uses a TensorFlow CNN to detect crop diseases from leaf images.
    - Trained on 87K+ images across 38 classes.
    - Achieves ~85% accuracy.
    - Built with FastAPI, Streamlit, and TensorFlow.
    """
)

# Link to dashboard
if st.sidebar.button("View Model Metrics"):
    st.session_state.page = "dashboard"

# Dashboard navigation
if "page" in st.session_state and st.session_state.page == "dashboard":
    st.subheader("Model Performance Metrics")
    metrics_image = "models/classification/training_metrics.png"
    if os.path.exists(metrics_image):
        st.image(metrics_image, caption="Training and Validation Metrics")
    else:
        st.warning("Metrics plot not found. Run training script to generate.")