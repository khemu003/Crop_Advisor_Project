import streamlit as st
import requests
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Crop Disease Advisor", page_icon="🌱", layout="wide")
st.title("🌾 AI-Powered Sustainable Agriculture Advisor")
st.markdown("Upload a crop leaf image or take a photo to detect diseases and get recommendations.")

# --- Option 1: Upload from file ---
uploaded_file = st.file_uploader("Choose an image from your device...", type=["jpg", "jpeg", "png"])

# --- Option 2: Capture from camera ---
camera_file = st.camera_input("Or take a picture with your camera")

# Use whichever file is provided
file_to_use = uploaded_file or camera_file

if file_to_use is not None:
    try:
        image = Image.open(file_to_use)
        st.image(image, caption="Selected Image", width=300)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        logger.error(f"Image display error: {str(e)}")
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            # Save image in memory
            image_data = io.BytesIO()
            image.save(image_data, format=image.format or "JPEG")
            image_data.seek(0)

            # Prepare request
            files = {"file": (file_to_use.name, image_data, "image/jpeg")}
            api_url = "http://127.0.0.1:8000/api/v1/predict"  # Local testing
            logger.debug(f"Sending request to API: {api_url} with file: {file_to_use.name}")
            
            response = requests.post(api_url, files=files, timeout=10)
            logger.debug(f"API response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                st.write(f"**Predicted Disease**: {result['predicted_class']}")
                st.write(f"**Confidence**: {result['confidence']:.2%}")
                st.write(f"**Recommendation**: {result['recommendation']}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                logger.error(f"API error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {str(e)}")
            logger.error(f"API connection error: {str(e)}")
        except ValueError as e:
            st.error(f"Error parsing API response: {str(e)}")
            logger.error(f"Response parsing error: {str(e)}")
