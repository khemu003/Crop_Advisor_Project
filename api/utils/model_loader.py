import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io

class ModelLoader:
    def __init__(self, model_path, class_labels_dir):
        """Initialize model and class labels."""
        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = sorted(os.listdir(class_labels_dir))  # e.g., ['Apple___healthy', ...]
    
    def preprocess_image(self, image_data):
        """Preprocess image from BytesIO or file path."""
        if isinstance(image_data, str):
            # Handle file path
            img = load_img(image_data, target_size=(224, 224))
        else:
            # Handle BytesIO
            img = Image.open(image_data)
            img = img.resize((224, 224))
            if img.mode != 'RGB':
                img = img.convert('RGB')
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    def predict(self, image_data):
        """Predict disease from image (file path or BytesIO)."""
        img_array = self.preprocess_image(image_data)
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = self.class_labels[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        return predicted_class, confidence

# Example usage (for testing)
if __name__ == "__main__":
    model_path = "models/saved_models/crop_disease_cnn.h5"
    class_labels_dir = "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    model_loader = ModelLoader(model_path, class_labels_dir)
    test_image = "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/Apple___healthy/0a8c7d65-0f9b-4d83-8f28-9ce3f2182a6b___RS_HL 7630.JPG"
    predicted_class, confidence = model_loader.predict(test_image)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")