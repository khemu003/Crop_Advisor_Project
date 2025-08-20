# import os
# import logging
# from typing import Tuple, List, Optional
# import numpy as np
# from PIL import Image

# logger = logging.getLogger(__name__)

# class ModelLoader:
#     """Model loader for crop disease prediction."""
    
#     def __init__(self, model_path: str, class_labels_dir: str):
#         """Initialize model loader."""
#         self.model_path = model_path
#         self.class_labels_dir = class_labels_dir
#         self.model = None
#         self.class_labels = []
        
#         # Check if TensorFlow is available
#         try:
#             import tensorflow as tf
#             self.tensorflow_available = True
#             logger.info("TensorFlow is available")
#         except ImportError:
#             self.tensorflow_available = False
#             logger.warning("TensorFlow not available, using mock predictions for testing")
        
#         # Load class labels
#         self._load_class_labels()
        
#         # Load model if TensorFlow is available
#         if self.tensorflow_available:
#             self._load_model()
    
#     def _load_class_labels(self):
#         """Load class labels from directory structure."""
#         try:
#             if os.path.exists(self.class_labels_dir):
#                 self.class_labels = sorted([d for d in os.listdir(self.class_labels_dir) 
#                                          if os.path.isdir(os.path.join(self.class_labels_dir, d))])
#                 logger.info(f"Loaded {len(self.class_labels)} class labels")
#             else:
#                 # Fallback class labels for testing
#                 self.class_labels = [
#                     'Apple___healthy', 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
#                     'Tomato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
#                     'Corn___healthy', 'Corn___Gray_leaf_spot', 'Corn___Common_rust'
#                 ]
#                 logger.warning("Class labels directory not found, using fallback labels")
#         except Exception as e:
#             logger.error(f"Error loading class labels: {e}")
#             self.class_labels = ['healthy', 'diseased']
    
#     def _load_model(self):
#         """Load the trained model."""
#         try:
#             if os.path.exists(self.model_path):
#                 import tensorflow as tf
#                 self.model = tf.keras.models.load_model(self.model_path)
#                 logger.info("Model loaded successfully")
#             else:
#                 logger.warning(f"Model file not found: {self.model_path}")
#         except Exception as e:
#             logger.error(f"Error loading model: {e}")
#             self.model = None
    
#     def preprocess_image(self, image: Image.Image) -> np.ndarray:
#         """Preprocess image for model input."""
#         try:
#             # Resize image to 224x224
#             image = image.resize((224, 224))
            
#             # Convert to numpy array and normalize
#             img_array = np.array(image) / 255.0
            
#             # Add batch dimension
#             img_array = np.expand_dims(img_array, axis=0)
            
#             return img_array
#         except Exception as e:
#             logger.error(f"Error preprocessing image: {e}")
#             raise
    
#     def predict(self, image: Image.Image) -> Tuple[str, float]:
#         """Predict disease class for the given image."""
#         try:
#             if not self.tensorflow_available or self.model is None:
#                 # Return mock prediction for testing
#                 return self._mock_prediction(image)
            
#             # Preprocess image
#             img_array = self.preprocess_image(image)
            
#             # Make prediction
#             predictions = self.model.predict(img_array)
#             predicted_class_idx = np.argmax(predictions[0])
#             confidence = float(predictions[0][predicted_class_idx])
            
#             # Get class label
#             if predicted_class_idx < len(self.class_labels):
#                 predicted_class = self.class_labels[predicted_class_idx]
#             else:
#                 predicted_class = "unknown"
            
#             return predicted_class, confidence
            
#         except Exception as e:
#             logger.error(f"Error during prediction: {e}")
#             # Return fallback prediction
#             return self._mock_prediction(image)
    
#     def _mock_prediction(self, image: Image.Image) -> Tuple[str, float]:
#         """Generate mock prediction for testing when TensorFlow is not available."""
#         # Simple mock prediction based on image size
#         width, height = image.size
        
#         # Mock logic: larger images tend to be "healthy"
#         if width * height > 100000:  # Large image
#             predicted_class = "Apple___healthy"
#             confidence = 0.85
#         else:
#             predicted_class = "Tomato___Bacterial_spot"
#             confidence = 0.78
        
#         logger.info(f"Mock prediction: {predicted_class} (confidence: {confidence})")
#         return predicted_class, confidence
    
#     def get_class_labels(self) -> List[str]:
#         """Get list of available class labels."""
#         return self.class_labels.copy()
    
#     def is_available(self) -> bool:
#         """Check if model is available for predictions."""
#         return self.tensorflow_available and self.model is not None


import os
import logging
from typing import Tuple, List
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class ModelLoader:
    """Model loader for crop disease prediction."""

    def __init__(self, model_path: str, class_labels_dir: str):
        """Initialize model loader."""
        self.model_path = model_path
        self.class_labels_dir = class_labels_dir
        self.model = None
        self.class_labels = []

        # Try importing TensorFlow
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            logger.info("TensorFlow is available")
        except ImportError:
            self.tensorflow_available = False
            logger.warning("TensorFlow not available, using mock predictions")

        # Load labels + model
        self._load_class_labels()
        if self.tensorflow_available:
            self._load_model()

    def _load_class_labels(self):
        """Load class labels from dataset directory."""
        try:
            if os.path.exists(self.class_labels_dir):
                self.class_labels = sorted([
                    d for d in os.listdir(self.class_labels_dir)
                    if os.path.isdir(os.path.join(self.class_labels_dir, d))
                ])
                logger.info(f"Loaded {len(self.class_labels)} class labels")
            else:
                # Fallback labels
                self.class_labels = [
                    'Apple___healthy', 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                    'Tomato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                    'Corn___healthy', 'Corn___Gray_leaf_spot', 'Corn___Common_rust'
                ]
                logger.warning("Class labels dir not found → using fallback")
        except Exception as e:
            logger.error(f"Error loading class labels: {e}")
            self.class_labels = ['healthy', 'diseased']

    def _load_model(self):
        """Load TensorFlow model."""
        try:
            if os.path.exists(self.model_path):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("✅ Model loaded successfully")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Resize + normalize image for model input."""
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """Predict disease class from image."""
        try:
            if not self.tensorflow_available or self.model is None:
                return self._mock_prediction(image)

            img_array = self.preprocess_image(image)
            predictions = self.model.predict(img_array)
            idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][idx])

            predicted_class = self.class_labels[idx] if idx < len(self.class_labels) else "unknown"
            return predicted_class, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._mock_prediction(image)

    def _mock_prediction(self, image: Image.Image) -> Tuple[str, float]:
        """Mock prediction when model unavailable."""
        width, height = image.size
        if width * height > 100000:
            return "Apple___healthy", 0.85
        else:
            return "Tomato___Bacterial_spot", 0.78

    def get_class_labels(self) -> List[str]:
        return self.class_labels.copy()

    def is_available(self) -> bool:
        return self.tensorflow_available and self.model is not None
