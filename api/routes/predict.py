from fastapi import APIRouter, File, UploadFile, HTTPException
from api.utils.model_loader import ModelLoader
import io

router = APIRouter()

# Initialize model loader
model_path = "models/saved_models/crop_disease_cnn.h5"
class_labels_dir = "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
model_loader = ModelLoader(model_path, class_labels_dir)

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """Predict crop disease from uploaded image."""
    try:
        # Read image data as BytesIO
        image_data = io.BytesIO(await file.read())
        
        # Predict
        predicted_class, confidence = model_loader.predict(image_data)
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "message": f"Detected disease: {predicted_class} with {confidence:.2%} confidence"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")