from fastapi import APIRouter, File, UploadFile, HTTPException
from api.utils.model_loader import ModelLoader
from rag.retriever import Retriever
from rag.generator import RecommendationGenerator
from database.db_utils import db_manager
import io
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)
router = APIRouter()

# Model and labels
model_path = "models/saved_models/crop_disease_cnn.h5"
class_labels_dir = "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

# Load model
model_loader = None
if os.path.exists(model_path):
    try:
        model_loader = ModelLoader(model_path, class_labels_dir)
    except Exception as e:
        logger.error(f"Model load failed: {e}")

# Load RAG
try:
    retriever = Retriever()
    generator = RecommendationGenerator()
except Exception as e:
    logger.error(f"RAG init failed: {e}")
    retriever = None
    generator = None


@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """Predict crop disease and provide recommendations."""
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image (JPEG, PNG, etc.)")

        if model_loader is None:
            raise HTTPException(503, "Prediction model unavailable")

        # Read file → BytesIO → PIL
        content = await file.read()
        file_size = len(content)
        image = Image.open(io.BytesIO(content)).convert("RGB")

        # Predict
        predicted_class, confidence = model_loader.predict(image)

        # Info from knowledge base
        retrieved_info = predicted_class
        try:
            with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(predicted_class + ":"):
                        retrieved_info = line.split(": ", 1)[1].strip()
                        break
        except FileNotFoundError:
            logger.warning("Knowledge base missing")

        # Generate recommendation
        recommendation = (
            generator.generate_recommendation(predicted_class, retrieved_info, confidence)
            if generator else f"No recommendation available for {predicted_class}"
        )

        # Save prediction
        try:
            db_manager.save_prediction(file.filename, predicted_class, confidence, recommendation)
        except Exception as e:
            logger.error(f"DB save failed: {e}")

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "message": f"Disease detected: {predicted_class} ({confidence:.2%})",
            "recommendation": recommendation,
            "retrieved_info": retrieved_info,
            "file_name": file.filename,
            "file_size_bytes": file_size,
            "file_type": file.content_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(500, f"Unexpected error: {str(e)}")


@router.get("/model-status")
async def get_model_status():
    return {
        "model_loaded": model_loader is not None,
        "rag_loaded": retriever is not None and generator is not None,
        "model_path": model_path,
        "model_exists": os.path.exists(model_path),
        "status": "ready" if model_loader and retriever and generator else "not_ready",
    }
