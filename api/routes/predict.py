from fastapi import APIRouter, File, UploadFile, HTTPException
from api.utils.model_loader import ModelLoader
from rag.retriever import Retriever
from rag.generator import RecommendationGenerator
import io
import os
import sqlite3

router = APIRouter()

# Initialize model loader, retriever, and generator
model_path = "models/saved_models/crop_disease_cnn.h5"
class_labels_dir = "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
model_loader = ModelLoader(model_path, class_labels_dir)
retriever = Retriever()
generator = RecommendationGenerator()

# SQLite database path
DB_PATH = "data/db/predictions.db"

def save_prediction(image_name, predicted_class, confidence, recommendation):
    """Save prediction to SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (image_name, predicted_class, confidence, recommendation)
        VALUES (?, ?, ?, ?)
        """,
        (image_name, predicted_class, confidence, recommendation)
    )
    conn.commit()
    conn.close()

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """Predict crop disease and provide recommendations."""
    try:
        # Read image data
        image_data = io.BytesIO(await file.read())
        
        # Predict
        predicted_class, confidence = model_loader.predict(image_data)
        
        # Retrieve relevant info
        retrieved = retriever.retrieve(predicted_class)
        retrieved_info = f"Class: {retrieved[0]['class_name']}, Info from knowledge base: {retrieved[0]['class_name']}"
        with open("data/knowledge_base.txt", "r") as f:
            for line in f:
                if line.startswith(predicted_class + ":"):
                    retrieved_info = line.split(": ", 1)[1].strip()
                    break
        
        # Generate recommendation
        recommendation = generator.generate_recommendation(predicted_class, retrieved_info, confidence)
        
        # Save to database
        save_prediction(file.filename, predicted_class, confidence, recommendation)
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "message": f"Detected disease: {predicted_class} with {confidence:.2%} confidence",
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")