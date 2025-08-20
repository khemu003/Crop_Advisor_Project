from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from database.db_utils import db_manager
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/predictions")
async def get_predictions(
    limit: int = Query(100, ge=1, le=1000, description="Number of predictions to return"),
    class_filter: Optional[str] = Query(None, description="Filter by predicted class"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """Get predictions with optional filtering."""
    try:
        if class_filter:
            predictions = db_manager.get_predictions_by_class(class_filter, limit)
        else:
            predictions = db_manager.get_predictions(limit)
        
        # Apply confidence filter if specified
        if min_confidence is not None:
            predictions = [p for p in predictions if p['confidence'] >= min_confidence]
        
        return {
            "success": True,
            "count": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve predictions: {str(e)}")

@router.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: int):
    """Get a specific prediction by ID."""
    try:
        prediction = db_manager.get_prediction_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {
            "success": True,
            "prediction": prediction
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prediction: {str(e)}")

@router.delete("/predictions/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """Delete a prediction by ID."""
    try:
        success = db_manager.delete_prediction(prediction_id)
        if not success:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {
            "success": True,
            "message": f"Prediction {prediction_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete prediction: {str(e)}")

@router.get("/statistics")
async def get_statistics():
    """Get database statistics and analytics."""
    try:
        stats = db_manager.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

@router.get("/classes")
async def get_classes():
    """Get list of all predicted classes."""
    try:
        stats = db_manager.get_statistics()
        classes = list(stats.get("class_distribution", {}).keys())
        return {
            "success": True,
            "classes": classes,
            "count": len(classes)
        }
    except Exception as e:
        logger.error(f"Error retrieving classes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve classes: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        stats = db_manager.get_statistics()
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": stats.get("total_predictions", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }