from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.routes.predict import router as predict_router
from api.routes.data import router as data_router
from api.routes.recommend import router as recommend_router
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Powered Crop Disease Prediction API",
    description="A comprehensive API for detecting crop diseases and providing treatment recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(predict_router, prefix="/api/v1", tags=["prediction"])
app.include_router(data_router, prefix="/api/v1", tags=["data"])
app.include_router(recommend_router, prefix="/api/v1", tags=["recommendations"])

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the AI-Powered Crop Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/api/v1/predict",
            "data": "/api/v1/predictions",
            "recommendations": "/api/v1/recommendations",
            "documentation": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "crop-disease-prediction-api",
        "version": "1.0.0"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail="An unexpected error occurred. Please try again later."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
