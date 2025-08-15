from fastapi import FastAPI
from api.routes.predict import router as predict_router

app = FastAPI(title="Crop Disease Prediction API")

# Include prediction router
app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the Crop Disease Prediction API"}
