from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from rag.generator import RecommendationGenerator
from rag.retriever import Retriever
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize RAG components
retriever = Retriever()
generator = RecommendationGenerator()

@router.get("/recommendations/{disease_class}")
async def get_recommendation(
    disease_class: str,
    confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Confidence level for context")
):
    """Get treatment recommendations for a specific disease class."""
    try:
        # Retrieve relevant information from knowledge base
        retrieved = retriever.retrieve(disease_class, k=3)
        
        if not retrieved:
            raise HTTPException(status_code=404, detail=f"No information found for disease class: {disease_class}")
        
        # Get the most relevant information
        retrieved_info = retrieved[0]['class_name']
        
        # Read from knowledge base for detailed information
        try:
            with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(disease_class + ":"):
                        retrieved_info = line.split(": ", 1)[1].strip()
                        break
        except FileNotFoundError:
            logger.warning("Knowledge base file not found, using retrieved info")
        
        # Generate recommendation using AI
        if confidence is None:
            confidence = 0.85  # Default confidence
        
        recommendation = generator.generate_recommendation(
            disease_class, 
            retrieved_info, 
            confidence
        )
        
        return {
            "success": True,
            "disease_class": disease_class,
            "confidence": confidence,
            "retrieved_info": retrieved_info,
            "recommendation": recommendation,
            "similar_diseases": [r['class_name'] for r in retrieved[1:]] if len(retrieved) > 1 else []
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendation for {disease_class}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendation: {str(e)}")

@router.get("/recommendations")
async def get_recommendations(
    query: str = Query(..., description="Search query for disease or symptoms"),
    limit: int = Query(5, ge=1, le=10, description="Number of recommendations to return")
):
    """Search for disease recommendations based on a query."""
    try:
        # Retrieve relevant documents
        retrieved = retriever.retrieve(query, k=limit)
        
        if not retrieved:
            return {
                "success": True,
                "query": query,
                "count": 0,
                "recommendations": [],
                "message": "No relevant diseases found for the query"
            }
        
        recommendations = []
        for item in retrieved:
            disease_class = item['class_name']
            
            # Get detailed information
            try:
                with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith(disease_class + ":"):
                            info = line.split(": ", 1)[1].strip()
                            break
                    else:
                        info = f"Information about {disease_class}"
            except FileNotFoundError:
                info = f"Information about {disease_class}"
            
            # Generate AI recommendation
            try:
                ai_recommendation = generator.generate_recommendation(
                    disease_class, 
                    info, 
                    0.8  # Default confidence for search results
                )
            except Exception as e:
                ai_recommendation = f"Error generating recommendation: {str(e)}"
            
            recommendations.append({
                "disease_class": disease_class,
                "relevance_score": 1.0 - item['distance'],  # Convert distance to relevance
                "basic_info": info,
                "ai_recommendation": ai_recommendation
            })
        
        return {
            "success": True,
            "query": query,
            "count": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error searching recommendations for query '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search recommendations: {str(e)}")

@router.get("/prevention-tips")
async def get_prevention_tips(
    crop_type: Optional[str] = Query(None, description="Specific crop type for targeted tips")
):
    """Get general prevention tips for crop diseases."""
    try:
        if crop_type:
            # Get crop-specific tips
            query = f"prevention tips for {crop_type} diseases"
            retrieved = retriever.retrieve(query, k=3)
            
            if retrieved:
                tips = []
                for item in retrieved:
                    disease_class = item['class_name']
                    try:
                        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith(disease_class + ":"):
                                    info = line.split(": ", 1)[1].strip()
                                    tips.append({
                                        "disease": disease_class,
                                        "tip": info
                                    })
                                    break
                    except FileNotFoundError:
                        continue
                
                return {
                    "success": True,
                    "crop_type": crop_type,
                    "tips": tips
                }
        
        # General prevention tips
        general_tips = [
            "Practice crop rotation to prevent disease buildup in soil",
            "Maintain proper spacing between plants for good air circulation",
            "Avoid overhead watering to reduce leaf wetness",
            "Remove and destroy infected plant debris",
            "Use disease-resistant varieties when available",
            "Monitor plants regularly for early disease detection",
            "Apply appropriate fungicides or bactericides as preventive measures",
            "Maintain healthy soil with proper fertilization and pH levels"
        ]
        
        return {
            "success": True,
            "crop_type": "general",
            "tips": [{"tip": tip} for tip in general_tips]
        }
    except Exception as e:
        logger.error(f"Error getting prevention tips: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get prevention tips: {str(e)}")

@router.get("/disease-info/{disease_class}")
async def get_disease_info(disease_class: str):
    """Get detailed information about a specific disease."""
    try:
        # Get basic information from knowledge base
        basic_info = ""
        try:
            with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(disease_class + ":"):
                        basic_info = line.split(": ", 1)[1].strip()
                        break
        except FileNotFoundError:
            basic_info = f"Information about {disease_class}"
        
        if not basic_info:
            raise HTTPException(status_code=404, detail=f"No information found for disease: {disease_class}")
        
        # Generate detailed AI explanation
        try:
            detailed_info = generator.generate_recommendation(
                disease_class,
                basic_info,
                0.9
            )
        except Exception as e:
            detailed_info = f"Error generating detailed information: {str(e)}"
        
        return {
            "success": True,
            "disease_class": disease_class,
            "basic_info": basic_info,
            "detailed_explanation": detailed_info,
            "treatment_recommendations": basic_info  # Basic info contains treatment recommendations
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting disease info for {disease_class}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get disease information: {str(e)}")