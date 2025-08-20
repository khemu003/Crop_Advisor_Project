from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import os
from typing import List, Optional
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

Base = declarative_base()

class Prediction(Base):
    """SQLAlchemy model for predictions table."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_name = Column(String(255), nullable=False)
    predicted_class = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    recommendation = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())

class MockDatabaseManager:
    """Mock database manager for testing without MySQL."""
    
    def __init__(self):
        """Initialize mock database manager."""
        self.predictions = []
        self.next_id = 1
        logger.info("Using mock database manager for testing")
    
    def save_prediction(self, image_name: str, predicted_class: str, 
                       confidence: float, recommendation: str) -> bool:
        """Save a new prediction to mock database."""
        try:
            prediction = {
                "id": self.next_id,
                "image_name": image_name,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "recommendation": recommendation,
                "timestamp": "2024-01-01T00:00:00"
            }
            self.predictions.append(prediction)
            self.next_id += 1
            logger.info(f"Mock prediction saved: {image_name} -> {predicted_class}")
            return True
        except Exception as e:
            logger.error(f"Error saving mock prediction: {str(e)}")
            return False
    
    def get_predictions(self, limit: int = 100) -> List[dict]:
        """Get recent predictions from mock database."""
        try:
            # Return predictions in reverse order (newest first)
            sorted_predictions = sorted(self.predictions, key=lambda x: x['id'], reverse=True)
            return sorted_predictions[:limit]
        except Exception as e:
            logger.error(f"Error retrieving mock predictions: {str(e)}")
            return []
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[dict]:
        """Get prediction by ID from mock database."""
        try:
            for pred in self.predictions:
                if pred['id'] == prediction_id:
                    return pred
            return None
        except Exception as e:
            logger.error(f"Error retrieving mock prediction {prediction_id}: {str(e)}")
            return None
    
    def get_predictions_by_class(self, predicted_class: str, limit: int = 50) -> List[dict]:
        """Get predictions filtered by predicted class from mock database."""
        try:
            filtered = [p for p in self.predictions if p['predicted_class'] == predicted_class]
            sorted_filtered = sorted(filtered, key=lambda x: x['id'], reverse=True)
            return sorted_filtered[:limit]
        except Exception as e:
            logger.error(f"Error retrieving mock predictions for class {predicted_class}: {str(e)}")
            return []
    
    def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a prediction by ID from mock database."""
        try:
            for i, pred in enumerate(self.predictions):
                if pred['id'] == prediction_id:
                    del self.predictions[i]
                    logger.info(f"Mock prediction {prediction_id} deleted")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error deleting mock prediction {prediction_id}: {str(e)}")
            return False
    
    def get_statistics(self) -> dict:
        """Get mock database statistics."""
        try:
            total_predictions = len(self.predictions)
            unique_classes = len(set(p['predicted_class'] for p in self.predictions))
            avg_confidence = sum(p['confidence'] for p in self.predictions) / total_predictions if total_predictions > 0 else 0.0
            
            # Get class distribution
            class_counts = {}
            for pred in self.predictions:
                class_counts[pred['predicted_class']] = class_counts.get(pred['predicted_class'], 0) + 1
            
            return {
                "total_predictions": total_predictions,
                "unique_classes": unique_classes,
                "average_confidence": round(avg_confidence, 3),
                "class_distribution": class_counts
            }
        except Exception as e:
            logger.error(f"Error retrieving mock statistics: {str(e)}")
            return {}

class DatabaseManager:
    """Database manager for MySQL operations."""
    
    def __init__(self):
        """Initialize MySQL database connection."""
        # Check if we should use mock mode
        if os.getenv("USE_MOCK_DB", "false").lower() == "true":
            logger.info("Using mock database mode")
            self._mock_mode = True
            self._mock_manager = MockDatabaseManager()
            return
        
        # Get database configuration from environment variables
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_NAME", "crop_advisor")
        db_user = os.getenv("DB_USER", "root")
        db_password = os.getenv("DB_PASSWORD", "")
        
        # Check if we have the minimum required credentials
        if not db_password or db_password == "your_mysql_password_here":
            logger.warning("MySQL credentials not configured, using mock database mode")
            self._mock_mode = True
            self._mock_manager = MockDatabaseManager()
            return
        
        self._mock_mode = False
        
        # Create MySQL connection string
        connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        try:
            # Create MySQL engine
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False
            )
            
            # Create tables if they don't exist
            Base.metadata.create_all(bind=self.engine)
            
            # Create session factory
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.SessionLocal = SessionLocal
            
            logger.info(f"Connected to MySQL database: {db_host}:{db_port}/{db_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MySQL database: {str(e)}")
            logger.warning("Falling back to mock database mode")
            self._mock_mode = True
            self._mock_manager = MockDatabaseManager()
    
    def get_session(self):
        """Get database session."""
        if self._mock_mode:
            return None
        return self.SessionLocal()
    
    def save_prediction(self, image_name: str, predicted_class: str, 
                       confidence: float, recommendation: str) -> bool:
        """Save a new prediction to database."""
        if self._mock_mode:
            return self._mock_manager.save_prediction(image_name, predicted_class, confidence, recommendation)
        
        try:
            session = self.get_session()
            prediction = Prediction(
                image_name=image_name,
                predicted_class=predicted_class,
                confidence=confidence,
                recommendation=recommendation
            )
            session.add(prediction)
            session.commit()
            session.close()
            logger.info(f"Prediction saved: {image_name} -> {predicted_class}")
            return True
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return False
    
    def get_predictions(self, limit: int = 100) -> List[dict]:
        """Get recent predictions from database."""
        if self._mock_mode:
            return self._mock_manager.get_predictions(limit)
        
        try:
            session = self.get_session()
            predictions = session.query(Prediction).order_by(
                Prediction.timestamp.desc()
            ).limit(limit).all()
            
            result = []
            for pred in predictions:
                result.append({
                    "id": pred.id,
                    "image_name": pred.image_name,
                    "predicted_class": pred.predicted_class,
                    "confidence": pred.confidence,
                    "recommendation": pred.recommendation,
                    "timestamp": pred.timestamp.isoformat() if pred.timestamp else None
                })
            
            session.close()
            return result
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[dict]:
        """Get prediction by ID."""
        if self._mock_mode:
            return self._mock_manager.get_prediction_by_id(prediction_id)
        
        try:
            session = self.get_session()
            prediction = session.query(Prediction).filter(
                Prediction.id == prediction_id
            ).first()
            
            if prediction:
                result = {
                    "id": prediction.id,
                    "image_name": prediction.image_name,
                    "predicted_class": prediction.predicted_class,
                    "confidence": prediction.confidence,
                    "recommendation": prediction.recommendation,
                    "timestamp": prediction.timestamp.isoformat() if prediction.timestamp else None
                }
                session.close()
                return result
            
            session.close()
            return None
        except Exception as e:
            logger.error(f"Error retrieving prediction {prediction_id}: {str(e)}")
            return None
    
    def get_predictions_by_class(self, predicted_class: str, limit: int = 50) -> List[dict]:
        """Get predictions filtered by predicted class."""
        if self._mock_mode:
            return self._mock_manager.get_predictions_by_class(predicted_class, limit)
        
        try:
            session = self.get_session()
            predictions = session.query(Prediction).filter(
                Prediction.predicted_class == predicted_class
            ).order_by(Prediction.timestamp.desc()).limit(limit).all()
            
            result = []
            for pred in predictions:
                result.append({
                    "id": pred.id,
                    "image_name": pred.image_name,
                    "predicted_class": pred.predicted_class,
                    "confidence": pred.confidence,
                    "recommendation": pred.recommendation,
                    "timestamp": pred.timestamp.isoformat() if pred.timestamp else None
                })
            
            session.close()
            return result
        except Exception as e:
            logger.error(f"Error retrieving predictions for class {predicted_class}: {str(e)}")
            return []
    
    def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a prediction by ID."""
        if self._mock_mode:
            return self._mock_manager.delete_prediction(prediction_id)
        
        try:
            session = self.get_session()
            prediction = session.query(Prediction).filter(
                Prediction.id == prediction_id
            ).first()
            
            if prediction:
                session.delete(prediction)
                session.commit()
                session.close()
                logger.info(f"Prediction {prediction_id} deleted")
                return True
            
            session.close()
            return False
        except Exception as e:
            logger.error(f"Error deleting prediction {prediction_id}: {str(e)}")
            return False
    
    def get_statistics(self) -> dict:
        """Get database statistics."""
        if self._mock_mode:
            return self._mock_manager.get_statistics()
        
        try:
            session = self.get_session()
            
            total_predictions = session.query(Prediction).count()
            unique_classes = session.query(Prediction.predicted_class).distinct().count()
            avg_confidence = session.query(func.avg(Prediction.confidence)).scalar() or 0.0
            
            # Get class distribution
            class_counts = session.query(
                Prediction.predicted_class,
                func.count(Prediction.id)
            ).group_by(Prediction.predicted_class).all()
            
            session.close()
            
            return {
                "total_predictions": total_predictions,
                "unique_classes": unique_classes,
                "average_confidence": round(avg_confidence, 3),
                "class_distribution": dict(class_counts)
            }
        except Exception as e:
            logger.error(f"Error retrieving statistics: {str(e)}")
            return {}

# Global database manager instance
db_manager = DatabaseManager()

if __name__ == "__main__":
    # Test database operations
    print("Testing database operations...")
    
    # Save a test prediction
    success = db_manager.save_prediction(
        image_name="test_image.jpg",
        predicted_class="Apple___healthy",
        confidence=0.95,
        recommendation="No treatment needed. Plant appears healthy."
    )
    print(f"Test prediction saved: {success}")
    
    # Get statistics
    stats = db_manager.get_statistics()
    print(f"Database statistics: {stats}")
    
    # Get recent predictions
    predictions = db_manager.get_predictions(limit=5)
    print(f"Recent predictions: {len(predictions)} found")