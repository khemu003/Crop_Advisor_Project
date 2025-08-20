import os
import logging
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np

logger = logging.getLogger(__name__)

class Retriever:
    """Retriever for semantic search using FAISS."""
    
    def __init__(self, index_path: str = "data/embeddings/faiss_index.bin"):
        """Initialize retriever."""
        self.index_path = index_path
        self.index = None
        self.embeddings = None
        self.class_names = []
        
        try:
            self._load_index()
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            logger.info("Retriever will use fallback search")
    
    def _load_index(self):
        """Load FAISS index and embeddings."""
        try:
            # Load embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Load FAISS index if it exists
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                
                # Load class names from metadata
                metadata_path = self.index_path.replace('.bin', '_metadata.txt')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                else:
                    # Fallback class names
                    self.class_names = [
                        'Apple___healthy', 'Apple___Apple_scab', 'Apple___Black_rot',
                        'Tomato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                        'Corn___healthy', 'Corn___Gray_leaf_spot', 'Corn___Common_rust'
                    ]
                
                logger.info(f"Loaded FAISS index with {len(self.class_names)} classes")
            else:
                logger.warning("FAISS index not found, using fallback search")
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = None
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most similar documents for the query."""
        try:
            if self.index is None or self.embeddings is None:
                return self._fallback_search(query, k)
            
            # Encode query using the correct method
            query_embedding = self.embeddings.embed_query(query)
            query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.class_names):
                    results.append({
                        'class_name': self.class_names[idx],
                        'distance': float(distance),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return self._fallback_search(query, k)
    
    def _fallback_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback search when FAISS is not available."""
        logger.info("Using fallback search")
        
        # Simple keyword-based search
        query_lower = query.lower()
        
        # Define some fallback classes with keywords
        fallback_classes = [
            ('Apple___healthy', ['apple', 'healthy', 'good', 'normal']),
            ('Apple___Apple_scab', ['apple', 'scab', 'disease', 'fungal']),
            ('Tomato___Bacterial_spot', ['tomato', 'bacterial', 'spot', 'disease']),
            ('Corn___Gray_leaf_spot', ['corn', 'gray', 'leaf', 'spot', 'disease']),
            ('Potato___Early_blight', ['potato', 'early', 'blight', 'disease'])
        ]
        
        # Find matches
        matches = []
        for class_name, keywords in fallback_classes:
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                matches.append({
                    'class_name': class_name,
                    'distance': 1.0 - (score / len(keywords)),  # Lower distance = better match
                    'rank': len(matches) + 1
                })
        
        # Sort by distance and limit results
        matches.sort(key=lambda x: x['distance'])
        return matches[:k]
    
    def get_all_classes(self) -> List[str]:
        """Get all available class names."""
        return self.class_names.copy()
    
    def is_available(self) -> bool:
        """Check if retriever is fully available."""
        return self.index is not None and self.embeddings is not None

if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("Apple___Cedar_apple_rust")
    print(results)