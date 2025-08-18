import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
import os

class Retriever:
    def __init__(self, index_path="data/embeddings/faiss_index.bin", metadata_path="data/embeddings/metadata.txt"):
        """Initialize FAISS retriever."""
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            self.metadata = [line.strip() for line in f]
    
    def retrieve(self, query, k=1):
        """Retrieve top-k relevant documents for the query."""
        query_embedding = np.array(self.embedder.embed_query(query), dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return [{"class_name": self.metadata[idx], "distance": distances[0][i]} for i, idx in enumerate(indices[0])]

if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("Apple___Cedar_apple_rust")
    print(results)