import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from pathlib import Path

def create_vector_db(knowledge_file, output_dir="data/embeddings"):
    """Embed knowledge base into FAISS vector database."""
    # Initialize embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Read knowledge base
    with open(knowledge_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into documents (one per class)
    documents = text.strip().split("\n")
    texts = [doc.split(": ", 1)[1] for doc in documents]
    metadata = [{"class_name": doc.split(": ", 1)[0]} for doc in documents]
    
    # Generate embeddings
    embeddings = embedder.embed_documents(texts)
    
    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype="float32"))
    
    # Save index and metadata
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        for meta in metadata:
            f.write(f"{meta['class_name']}\n")
    
    print(f"Vector database created at {output_dir}")

if __name__ == "__main__":
    knowledge_file = "data/knowledge_base.txt"
    create_vector_db(knowledge_file)