import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

"""
retriever.py

This module provides the semantic retrieval capabilities. It loads the pre-computed
embeddings and the filtered corpus. Given a new customer message, it finds the top 3
most similar historical conversations. It also reports if the retrieval confidence 
is low based on a similarity threshold of 0.3.
"""

class SupportRetriever:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        processed_dir = project_root / "data" / "processed"
        
        embeddings_path = processed_dir / "embeddings.npy"
        corpus_path = processed_dir / "corpus.csv"
        
        # Delay loading until needed or load on init
        self.embeddings = np.load(embeddings_path)
        self.corpus_df = pd.read_csv(corpus_path)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.threshold = 0.3

    def get_similar_cases(self, customer_message, top_k=3):
        """
        Retrieves the top_k most similar historical cases for a given message.
        
        Returns:
            list of dicts containing 'conversation_id', 'customer_message', 
            'brand_reply', and 'similarity_score'.
            bool indicating if confidence is low (highest score < 0.3).
        """
        # Encode the query
        query_embedding = self.model.encode([customer_message])
        
        # Compute cosine similarities between query and all corpus embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get indices of top_k highest scores
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved_cases = []
        best_score = float('-inf')
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score > best_score:
                best_score = score
                
            case_data = self.corpus_df.iloc[idx]
            retrieved_cases.append({
                "conversation_id": case_data["conversation_id"],
                "customer_message": case_data["customer_message"],
                "brand_reply": case_data["brand_reply"],
                "similarity_score": score
            })
            
        low_confidence = best_score < self.threshold
        
        return retrieved_cases, low_confidence

# Singleton-like instantiation to avoid reloading the model and embeddings multiple times
_retriever_instance = None

def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SupportRetriever()
    return _retriever_instance
