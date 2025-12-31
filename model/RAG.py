import os
from .Embedding import EmbeddingModel
import numpy as np
from typing import List, Tuple


class RAG:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.data_system = os.getenv("DATA_RETRIEVEL")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    async def retrieve(self, data: str, embedding_model: EmbeddingModel, chunks: List[dict], top_k: int = 5) -> List[Tuple[dict, float]]:
        query_embedding = self.embedding_model.encode(data)
        
        results = []
        for chunk in chunks:
            chunk_embedding = chunk.get('embedding', [])
            if chunk_embedding:
                similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                results.append((chunk, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

        

