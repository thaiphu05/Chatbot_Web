from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(os.getenv("MODEL_PATH"))
    
    def encode(self, data: str):
        return self.model.encode(data).tolist()

    