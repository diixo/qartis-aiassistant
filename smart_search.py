
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path


class SmartSearch_FAISS:

    def __init__(self, modelname="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(modelname)
        self.index = None


    def open_file(self, filepath="embeddings.bin"):
        path = Path(filepath)
        if path.exists():
            self.index = faiss.read_index(filepath)
            return True
        return False


    def texts_to_vector(self, texts):
        if self.model: 
            embeddings = self.model.encode(texts)
            return embeddings, embeddings.shape[1]
        return None, 0


    def add_texts_to_index(self, texts: list) -> bool:
        vector, dimension = self.texts_to_vector(texts)
        if not self.index and dimension > 0:
            self.index = faiss.IndexFlatL2(dimension)

        if (self.index is not None) and (vector is not None):
            self.index.add(vector)
            return True
        return False


    def save_index(self, filepath="embeddings.bin"):
        if self.index:
            faiss.write_index(self.index, filepath)


    def add_str_to_index(self, text: str) -> bool:
        return self.add_texts_to_index([ text ])


    def search(self, query_text: str, k: int = 20):
        if self.model:
            query_embedding = self.model.encode([query_text])

            distances, indices = self.index.search(query_embedding, k)

            if indices.size > 0:
                return indices[0].tolist(), distances[0].tolist()
        return [], []
