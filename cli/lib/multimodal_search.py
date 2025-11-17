from PIL import Image
import os
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import PROJECT_ROOT, CACHE_DIR, load_movies
import heapq
import numpy as np

class MultiModalSearch:
    def __init__(self,documents,  model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.clip_embeddings_path = os.path.join(CACHE_DIR, "clip_embeddings.npy")
        self.text_embeddings = None

    def build_embeddings(self) -> None:
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        with open(self.clip_embeddings_path, 'wb') as f:
            np.save(f, self.text_embeddings)

    def load_or_create_embeddings(self) -> None:
        if not os.path.exists(self.clip_embeddings_path):
            return self.build_embeddings()
        with open(self.clip_embeddings_path, "rb") as f:
            self.text_embeddings = np.load(f)
         
    def embed_image(self, image_path: str):
        if image_path.isspace() or len(image_path) == 0:
            raise ValueError("Empty string")
        img = Image.open(image_path)
        embedding = self.model.encode([img])
        return embedding[0]
    
    def search_with_image(self, image_path: str):
        img_embedding = self.embed_image(image_path)
        similarity_list = []
        for text_embedding, doc in zip(self.text_embeddings, self.documents):
            similarity = cosine_similarity(img_embedding, text_embedding)
            similarity_list.append({
                "title" : doc['title'],
                "description" : doc['description'],
                "score" : similarity
            })
        
        most_similar = heapq.nlargest(5, similarity_list, key=lambda item: item['score'])
        return most_similar

def verify_image_embedding(image: str):
    image_path = os.path.join(PROJECT_ROOT, image)
    multimodal_search = MultiModalSearch()
    embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image: str):
    documents = load_movies()
    image_path = os.path.join(PROJECT_ROOT, image)
    multimodal_search = MultiModalSearch(documents)
    multimodal_search.load_or_create_embeddings()
    results = multimodal_search.search_with_image(image_path)
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['title']} (similarity: {res['score']:.3f})\n   {res['description'][:100]}")

