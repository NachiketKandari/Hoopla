import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from collections import defaultdict, Counter
import math
import re
from cli.lib.search_utils import CACHE_DIR, PROJECT_ROOT

# Configuration
README_FILES = [
    "README.md",
    "cli/README.md",
    "cli/lib/README.md",
    "app/README.md"
]
INDEX_FILE = os.path.join(CACHE_DIR, "readmes.pkl")
CHUNK_SIZE = 500  # Characters
OVERLAP = 50

class ReadmeRAG:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.model = None
        self.bm25_index = defaultdict(set)
        self.doc_lengths = {}
        self.avg_doc_len = 0
        self.term_freqs = []
        self.idf = {}

    def load_readmes(self):
        documents = []
        for rel_path in README_FILES:
            file_path = os.path.join(PROJECT_ROOT, rel_path)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({"path": rel_path, "content": content})
            else:
                print(f"Warning: {file_path} not found.")
        return documents

    def chunk_text(self, text, source):
        chunks = []
        # Simple sliding window chunking
        for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
            chunk_content = text[i:i + CHUNK_SIZE]
            chunks.append({
                "source": source,
                "content": chunk_content,
                "id": len(self.chunks) + len(chunks)
            })
        return chunks

    def build_index(self):
        print("Building index...")
        docs = self.load_readmes()
        self.chunks = []
        for doc in docs:
            self.chunks.extend(self.chunk_text(doc['content'], doc['path']))

        # Semantic Embeddings
        print("Generating embeddings...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [c['content'] for c in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        # BM25 Indexing
        print("Building BM25 index...")
        total_len = 0
        doc_count = len(self.chunks)
        
        for chunk in self.chunks:
            tokens = self.tokenize(chunk['content'])
            length = len(tokens)
            self.doc_lengths[chunk['id']] = length
            total_len += length
            
            counts = Counter(tokens)
            self.term_freqs.append(counts)
            
            for term in counts:
                self.bm25_index[term].add(chunk['id'])

        self.avg_doc_len = total_len / doc_count if doc_count > 0 else 0
        
        # Calculate IDF
        for term, doc_ids in self.bm25_index.items():
            df = len(doc_ids)
            self.idf[term] = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

        self.save_index()
        print(f"Index built with {len(self.chunks)} chunks.")

    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def save_index(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'bm25_index': self.bm25_index,
                'doc_lengths': self.doc_lengths,
                'avg_doc_len': self.avg_doc_len,
                'term_freqs': self.term_freqs,
                'idf': self.idf
            }, f)

    def load_index(self):
        if not os.path.exists(INDEX_FILE):
            self.build_index()
            return

        with open(INDEX_FILE, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            self.bm25_index = data['bm25_index']
            self.doc_lengths = data['doc_lengths']
            self.avg_doc_len = data['avg_doc_len']
            self.term_freqs = data['term_freqs']
            self.idf = data['idf']
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def bm25_score(self, query, k1=1.5, b=0.75):
        tokens = self.tokenize(query)
        scores = defaultdict(float)
        
        for term in tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            
            for doc_id in self.bm25_index[term]:
                tf = self.term_freqs[doc_id][term]
                doc_len = self.doc_lengths[doc_id]
                
                num = tf * (k1 + 1)
                den = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                scores[doc_id] += idf_val * (num / den)
        
        return scores

    def semantic_search(self, query):
        if self.model is None:
             self.model = SentenceTransformer('all-MiniLM-L6-v2')
             
        query_vec = self.model.encode([query])[0]
        scores = {}
        
        # Cosine similarity
        norm_q = np.linalg.norm(query_vec)
        for i, doc_vec in enumerate(self.embeddings):
            norm_d = np.linalg.norm(doc_vec)
            if norm_d == 0 or norm_q == 0:
                score = 0
            else:
                score = np.dot(query_vec, doc_vec) / (norm_q * norm_d)
            scores[self.chunks[i]['id']] = score
            
        return scores

    def rrf_search(self, query, k=60, limit=5):
        if not self.chunks:
            self.load_index()
            
        bm25_scores = self.bm25_score(query)
        sem_scores = self.semantic_search(query)
        
        # Rank
        bm25_ranked = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True)
        sem_ranked = sorted(sem_scores.keys(), key=lambda x: sem_scores[x], reverse=True)
        
        rrf_scores = defaultdict(float)
        
        for rank, doc_id in enumerate(bm25_ranked):
            rrf_scores[doc_id] += 1 / (k + rank + 1)
            
        for rank, doc_id in enumerate(sem_ranked):
            rrf_scores[doc_id] += 1 / (k + rank + 1)
            
        top_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:limit]
        return [self.chunks[i] for i in top_ids]

    def rewrite_query(self, query, context_chunks, api_key=None):
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY not found.")
            return query

        client = genai.Client(api_key=api_key)
        
        context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['content']}" for c in context_chunks])
        
        prompt = f"""
        You are a helpful assistant for the Hoopla RAG Toolkit.
        Based on the following context from the project documentation, rewrite the user's query to be more specific and technical, suitable for a RAG system search.
        The rewritten query should be a single sentence and should be a direct translation of the user's query into a query that can be used to search the project documentation.
        
        Context:
        {context_text}
        
        User Query: {query}
        
        Rewritten Query:
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return query
