import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
import json
import os

warnings.filterwarnings('ignore')

class TitleSearchEngine:
    def __init__(self, 
                 index_path="faiss_index.bin", 
                 model_path="trained-title-model", 
                 metadata_path="title_ids.json"):
        """
        Step 8: Load the FAISS Index, the Fine-Tuned NLP Encoder, and the Memory mapping ID dictionary 
        to execute semantic clustering.
        """
        print(f"Loading FAISS index from {index_path}...")
        try:
            self.index = faiss.read_index(index_path)
        except Exception as e:
            raise Exception(f"Failed to load FAISS index: {e}")

        print(f"Loading Sentence Model from {model_path}...")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Defaulting to base paraphrase-multilingual-MiniLM-L12-v2.")
            model_path = "paraphrase-multilingual-MiniLM-L12-v2"
        self.model = SentenceTransformer(model_path)
        
        print(f"Loading Metadata from {metadata_path}...")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load metadata json: {e}")

    def clean_query(self, query):
        import re
        q = query.lower()
        q = re.sub(r'[^\w\s]', '', q)
        q = re.sub(r'\s+', ' ', q).strip()
        return q

    def search_title(self, query, top_k=10):
        # 1. Clean the incoming string
        clean_q = self.clean_query(query)
        if not clean_q:
            return []
            
        # 2. Embed the Query
        # Reshape for FAISS inner-product processing
        emb = self.model.encode([clean_q])
        emb = np.array(emb, dtype=np.float32)
        faiss.normalize_L2(emb)

        # 3. Search the pre-built FAISS memory maps
        distances, indices = self.index.search(emb, top_k)
        
        # 4. Return formatted response combining distances and JSON IDs
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1: 
                continue # FAISS pad
                
            # dist represents dot product -> equivalent to cosine similarity here as everything is L2 normalized
            percentage_score = float(dist) * 100
            
            # Prevent floating point overflows past 100%
            percentage_score = min(100.0, percentage_score)
            
            meta = self.metadata[idx]
            results.append({
                "rank": rank + 1,
                "score": round(percentage_score, 2),
                "english_title": meta.get("original_english"),
                "hindi_title": meta.get("original_hindi"),
                "state": meta.get("state"),
                "periodicity": meta.get("periodicity")
            })
            
        return results

# Exposing as an easy function definition for global scripts to import and execute as requested
_engine_instance = None
def search_title(query, top_k=10):
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TitleSearchEngine()
    
    return _engine_instance.search_title(query, top_k=top_k)

if __name__ == "__main__":
    # Test execution
    print("\n--- Testing FAISS Search ---")
    results = search_title("Morning Chronicle", top_k=3)
    for r in results:
        print(f"Score: {r['score']}% | ENG: {r['english_title']} | HI: {r['hindi_title']}")
