import json
import faiss
import numpy as np
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
from sentence_transformers import SentenceTransformer

# Load model ONCE in-process (no HTTP round-trip = instant embeddings)
print("Loading SentenceTransformer model in-process...")
_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded.")
# --- SQLAlchemy Setup ---
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/titles.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TitleRecord(Base):
    __tablename__ = "titles"
    id = Column(Integer, primary_key=True, index=True)
    title_name = Column(String, unique=True, index=True)
    status = Column(String, default="Approved")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- FAISS TitleDatabase ---
class TitleDatabase:
    def __init__(self):
        self.titles = []
        self._titles_set = set()  # Pre-computed lowercase set for O(1) lookups
        self.dimension = 384 # paraphrase-multilingual-MiniLM-L12-v2 dim
        self.faiss_path = os.path.join(os.path.dirname(__file__), "../data_pipeline/faiss_index.bin")
        self.ids_path = os.path.join(os.path.dirname(__file__), "../data_pipeline/title_ids.json")
        
        # Load pre-trained FAISS index if available
        if os.path.exists(self.faiss_path) and os.path.exists(self.ids_path):
            print(f"Loading Pre-Trained FAISS Index from {self.faiss_path}...")
            self.index = faiss.read_index(self.faiss_path)
            with open(self.ids_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                self.titles = [r["original_english"] for r in records if "original_english" in r]
            self._titles_set = {t.lower() for t in self.titles}
            print(f"Successfully loaded {len(self.titles)} titles into memory.")
        else:
            print("Warning: FAISS index not found. Generating empty index.")
            self.index = faiss.IndexFlatIP(self.dimension)

    def _get_embedding(self, text):
        try:
            # In-process embedding — no HTTP overhead
            emb = _model.encode([text])[0].astype(np.float32)
            faiss.normalize_L2(emb.reshape(1, -1))
            return emb
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    def load_from_db(self):
        db_session = SessionLocal()
        records = db_session.query(TitleRecord).all()
        
        # Merge new SQL titles into FAISS if they don't exist in the loaded JSON yet
        new_insertions = 0
        for rec in records:
            if rec.title_name.lower() not in self._titles_set:
                self._add_to_faiss(rec.title_name)
                new_insertions = new_insertions + 1
                
        if new_insertions > 0:
            print(f"Injected {new_insertions} new SQL approvals into FAISS index.")
            
        db_session.close()

    def _add_to_faiss(self, title):
        if title.lower() not in self._titles_set:
            self.titles.append(title)
            self._titles_set.add(title.lower())
            emb = self._get_embedding(title)
            self.index.add(emb.reshape(1, -1))

    def add_title(self, title):
        # 1. Add to SQLite
        try:
            db_session = SessionLocal()
            record = TitleRecord(title_name=title, status="Approved")
            db_session.add(record)
            db_session.commit()
            db_session.close()
            # 2. Add to FAISS index in memory
            self._add_to_faiss(title)
        except Exception as e:
            print(f"Failed to insert title into DB: {e}")

    def search_similar(self, title, top_k=5):
        if len(self.titles) == 0:
            return []
        
        emb = self._get_embedding(title)
        distances, indices = self.index.search(emb.reshape(1, -1), top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                score = float(distances[0][i]) * 100 
                score = min(100.0, score)
                results.append((self.titles[idx], score))
                
        return results

    def get_all_titles(self):
        return self.titles

    def get_titles_set(self):
        return self._titles_set

# Global instance
db = TitleDatabase()

def load_existing_titles():
    # Only called once on startup to seed FAISS from SQLite
    db.load_from_db()
    return db.get_all_titles()

