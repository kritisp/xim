from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load a multilingual model to handle conceptual matching across languages globally
model = None

@app.on_event("startup")
def load_model():
    global model
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

class InputText(BaseModel):
    text: str

@app.post("/embed")
def embed(data: InputText):
    embedding = model.encode([data.text])[0].tolist()
    return {"embedding": embedding}