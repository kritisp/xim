from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rules import check_rules
from similarity import compute_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TitleInput(BaseModel):
    title: str

@app.post("/verify")
async def verify_title(data: TitleInput):

    title = data.title

    # Step 1 — Rules Check (Prefix, Disallowed Words, Periodicity)
    rule_result = check_rules(title)
    if rule_result["blocked"]:
        return {
            "title": title,
            "status": "Rejected",
            "reason": rule_result["reason"],
            "similarity_score": 100,
            "verification_probability": 0
        }

    # Step 2 — Combination Check
    from similarity import check_combination
    combo_result = check_combination(title)
    if combo_result["blocked"]:
        return {
            "title": title,
            "status": "Rejected",
            "reason": combo_result["reason"],
            "similarity_score": 100,
            "verification_probability": 0
        }

    # Step 3 — Similarity Calculation (Semantic + Phonetic)
    similarity_score = compute_similarity(title)

    # Step 4 — Verification Probability Calculation
    probability = max(0, 100 - similarity_score)
    
    # Let's say a threshold of 80% means auto-reject
    status = "Approved" if probability > 20 else "Rejected"
    reason = "Title is unique and follows guidelines" if status == "Approved" else f"Title is too similar to existing titles ({similarity_score:.2f}% match)"

    if status == "Approved":
        from database import db
        # Add to memory FAISS index AND SQLite database
        db.add_title(title)

    return {
        "title": title,
        "status": status,
        "reason": reason,
        "similarity_score": round(similarity_score, 2),
        "verification_probability": round(probability, 2)
    }
