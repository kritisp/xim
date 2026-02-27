from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rules import check_rules
from similarity import compute_similarity, check_combination

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
    all_details = []

    # Step 1 — Rules Check (Prefix, Disallowed Words, Periodicity)
    rule_result = check_rules(title)
    all_details.extend(rule_result.get("details", []))

    if rule_result["blocked"]:
        # Rule-based rejection — return immediately (no slow similarity needed)
        return {
            "title": title,
            "status": "Rejected",
            "reason": rule_result["reason"],
            "similarity_score": 100,
            "verification_probability": 0,
            "details": all_details
        }

    # Step 2 — Combination Check
    combo_result = check_combination(title)
    all_details.extend(combo_result.get("details", []))

    if combo_result.get("blocked"):
        return {
            "title": title,
            "status": "Rejected",
            "reason": combo_result["reason"],
            "similarity_score": 100,
            "verification_probability": 0,
            "details": all_details
        }

    # Step 3 — Similarity Calculation (Semantic + Phonetic)
    # Only runs if rules/combination passed — this is the slow step (model service call)
    similarity_score, similarity_details = compute_similarity(title)
    all_details.extend(similarity_details)

    # Step 4 — Verification Probability Calculation
    probability = max(0, 100 - similarity_score)
    
    # Threshold: 50% similarity means auto-reject
    status = "Approved" if similarity_score < 50 else "Rejected"
    reason = "Title is unique and follows guidelines" if status == "Approved" else f"Title is too similar to existing titles ({similarity_score:.2f}% match)"

    if status == "Approved":
        from database import db
        db.add_title(title)

    return {
        "title": title,
        "status": status,
        "reason": reason,
        "similarity_score": round(similarity_score, 2),
        "verification_probability": round(probability, 2),
        "details": all_details
    }
