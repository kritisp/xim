import numpy as np
import jellyfish

from database import db

def phonetic_similarity(a, b):
    # Jaro-Winkler for character similarity + Metaphone check
    jw_score = jellyfish.jaro_winkler_similarity(a.lower(), b.lower()) * 100
    
    # If the words sound exactly alike phonetically (e.g., Namaskar vs Namascar)
    is_metaphone_match = jellyfish.metaphone(a) == jellyfish.metaphone(b)
    if is_metaphone_match:
        return max(jw_score, 100.0), "Metaphone exact match"
    return jw_score, "Jaro-Winkler"

def check_combination(new_title):
    t = new_title.lower()
    details = []
    
    # Use pre-computed O(1) lookup set
    existing_set = db.get_titles_set()
    
    # Space-separated combinations check
    words = t.split()
    if len(words) == 2:
        if words[0] in existing_set and words[1] in existing_set:
            details.append({
                "check_type": "combination",
                "description": f"Title is a combination of two existing titles: '{words[0]}' + '{words[1]}'",
                "matched_title": f"{words[0]} + {words[1]}",
                "score": 100,
                "method": "Exact word match"
            })
            return {"blocked": True, "reason": details[0]["description"], "details": details}

    return {"blocked": False, "details": []}

def compute_similarity(title):
    max_score = 0.0
    details = []

    # 1. Semantic Similarity Search via FAISS (Top 5 matches)
    semantic_results = db.search_similar(title, top_k=5)
    
    # Analyze the top matches
    for existing, sem_score in semantic_results:
        # Prevent matching against itself
        if existing.lower() == title.lower():
            continue

        # Add semantic match detail if score is significant (> 40%)
        if sem_score > 40:
            details.append({
                "check_type": "semantic",
                "description": f"Semantically similar to existing title '{existing}'",
                "matched_title": existing,
                "score": round(sem_score, 2),
                "method": "FAISS cosine similarity"
            })
            
        # String/Phonetic similarity for the closest semantic hits
        phon_score, phon_method = phonetic_similarity(title, existing)
        
        # Add phonetic match detail if score is significant (> 60%)
        if phon_score > 60:
            details.append({
                "check_type": "phonetic",
                "description": f"Phonetically similar to existing title '{existing}'",
                "matched_title": existing,
                "score": round(phon_score, 2),
                "method": phon_method
            })
        
        # We take the maximum of semantic meaning or phonetic spelling
        combined = max(sem_score, phon_score)
        if combined > max_score:
            max_score = combined

    # Sort details by score descending so the strongest matches appear first
    details.sort(key=lambda d: d.get("score", 0) or 0, reverse=True)

    return max_score, details
