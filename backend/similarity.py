import numpy as np
import jellyfish

from database import db

def phonetic_similarity(a, b):
    # Jaro-Winkler for character similarity + Metaphone check
    jw_score = jellyfish.jaro_winkler_similarity(a.lower(), b.lower()) * 100
    
    # If the words sound exactly alike phonetically (e.g., Namaskar vs Namascar)
    if jellyfish.metaphone(a) == jellyfish.metaphone(b):
        return max(jw_score, 100.0) # Exact phonetic match is 100%
    return jw_score

def check_combination(new_title):
    t = new_title.lower()
    
    # Use pre-computed O(1) lookup set — no per-request rebuild
    existing_set = db.get_titles_set()
    
    # Space-separated combinations check (O(1) lookups only)
    words = t.split()
    if len(words) == 2:
        if words[0] in existing_set and words[1] in existing_set:
             return {"blocked": True, "reason": f"Combination of existing titles: '{words[0]}' and '{words[1]}'"}

    return {"blocked": False}

def compute_similarity(title):
    max_score = 0.0

    # 1. Semantic Similarity Search via FAISS (Top 5 matches)
    semantic_results = db.search_similar(title, top_k=5)
    
    # Analyze the top matches
    for existing, sem_score in semantic_results:
        # Prevent matching against itself if already in the index
        if existing.lower() == title.lower():
            continue
            
        # String/Phonetic similarity for the closest semantic hits
        phon_score = phonetic_similarity(title, existing)
        
        # We take the maximum of semantic meaning or phonetic spelling
        combined = max(sem_score, phon_score)
        if combined > max_score:
            max_score = combined

    return max_score
