import json

# Load rules dynamically from JSON
import os
rules_path = os.path.join(os.path.dirname(__file__), "../data/disallowed_rules.json")
try:
    with open(rules_path, "r", encoding="utf-8") as file:
        rules_data = json.load(file)
        DISALLOWED_WORDS = rules_data.get("disallowed_words", [])
        DISALLOWED_PREFIXES = rules_data.get("disallowed_prefixes", [])
        PERIODICITY_WORDS = rules_data.get("periodicity_words", [])
except FileNotFoundError:
    DISALLOWED_WORDS = []
    DISALLOWED_PREFIXES = []
    PERIODICITY_WORDS = []

from database import db

def check_rules(title):
    t = title.lower()
    details = []

    # Rule 1 — Disallowed words
    for word in DISALLOWED_WORDS:
        if f" {word} " in f" {t} ":
            details.append({
                "check_type": "disallowed_word",
                "description": f"Contains disallowed word '{word}'",
                "matched_word": word,
                "matched_title": None,
                "score": None
            })

    # Rule 2 — Disallowed prefixes/suffixes
    for prefix in DISALLOWED_PREFIXES:
        if t.startswith(prefix + " "):
            details.append({
                "check_type": "disallowed_prefix",
                "description": f"Title starts with disallowed prefix '{prefix}'",
                "matched_word": prefix,
                "matched_title": None,
                "score": None
            })
        if t.endswith(" " + prefix):
            details.append({
                "check_type": "disallowed_suffix",
                "description": f"Title ends with disallowed suffix '{prefix}'",
                "matched_word": prefix,
                "matched_title": None,
                "score": None
            })

    # Rule 3 — Periodicity + Existing Title
    found_periodicity = [p for p in PERIODICITY_WORDS if f" {p} " in f" {t} " or t.startswith(p + " ") or t.endswith(" " + p)]
    
    if found_periodicity:
        cleaned_t = t
        for p in found_periodicity:
            cleaned_t = cleaned_t.replace(p, "").strip()
            
        if cleaned_t in db.get_titles_set():
            details.append({
                "check_type": "periodicity",
                "description": f"Adding periodicity word '{found_periodicity[0]}' to existing title '{cleaned_t}'",
                "matched_word": found_periodicity[0],
                "matched_title": cleaned_t,
                "score": None
            })

    if details:
        # Use the first detail as the primary reason
        return {"blocked": True, "reason": details[0]["description"], "details": details}

    return {"blocked": False, "reason": "", "details": []}