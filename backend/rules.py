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

    # Rule 1 — Disallowed words
    for word in DISALLOWED_WORDS:
        # Check as whole word
        if f" {word} " in f" {t} ":
            return {"blocked": True, "reason": f"Contains disallowed word: '{word}'"}

    # Rule 2 — Disallowed prefixes/suffixes
    for prefix in DISALLOWED_PREFIXES:
        if t.startswith(prefix + " "):
            return {"blocked": True, "reason": f"Disallowed prefix: '{prefix}'"}
        if t.endswith(" " + prefix):
            return {"blocked": True, "reason": f"Disallowed suffix: '{prefix}'"}

    # Rule 3 — Periodicity + Existing Title
    # Extract potential periodicity words from the title
    found_periodicity = [p for p in PERIODICITY_WORDS if f" {p} " in f" {t} " or t.startswith(p + " ") or t.endswith(" " + p)]
    
    if found_periodicity:
        # Remove the periodicity word and check if it resembles an existing title
        cleaned_t = t
        for p in found_periodicity:
            cleaned_t = cleaned_t.replace(p, "").strip()
            
        # Use pre-computed O(1) set — no per-request rebuild
        if cleaned_t in db.get_titles_set():
            return {"blocked": True, "reason": f"Cannot form a new title by adding periodicity '{found_periodicity[0]}' to existing title"}

    return {"blocked": False, "reason": ""}