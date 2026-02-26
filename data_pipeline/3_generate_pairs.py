import pandas as pd
import json
import random
import itertools

def generate_pairs(input_csv="combined_preprocessed.csv", output_json="training_pairs.json"):
    """
    Step 4: Generates positive, weak positive, and negative title pairs to fine-tune 
    the SentenceTransformer model for Semantic Similarity.
    """
    try:
        df = pd.read_csv(input_csv).dropna(subset=["title_en_clean"])
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Run preprocessing first.")
        return

    titles_en = df["title_en_clean"].tolist()
    titles_hi = df["title_hi_clean"].dropna().tolist()
    
    positive_pairs = []
    weak_positive_pairs = []
    negative_pairs = []

    # 1. Positive Pairs (Exact conceptual match)
    # a) English <-> Hindi Translations
    for i, row in df.dropna(subset=["title_en_clean", "title_hi_clean"]).iterrows():
        positive_pairs.append([row["title_en_clean"], row["title_hi_clean"]])
    
    # b) Titles differing by City/State suffix (same root)
    # Simulated heuristically for training representation: a base title mapped to base + suffix
    roots = ["the daily chronicle", "morning herald", "new dawn"]
    cities = ["delhi", "mumbai", "bengaluru"]
    for root in roots:
        for city in cities:
            positive_pairs.append([root, f"{root} {city}"])
            
    # c) Spelling / Phonetics (using real string mutations for training)
    # Example: namaskar <-> namascar
    spelling_vars = [
        ("namaskar", "namascar"), ("news", "nws"), 
        ("samachar", "samacar"), ("daily", "daili")
    ]
    positive_pairs.extend([list(pair) for pair in spelling_vars])

    # 2. Weak Positive Pairs
    # a) Similar conceptual meaning but different lexical words
    conceptual_vars = [
        ("morning herald", "sunrise chronicle"),
        ("morning herald", "dawn dispatch"),
        ("evening express", "dusk dispatch"),
        ("night watch", "midnight observer")
    ]
    weak_positive_pairs.extend([list(pair) for pair in conceptual_vars])
    
    # b) Suffix modification (simulating dropped "today" or "daily")
    for root in roots:
        weak_positive_pairs.append([f"{root} today", root])

    # 3. Negative Pairs
    # a) Random mismatches
    for _ in range(500):
        t1 = random.choice(titles_en) if titles_en else "random test 1"
        t2 = random.choice(titles_en) if titles_en else "random test 2"
        if t1 != t2:
            negative_pairs.append([t1, t2])
            
    # b) Same prefix, entirely different meaning 
    # (e.g., "The Police Chronicle" vs "The Flower Chronicle")
    negative_pairs.append(["the police chronicle", "the flower chronicle"])
    negative_pairs.append(["indian express", "indian agriculture"])

    dataset = {
        "positive": positive_pairs,
        "weak_positive": weak_positive_pairs,
        "negative": negative_pairs
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Generated {len(positive_pairs)} positive, {len(weak_positive_pairs)} weak positive, and {len(negative_pairs)} negative pairs.")
    print(f"Saved dataset to {output_json}")

if __name__ == "__main__":
    generate_pairs()
