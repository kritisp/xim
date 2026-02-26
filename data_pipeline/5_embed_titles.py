import pandas as pd
import numpy as np
import json
import torch
from sentence_transformers import SentenceTransformer
import os

def create_embeddings(
    input_csv="combined_preprocessed.csv",
    model_path="paraphrase-multilingual-MiniLM-L12-v2", # Changed default model_path
    output_npy="title_embeddings.npy",
    output_json="title_ids.json"
):
    """
    Loads the NLP model and generates embeddings for every title in the preprocessed CSV.
    """
    # Try to load local model, fallback to HuggingFace
    print(f"Loading SentenceTransformer Model...")
    try:
        model = SentenceTransformer("trained-title-model/")
        print("Loaded customized PRGI trained-title-model/")
    except Exception:
        print("Custom model missing/failed, falling back to base 'paraphrase-multilingual-MiniLM-L12-v2'")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print(f"Loading processed dataset '{input_csv}'...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found.")
        return

    # Extract target titles to embed
    # Priority: Cleaned English -> Cleaned Hindi -> Raw English
    def get_best_target_string(row):
        if pd.notna(row.get("title_en_clean")) and str(row["title_en_clean"]).strip():
            return str(row["title_en_clean"])
        elif pd.notna(row.get("title_hi_clean")) and str(row["title_hi_clean"]).strip():
            return str(row["title_hi_clean"])
        return str(row.get("Title Name (English)", "unknown_title"))

    print("Extracting strings to embed...")
    targets = df.apply(get_best_target_string, axis=1).tolist()
    
    # Track the metadata / ids for searching
    title_metadata = []
    print("Extracting metadata...")
    for idx, row in df.iterrows():
        title_metadata.append({
            "idx": idx,
            "original_english": str(row.get("Title Name (English)", "")),
            "original_hindi": str(row.get("Hindi Title", "")),
            "state": str(row.get("State", "")),
            "periodicity": str(row.get("Periodicity", ""))
        })

    # Encode in batches to save memory
    print(f"Generating embeddings for {len(targets)} titles...")
    # This process handles batching and converts to numpy automatically
    embeddings = model.encode(targets, batch_size=256, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Save outputs
    print(f"Saving embeddings array to {output_npy} (Shape: {embeddings.shape})...")
    np.save(output_npy, embeddings)

    print(f"Saving Title metadata to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(title_metadata, f, indent=4)

    print("Embedding Generation Complete.")

if __name__ == "__main__":
    create_embeddings()
