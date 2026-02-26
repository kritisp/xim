import json
import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def train_model(
    model_name="paraphrase-multilingual-MiniLM-L12-v2", 
    data_path="training_pairs.json", 
    output_dir="trained-title-model", 
    epochs=2, 
    batch_size=16
):
    """
    Step 5: Fine-Tunes a Multilingual Sentence Transformer model on the PRGI datasets.
    """
    print(f"Loading base model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # Initialize Model for Semantic Similarity Learning
    model = SentenceTransformer(model_name, device=device)

    # 1. Load the generated Training Pairs JSON
    print(f"Loading training data from {data_path}...")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Generate pairs first.")
        return

    train_examples = []
    
    # 2. Add Labelled Examples based on Cosine Similarity Targets
    # Positives = 1.0 similarity target
    for pair in data.get("positive", []):
        if len(pair) == 2:
            train_examples.append(InputExample(texts=[pair[0], pair[1]], label=1.0))
            
    # Weak Positives = 0.8 similarity target (similar concepts, differing words)
    for pair in data.get("weak_positive", []):
        if len(pair) == 2:
            train_examples.append(InputExample(texts=[pair[0], pair[1]], label=0.8))
            
    # Negatives = 0.0 similarity target
    for pair in data.get("negative", []):
        if len(pair) == 2:
            train_examples.append(InputExample(texts=[pair[0], pair[1]], label=0.0))

    if not train_examples:
        print("No training examples found in JSON.")
        return

    print(f"Loaded {len(train_examples)} input pairs for training.")

    # 3. Create DataLoader and Training Loss Generator
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    print(f"Starting Fine-Tuning for {epochs} Epoch(s)...")
    # 4. Train
    print("Starting Fine-Tuning for {} Epoch(s)...".format(epochs))
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True
    )

    # 5. Save the tuned model locally
    print(f"Training complete. Saving fine-tuned model to {output_dir}/")
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"Successfully saved to {output_dir}. You can now load this model in your pipeline.")

if __name__ == "__main__":
    train_model()
