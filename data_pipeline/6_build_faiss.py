import numpy as np
import faiss
import os

def build_faiss_index(
    input_npy="title_embeddings.npy", 
    output_bin="faiss_index.bin"
):
    """
    Step 7: Builds and saves a highly optimized FAISS Index for similarity search 
    using the generated PRGI title embeddings array.
    """
    print(f"Loading embeddings from '{input_npy}'...")
    try:
        embeddings = np.load(input_npy)
    except FileNotFoundError:
        print(f"Error: {input_npy} not found. Generate embeddings first.")
        return

    # Ensure precision format for FAISS
    # We must explicitly cast to float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    total_vectors = embeddings.shape[0]
    
    print(f"Loaded {total_vectors} embeddings of dimension {dimension}.")

    # L2 Normalization required for Cosine Similarity when using IndexFlatIP
    print("Applying L2 Normalization for Cosine Similarity Mapping...")
    faiss.normalize_L2(embeddings)

    # Initialize IndexFlatIP
    # FlatIP calculates inner product, which = cosine similarity on normalized vectors.
    print(f"Building FAISS IndexFlatIP({dimension})...")
    index = faiss.IndexFlatIP(dimension)

    # Add normalized vectors to the index
    print(f"Adding {total_vectors} normalized embedding vectors into the index...")
    index.add(embeddings)

    print(f"Index built. Total vectors in FAISS index: {index.ntotal}")

    # Save to disk
    print(f"Writing index binary file '{output_bin}' to disk...")
    faiss.write_index(index, output_bin)
    
    print("FAISS serialization complete.")

if __name__ == "__main__":
    build_faiss_index()
