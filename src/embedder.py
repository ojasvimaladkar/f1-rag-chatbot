import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROCESSED_DATA_PATH = Path("data/processed")
EMBEDDINGS_PATH = Path("embeddings")
EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)

# This is the embedding model we're using
# all-MiniLM-L6-v2 is small (80MB), fast, and good quality
# It converts text into 384-dimensional vectors
# Downloads automatically on first run, cached after that
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    """Load our processed chunks from disk."""
    path = PROCESSED_DATA_PATH / "chunks.json"
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def generate_embeddings(chunks: list[dict]) -> np.ndarray:
    """
    Convert each chunk's text into an embedding vector.
    
    What's happening here:
    - We load a pretrained model (downloads ~80MB first time)
    - We pass all chunk texts through the model
    - Each text becomes a 384-dimensional vector
    - Similar texts will have similar vectors
    - We return a 2D array: shape (301, 384)
      → 301 chunks, each with 384 numbers
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    print("(This downloads ~80MB on first run, then cached forever)")
    model = SentenceTransformer(MODEL_NAME)

    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")
    
    # batch_size=32 means we process 32 chunks at a time
    # show_progress_bar gives us a nice progress indicator
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"  → {embeddings.shape[0]} chunks")
    print(f"  → {embeddings.shape[1]} dimensions per chunk")
    return embeddings


def save_embeddings(embeddings: np.ndarray):
    """Save embeddings as a numpy file."""
    path = EMBEDDINGS_PATH / "embeddings.npy"
    np.save(path, embeddings)
    print(f"Saved embeddings → {path}")


def main():
    print("=" * 50)
    print("Embedding Generation Starting")
    print("=" * 50)

    chunks = load_chunks()
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings)

    print("=" * 50)
    print("Done! Embeddings saved.")
    print("=" * 50)


if __name__ == "__main__":
    main()