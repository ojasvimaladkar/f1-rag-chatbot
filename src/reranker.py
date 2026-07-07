from sentence_transformers import CrossEncoder
from pathlib import Path

# This cross-encoder model reads query + chunk together
# and outputs a single relevance score
# Much more accurate than bi-encoder similarity
# Downloads ~80MB on first run, cached after
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Re-ranks retrieved chunks using a cross-encoder model.
    
    Why this matters:
    - Bi-encoder (FAISS) embeds query and chunks separately → fast but approximate
    - Cross-encoder reads query+chunk together → slower but much more accurate
    - We use bi-encoder to get 10 candidates, cross-encoder to pick best 3-5
    - This two-stage pipeline is used by Google, Cohere, and most serious RAG systems
    """

    def __init__(self):
        self.model = None

    def load(self):
        """Load the cross-encoder model."""
        print(f"Loading re-ranker model: {RERANKER_MODEL}")
        print("(Downloads ~80MB on first run, cached after)")
        self.model = CrossEncoder(RERANKER_MODEL)
        print("Re-ranker ready.")

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        Re-rank chunks by relevance to query.
        
        How it works:
        1. Create pairs of (query, chunk_text) for every chunk
        2. Feed all pairs through the cross-encoder simultaneously
        3. Get a relevance score for each pair
        4. Sort by score and return top_k
        
        Args:
            query: the user's question
            chunks: list of chunk dicts from hybrid search
            top_k: how many to return after re-ranking
            
        Returns:
            top_k most relevant chunks, sorted by relevance
        """
        if not chunks:
            return []

        # Build (query, chunk_text) pairs
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # Score all pairs — cross encoder reads both together
        scores = self.model.predict(pairs)

        # Attach scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by rerank score descending and return top_k
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from vector_store import HybridVectorStore

    # Load vector store
    store = HybridVectorStore()
    store.load()

    # Load reranker
    reranker = Reranker()
    reranker.load()

    query = "Who won the 2023 Formula 1 drivers championship?"

    print(f"\nQuery: {query}")
    print("\n--- Before Re-ranking (top 10 from hybrid search) ---")
    candidates = store.search(query, top_k=10)
    for i, chunk in enumerate(candidates):
        print(f"{i+1}. [{chunk['source']}] score={chunk['search_score']:.4f} | {chunk['text'][:100]}...")

    print("\n--- After Re-ranking (top 5) ---")
    reranked = reranker.rerank(query, candidates, top_k=5)
    for i, chunk in enumerate(reranked):
        print(f"{i+1}. [{chunk['source']}] rerank_score={chunk['rerank_score']:.4f} | {chunk['text'][:100]}...")