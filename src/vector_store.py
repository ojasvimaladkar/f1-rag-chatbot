import json
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROCESSED_DATA_PATH = Path("data/processed")
EMBEDDINGS_PATH = Path("embeddings")
FAISS_INDEX_PATH = Path("embeddings/faiss_index")
FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"


class HybridVectorStore:
    """
    Combines FAISS (semantic search) with BM25 (keyword search).
    
    At query time:
    1. FAISS finds top-K chunks by semantic similarity
    2. BM25 finds top-K chunks by keyword match
    3. We merge and re-score both lists (Reciprocal Rank Fusion)
    4. Return the best combined results
    """

    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        self.model = None

    def load(self):
        """Load everything from disk."""
        print("Loading vector store...")

        # Load chunks
        with open(PROCESSED_DATA_PATH / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"  Loaded {len(self.chunks)} chunks")

        # Load embeddings
        self.embeddings = np.load(EMBEDDINGS_PATH / "embeddings.npy")
        print(f"  Loaded embeddings: {self.embeddings.shape}")

        # Load embedding model (cached, fast after first run)
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"  Loaded embedding model: {MODEL_NAME}")

        # Build or load FAISS index
        faiss_path = FAISS_INDEX_PATH / "index.faiss"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            print(f"  Loaded FAISS index from disk")
        else:
            self._build_faiss_index()

        # Build BM25 index (always built in memory, fast)
        self._build_bm25_index()

        print("Vector store ready.")

    def _build_faiss_index(self):
        """
        Build FAISS index from embeddings.
        
        IndexFlatIP = Flat index using Inner Product (dot product)
        For normalized vectors, inner product == cosine similarity
        We normalize first, then use IP for fast cosine search
        """
        print("  Building FAISS index...")
        dimension = self.embeddings.shape[1]  # 384

        # Normalize embeddings so inner product = cosine similarity
        normalized = self.embeddings.copy().astype(np.float32)
        faiss.normalize_L2(normalized)

        # Create and populate the index
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(normalized)

        # Save to disk so we don't rebuild every time
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH / "index.faiss"))
        print(f"  FAISS index built and saved ({self.faiss_index.ntotal} vectors)")

    def _build_bm25_index(self):
        """
        Build BM25 index from chunk texts.
        
        BM25 works on tokenized text (list of words).
        We do simple whitespace tokenization here.
        In production you'd use a proper tokenizer.
        """
        print("  Building BM25 index...")
        tokenized = [chunk["text"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built ({len(tokenized)} documents)")

    def _search_faiss(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """
        Search FAISS for semantically similar chunks.
        Returns list of (chunk_index, score) tuples.
        """
        # Embed the query using the same model
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search — returns distances and indices
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # -1 means no result
                results.append((int(idx), float(score)))

        return results

    def _search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """
        Search BM25 for keyword matches.
        Returns list of (chunk_index, score) tuples.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only include if there's an actual match
                results.append((int(idx), float(scores[idx])))

        return results

    def _reciprocal_rank_fusion(
        self,
        faiss_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        k: int = 60
    ) -> list[tuple[int, float]]:
        """
        Merge FAISS and BM25 results using Reciprocal Rank Fusion.
        
        RRF is a simple, effective way to combine ranked lists.
        For each chunk, its RRF score = sum of 1/(k + rank) across all lists.
        k=60 is the standard value from the original RRF paper.
        
        Example:
        Chunk A is rank 1 in FAISS, rank 3 in BM25
        → RRF score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
        
        Chunk B is rank 1 in BM25 only
        → RRF score = 1/(60+1) = 0.0164
        
        Chunk A wins because it appeared in both lists.
        """
        rrf_scores = {}

        for rank, (idx, _) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Main search function. 
        Runs hybrid search and returns top_k chunks with metadata.
        """
        # Get candidates from both search methods
        faiss_results = self._search_faiss(query, top_k=top_k)
        bm25_results = self._search_bm25(query, top_k=top_k)

        # Merge using RRF
        fused_results = self._reciprocal_rank_fusion(faiss_results, bm25_results)

        # Return top_k chunks with their content
        final_results = []
        for idx, score in fused_results[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["search_score"] = score
            final_results.append(chunk)

        return final_results


# ─────────────────────────────────────────
# TEST — Run this file directly to test search
# ─────────────────────────────────────────

if __name__ == "__main__":
    store = HybridVectorStore()
    store.load()

    test_queries = [
        "Who won the 2023 Formula 1 championship?",
        "Tell me about Max Verstappen",
        "What is DRS in Formula 1?",
        "Monaco Grand Prix results"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        results = store.search(query, top_k=3)
        for i, chunk in enumerate(results):
            print(f"\nResult {i+1} (score: {chunk['search_score']:.4f})")
            print(f"Source: {chunk['source']}")
            print(f"Text: {chunk['text'][:200]}...")