import os
from groq import Groq
from dotenv import load_dotenv
from vector_store import HybridVectorStore
from reranker import Reranker

load_dotenv()

# How many chunks to retrieve and re-rank
TOP_K_RETRIEVAL = 10   # retrieve this many from hybrid search
TOP_K_RERANK = 5       # keep this many after re-ranking

# System prompt — this is prompt engineering
# It tells the LLM exactly how to behave
# Notice we're very explicit: answer ONLY from context, cite sources, admit uncertainty
SYSTEM_PROMPT = """You are an F1 data assistant. Your ONLY job is to answer questions using the context documents provided to you.

STRICT RULES:
1. ONLY use information explicitly stated in the provided context
2. If the context contains the answer, give it directly and cite the source
3. If the context does NOT contain the answer, say exactly: "I don't have that specific information in my knowledge base."
4. NEVER add facts from your own training data
5. NEVER say things like "generally" or "typically" — only state what the documents say
6. Always end your answer with: (Source: [source name])

FORMAT:
- Be concise and direct
- One paragraph maximum for simple facts
- Bullet points for lists
- Always cite which document the information came from"""

def expand_query(query: str, groq_client) -> str:
    """
    Rewrite the user query to be more search-friendly.
    
    "Who came second" → "second place 2021 Formula 1 drivers championship standings"
    
    This helps both FAISS and BM25 find more relevant chunks.
    """
    prompt = f"""Rewrite this question as a search query that would find relevant F1 documents.
Make it more explicit with key terms. Keep it under 20 words.
Reply with ONLY the rewritten query, nothing else.

Question: {query}
Search query:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    expanded = response.choices[0].message.content.strip()
    return expanded

class RAGPipeline:
    """
    Full RAG pipeline connecting retrieval to generation.
    
    Retrieval  →  vector store (hybrid search + rerank)
    Generation →  Groq API (Llama 3.3 70B)
    """

    def __init__(self):
        self.vector_store = HybridVectorStore()
        self.reranker = Reranker()
        self.groq_client = None
        self.is_loaded = False

    def load(self):
        """Load all components."""
        print("Initializing RAG Pipeline...")

        # Load vector store
        self.vector_store.load()

        # Load reranker
        self.reranker.load()

        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("Groq client ready.")

        self.is_loaded = True
        print("RAG Pipeline ready.\n")

    def _build_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a context string for the prompt.
        
        Each chunk gets a numbered header so Claude can reference it.
        We include the source so the LLM can cite it.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            metadata = chunk.get("metadata", {})

            # Build a descriptive header based on source type
            if source == "race_results":
                race = metadata.get("race", "")
                season = metadata.get("season", "")
                header = f"[{i}] Race Result — {race} {season}"
            elif source == "standings":
                year = metadata.get("year", "")
                stype = metadata.get("type", "")
                header = f"[{i}] Standings — {stype} {year}"
            elif source == "wikipedia":
                page = metadata.get("page", "")
                header = f"[{i}] Wikipedia — {page}"
            elif source == "drivers":
                driver = metadata.get("driver", "")
                header = f"[{i}] Driver Profile — {driver}"
            elif source == "constructors":
                team = metadata.get("team", "")
                header = f"[{i}] Constructor — {team}"
            else:
                header = f"[{i}] {source}"

            context_parts.append(f"{header}\n{chunk['text']}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the full user message combining context and question.
        
        Structure:
        - Context section (retrieved chunks)
        - Clear separator
        - The actual question
        
        This structure helps the LLM distinguish between
        reference material and the actual question.
        """
        return f"""Here is the relevant context from the F1 knowledge base:

{context}

---

Based on the context above, please answer this question:
{query}"""

    def query(self, user_question: str) -> dict:
        """
        Main method — takes a question, returns answer + sources.
        
        Returns a dict with:
        - answer: the LLM's response
        - sources: list of chunks used
        - retrieved_count: how many chunks were retrieved
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        # Step 1 — Hybrid search
        # Step 1 — Expand query for better retrieval
        expanded_query = expand_query(user_question, self.groq_client)
        print(f"  Expanded query: {expanded_query}")

        # Step 2 — Hybrid search on expanded query
        candidates = self.vector_store.search(
            expanded_query,
            top_k=TOP_K_RETRIEVAL
        )

        # Step 2 — Re-rank
        top_chunks = self.reranker.rerank(
            user_question,
            candidates,
            top_k=TOP_K_RERANK
        )

        # Step 3 — Build context and prompt
        context = self._build_context(top_chunks)
        prompt = self._build_prompt(user_question, context)

        # Step 4 — Call Groq API
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,   # low temperature = more factual, less creative
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        # Step 5 — Return answer + metadata
        return {
            "answer": answer,
            "sources": [
                {
                    "source": chunk["source"],
                    "metadata": chunk["metadata"],
                    "text_preview": chunk["text"][:150],
                    "rerank_score": chunk.get("rerank_score", 0)
                }
                for chunk in top_chunks
            ],
            "retrieved_count": len(candidates)
        }


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load()

    test_questions = [
        "Who won the 2021 Formula 1 drivers championship?",
        "Tell me about Max Verstappen's career",
        "What is DRS and how does it work?",
        "Which team won the most constructors championships between 2018 and 2024?"
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        result = pipeline.query(question)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used ({len(result['sources'])}):")
        for i, src in enumerate(result['sources'], 1):
            print(f"  {i}. [{src['source']}] {src['text_preview'][:80]}...")