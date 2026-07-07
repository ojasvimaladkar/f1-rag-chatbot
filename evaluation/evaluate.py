import sys
import json
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import os

sys.path.append(str(Path(__file__).parent.parent / "src"))
from rag_pipeline import RAGPipeline

load_dotenv()

# ─────────────────────────────────────────
# We use the LLM itself as a judge
# This is called LLM-as-a-judge evaluation
# It's actually what RAGAS does internally too
# ─────────────────────────────────────────

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def llm_score(prompt: str) -> float:
    """Ask LLM to score something 0-10, return as 0-1."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    text = response.choices[0].message.content.strip()
    try:
        score = float(text.split()[0].replace(",", "."))
        return min(max(score / 10, 0), 1)
    except:
        return 0.5


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """Does the answer stick to the retrieved context?"""
    context_text = "\n".join(contexts[:3])
    prompt = f"""Rate from 0-10 how faithful this answer is to the provided context.
10 = answer only uses information from context.
0 = answer contains information not in context.
Reply with ONLY a number.

Context:
{context_text}

Answer:
{answer}

Score:"""
    return llm_score(prompt)


def score_relevancy(question: str, answer: str) -> float:
    """Does the answer actually address the question?"""
    prompt = f"""Rate from 0-10 how well this answer addresses the question.
10 = directly and completely answers the question.
0 = completely irrelevant or doesn't answer at all.
Reply with ONLY a number.

Question: {question}
Answer: {answer}

Score:"""
    return llm_score(prompt)


def score_context_quality(question: str, contexts: list[str]) -> float:
    """Are the retrieved chunks relevant to the question?"""
    context_text = "\n".join(contexts[:3])
    prompt = f"""Rate from 0-10 how relevant these retrieved passages are to the question.
10 = passages directly contain information needed to answer.
0 = passages are completely unrelated to the question.
Reply with ONLY a number.

Question: {question}

Retrieved passages:
{context_text}

Score:"""
    return llm_score(prompt)


def evaluate():
    print("=" * 60)
    print("F1 RAG Chatbot — Evaluation")
    print("=" * 60)

    # Load pipeline
    print("\nLoading RAG pipeline...")
    pipeline = RAGPipeline()
    pipeline.load()

    # Load eval dataset
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} evaluation questions\n")

    results = []
    faithfulness_scores = []
    relevancy_scores = []
    context_scores = []

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i+1}/{len(dataset)}] {question[:55]}...")

        # Run pipeline
        result = pipeline.query(question)
        answer = result["answer"]
        contexts = [s["text_preview"] for s in result["sources"]]

        # Score all three metrics
        faith = score_faithfulness(answer, contexts)
        relev = score_relevancy(question, answer)
        ctx_q = score_context_quality(question, contexts)

        faithfulness_scores.append(faith)
        relevancy_scores.append(relev)
        context_scores.append(ctx_q)

        print(f"  Faithfulness: {faith:.2f} | Relevancy: {relev:.2f} | Context: {ctx_q:.2f}")

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_quality": ctx_q
        })

        time.sleep(1)

    # Aggregate scores
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relev = sum(relevancy_scores) / len(relevancy_scores)
    avg_ctx = sum(context_scores) / len(context_scores)
    overall = (avg_faith + avg_relev + avg_ctx) / 3

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SCORES")
    print("=" * 60)
    print(f"Faithfulness:      {avg_faith:.4f}  (are answers grounded in context?)")
    print(f"Answer Relevancy:  {avg_relev:.4f}  (do answers address the question?)")
    print(f"Context Quality:   {avg_ctx:.4f}  (is retrieval finding right chunks?)")
    print(f"Overall Score:     {overall:.4f}")
    print("=" * 60)

    # Save results
    output = {
        "scores": {
            "faithfulness": avg_faith,
            "answer_relevancy": avg_relev,
            "context_quality": avg_ctx,
            "overall": overall
        },
        "per_question": results
    }

    scores_path = Path(__file__).parent / "eval_scores.json"
    with open(scores_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved → {scores_path}")
    print("Add these scores to your README!")


if __name__ == "__main__":
    evaluate()