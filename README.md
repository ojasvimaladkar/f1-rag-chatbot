# 🏎️ F1 RAG Chatbot

> A production-grade Retrieval-Augmented Generation (RAG) system that answers Formula 1 questions using hybrid search, re-ranking, and grounded LLM responses.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

F1 RAG Chatbot retrieves relevant information from a curated Formula 1 knowledge base before generating answers with **Llama 3.3 70B (Groq)**. Instead of relying solely on the language model, the system grounds every response using retrieved documents, reducing hallucinations and providing source citations.

### Features

- Hybrid retrieval using **FAISS** and **BM25**
- Reciprocal Rank Fusion (RRF)
- Cross-encoder re-ranking
- LLM-powered query expansion
- Grounded answer generation
- Source citations
- FastAPI backend
- Streamlit frontend
- SQLite query logging
- Automated evaluation pipeline

---

## Demo

### Example 1

**Question**

> Who won the 2023 Formula 1 World Championship?

**Answer**

> Max Verstappen won the 2023 Formula One World Championship driving for Red Bull Racing with **575 points** and a record-breaking **19 race victories**, securing his third consecutive title.

**Source**

`season_summary_2023`

### Example 2

**Question**

> What is DRS?

**Answer**

> Drag Reduction System (DRS) is a driver-adjustable rear wing that reduces aerodynamic drag, increasing straight-line speed to improve overtaking opportunities. It can only be activated under FIA-defined conditions.

**Source**

`Wikipedia - Drag Reduction System`

---

## Architecture

```text
                    User Question
                          │
                          ▼
               Query Expansion (LLM)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
     FAISS Search                    BM25 Search
     (Semantic)                     (Keyword)
          │                               │
          └───────────────┬───────────────┘
                          ▼
              Reciprocal Rank Fusion
                          ▼
             Cross-Encoder Re-ranking
                          ▼
          Llama 3.3 70B (Grounded Answer)
                          ▼
            Streamlit UI + Source Citations
```

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| LLM | Llama 3.3 70B (Groq) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Re-ranking | ms-marco-MiniLM-L-6-v2 |
| Backend | FastAPI |
| Frontend | Streamlit |
| Database | SQLite |

---

## Knowledge Base

| Source | Documents |
|---------|----------:|
| Jolpica Race Results (2000–2024) | 531 |
| Championship Standings | 51 |
| Driver Profiles | 127 |
| Constructor Profiles | 38 |
| Wikipedia Articles | 786 |
| Season Summaries | 7 |
| **Total** | **1,540** |

---

## Evaluation

The pipeline was evaluated using an **LLM-as-a-Judge** methodology on a curated 10-question benchmark.

| Metric | Score |
|---------|------:|
| Answer Relevancy | **1.00** |
| Context Quality | **0.94** |
| Faithfulness | **0.80** |
| **Overall** | **0.91** |

---

## Project Structure

```text
f1-rag-chatbot/
│
├── api/
│   └── main.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── eval_dataset.json
│   └── eval_scores.json
│
├── src/
│   ├── chunker.py
│   ├── data_collector.py
│   ├── embedder.py
│   ├── rag_pipeline.py
│   ├── reranker.py
│   └── vector_store.py
│
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ojasvimaladkar/f1-rag-chatbot.git
cd f1-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Linux/macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file.

```bash
cp .env.example .env
```

Add your Groq API key.

```env
GROQ_API_KEY=your_api_key
```

---

## Build the Knowledge Base

```bash
python src/data_collector.py
python src/chunker.py
python src/embedder.py
```

---

## Run the Application

Start the FastAPI backend.

```bash
uvicorn api.main:app --reload
```

In another terminal, launch the Streamlit frontend.

```bash
streamlit run app.py
```

Open your browser and navigate to:

```
http://localhost:8501
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/query` | POST | Ask the chatbot a question |
| `/logs` | GET | View recent query history |
| `/stats` | GET | Usage statistics |

---

## Future Improvements

- Support live Formula 1 season updates
- Migrate to a hosted vector database (Pinecone/Qdrant)
- Fine-tune embedding models on Formula 1 data
- Add qualifying, sprint, and lap-time information
- Expand automated evaluation with RAGAS

---

## Limitations

- Knowledge base covers seasons from **2000–2024**
- No live race updates
- General-purpose embeddings may miss niche Formula 1 terminology
- Faithfulness may decrease when retrieved context is incomplete

---

## License

This project is licensed under the **MIT License**.
