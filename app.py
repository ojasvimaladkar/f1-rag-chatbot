import streamlit as st
import requests
import json
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="F1 RAG Chatbot",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# Makes it look professional
# ─────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #E10600;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #1a1a1a;
        border-left: 3px solid #E10600;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .metric-card {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stChatMessage {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# Streamlit reruns the entire script on every
# interaction. Session state persists data
# between reruns — this is how chat history works.
# ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ─────────────────────────────────────────
# API FUNCTIONS
# ─────────────────────────────────────────

def query_api(question: str) -> dict | None:
    """Send question to FastAPI backend and return result."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_stats() -> dict | None:
    """Fetch usage stats from API."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        return response.json()
    except:
        return None


def get_logs() -> list:
    """Fetch recent query logs from API."""
    try:
        response = requests.get(f"{API_URL}/logs?limit=10", timeout=5)
        return response.json().get("logs", [])
    except:
        return []


def check_api_health() -> bool:
    """Check if FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False


# ─────────────────────────────────────────
# SOURCE BADGE COLORS
# ─────────────────────────────────────────

SOURCE_COLORS = {
    "race_results": "#E10600",
    "standings": "#FFA500",
    "wikipedia": "#3366CC",
    "drivers": "#00A651",
    "constructors": "#9B59B6"
}

SOURCE_LABELS = {
    "race_results": "🏁 Race Result",
    "standings": "🏆 Standings",
    "wikipedia": "📖 Wikipedia",
    "drivers": "👤 Driver Profile",
    "constructors": "🏎️ Constructor"
}


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏎️ F1 RAG Chatbot")
    st.markdown("---")

    # API status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.info("Run: `uvicorn api.main:app --reload --port 8000`")

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Session Stats")
    stats = get_stats()
    if stats and stats["total_queries"] > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", stats["total_queries"])
        with col2:
            st.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
    else:
        st.caption("No queries yet")

    if st.session_state.last_latency:
        st.metric("Last Query", f"{st.session_state.last_latency:.0f}ms")

    st.markdown("---")

    # Knowledge base info
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
    - 🏁 **153** race results (2018-2024)
    - 🏆 **14** standings tables
    - 👤 **81** driver profiles
    - 🏎️ **23** constructors
    - 📖 **30** Wikipedia articles
    """)

    st.markdown("---")

    # Example questions
    st.markdown("### 💡 Try Asking")
    example_questions = [
        "Who won the 2021 F1 championship?",
        "Tell me about Max Verstappen",
        "What is DRS in Formula 1?",
        "Which team won the 2022 constructors title?",
        "Who drove for Ferrari in 2023?",
        "What happened at the 2021 Monaco GP?"
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.prefill_question = q

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_latency = None
        st.rerun()

    st.markdown("---")
    st.caption("Built with FAISS + BM25 + Cross-Encoder + Llama 3.3 70B")


# ─────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────

st.markdown('<div class="main-header">🏎️ F1 RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask anything about Formula 1 — powered by RAG</div>', unsafe_allow_html=True)

# Split into chat and sources panel
chat_col, sources_col = st.columns([2, 1])

with chat_col:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle prefilled question from sidebar buttons
    prefill = st.session_state.pop("prefill_question", None)

    # Chat input
    user_input = st.chat_input("Ask an F1 question...") or prefill

    if user_input:
        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get answer from API
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = query_api(user_input)

            if result:
                answer = result["answer"]
                st.markdown(answer)

                # Store for sources panel
                st.session_state.last_sources = result["sources"]
                st.session_state.last_latency = result["latency_ms"]
                st.session_state.total_queries += 1

                # Show latency
                st.caption(f"⚡ {result['latency_ms']:.0f}ms | 📄 {len(result['sources'])} sources")

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                st.error("Failed to get response from API.")

with sources_col:
    st.markdown("### 📄 Retrieved Sources")

    if st.session_state.last_sources:
        st.caption("Sources used for last answer:")
        for i, source in enumerate(st.session_state.last_sources, 1):
            source_type = source["source"]
            label = SOURCE_LABELS.get(source_type, source_type)
            color = SOURCE_COLORS.get(source_type, "#888")
            metadata = source["metadata"]
            score = source["rerank_score"]

            with st.expander(f"{label} — score: {score:.2f}"):
                # Show metadata
                for key, value in metadata.items():
                    st.caption(f"**{key}:** {value}")
                st.caption(f"**Relevance score:** {score:.4f}")
                st.markdown("**Preview:**")
                st.text(source["text_preview"])
    else:
        st.caption("Sources will appear here after your first question.")

    st.markdown("---")

    # Observability — recent query log
    st.markdown("### 🔍 Recent Queries")
    logs = get_logs()
    if logs:
        for log in logs[:5]:
            with st.expander(f"Q: {log['question'][:40]}..."):
                st.caption(f"⏱️ {log['latency_ms']:.0f}ms | 📅 {log['timestamp']}")
                st.markdown(f"**A:** {log['answer'][:200]}...")
    else:
        st.caption("Query history will appear here.")