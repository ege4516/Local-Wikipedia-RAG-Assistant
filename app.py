"""
Streamlit entry point — Local Wikipedia RAG chatbot.
Run with: streamlit run app.py
"""

import logging
import os
import sys

import streamlit as st

# ── ensure project root is on the path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    PEOPLE,
    PLACES,
)
from db.sqlite_store import SQLiteStore
from db.vector_store import VectorStore
from generation.generator import Generator
from ingest.chunker import TextChunker
from ingest.embedder import OllamaEmbedder
from ingest.wikipedia_scraper import WikipediaScraper
from retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local Wikipedia RAG",
    page_icon="📚",
    layout="wide",
)

# ── Session state defaults ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []
if "last_category" not in st.session_state:
    st.session_state.last_category = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ── Cached resource initialisation ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_stores():
    return SQLiteStore(), VectorStore()


@st.cache_resource(show_spinner=False)
def get_embedder():
    return OllamaEmbedder()


# ── Ingestion pipeline ───────────────────────────────────────────────────────
def run_ingestion(progress_bar, status_text):
    sqlite_store, vector_store = get_stores()
    embedder = get_embedder()
    scraper = WikipediaScraper()
    chunker = TextChunker()

    all_entities = [(t, "person") for t in PEOPLE] + [(t, "place") for t in PLACES]
    total = len(all_entities)
    completed = 0

    for title, entity_type in all_entities:
        status_text.text(f"Scraping: {title}...")
        data = scraper.scrape_page(title)
        if not data:
            status_text.text(f"Skipped (not found): {title}")
            completed += 1
            progress_bar.progress(completed / total)
            continue

        entity_id = sqlite_store.insert_entity(
            title=data["title"],
            entity_type=entity_type,
            url=data["url"],
            raw_text=data["content"],
        )

        # Remove stale chunks/vectors before re-inserting
        sqlite_store.delete_chunks_for_entity(entity_id)
        vector_store.delete_entity(data["title"], entity_type)

        chunks = chunker.chunk_text(
            text=data["content"],
            source_title=data["title"],
            entity_type=entity_type,
            url=data["url"],
        )

        if not chunks:
            completed += 1
            progress_bar.progress(completed / total)
            continue

        # Persist chunks to SQLite
        for chunk in chunks:
            sqlite_store.insert_chunk(
                entity_id=entity_id,
                chunk_index=chunk["metadata"]["chunk_index"],
                chunk_text=chunk["text"],
                token_count=chunk["token_count"],
            )

        # Embed and store in ChromaDB
        status_text.text(f"Embedding {len(chunks)} chunks for: {title}...")
        texts = [c["text"] for c in chunks]
        embeddings = embedder.get_embeddings_batch(texts)

        valid_pairs = [
            (c, e) for c, e in zip(chunks, embeddings) if e is not None
        ]
        if valid_pairs:
            valid_chunks, valid_embeddings = zip(*valid_pairs)
            vector_store.add_chunks(
                list(valid_chunks), list(valid_embeddings), entity_type
            )

        completed += 1
        progress_bar.progress(completed / total)

    counts = vector_store.collection_counts()
    status_text.text(
        f"Done! People vectors: {counts['people_collection']} | "
        f"Place vectors: {counts['places_collection']}"
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Wikipedia RAG")
    st.markdown("---")

    # Model selector
    model_options = ["llama3.2:3b", "mistral", "phi3"]
    selected_model = st.selectbox("LLM Model", model_options, index=0)

    # Top-k slider
    top_k = st.slider("Top-k chunks", min_value=1, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)

    # Show chunks toggle
    show_chunks = st.toggle("Show retrieved chunks", value=False)

    st.markdown("---")

    # Ingest data button
    if st.button("Ingest Data", type="primary", use_container_width=True):
        with st.spinner("Running ingestion pipeline..."):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            try:
                run_ingestion(progress_bar, status_text)
                st.success("Ingestion complete!")
                # Clear the resource cache so counts refresh
                get_stores.clear()
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                logger.exception("Ingestion error")

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_chunks = []
        st.session_state.last_category = ""
        st.rerun()

    # Reset entire system (clear data + chat)
    if st.button("Reset System", use_container_width=True, type="secondary"):
        try:
            sqlite_store, vector_store = get_stores()
            sqlite_store.reset_all()
            vector_store.reset_all()
            st.session_state.messages = []
            st.session_state.last_chunks = []
            st.session_state.last_category = ""
            get_stores.clear()
            st.success("System reset — all data cleared. Click 'Ingest Data' to reload.")
        except Exception as exc:
            st.error(f"Reset failed: {exc}")
            logger.exception("Reset error")

    st.markdown("---")

    # DB stats
    try:
        sqlite_store, vector_store = get_stores()
        counts = vector_store.collection_counts()
        st.caption(
            f"Entities in DB: {sqlite_store.get_entity_count()}  \n"
            f"People vectors: {counts['people_collection']}  \n"
            f"Place vectors: {counts['places_collection']}"
        )
    except Exception:
        st.caption("DB not initialised yet.")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("Local Wikipedia RAG Chat")
st.caption("Ask about famous people and places. Everything runs locally.")

# Chat input — accept new query, save to session state, rerun
if user_input := st.chat_input("Ask about a person or place..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_query = user_input
    st.rerun()

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show retrieved chunks from last turn
if show_chunks and st.session_state.last_chunks:
    with st.expander(
        f"Retrieved chunks (category: {st.session_state.last_category})",
        expanded=False,
    ):
        for i, chunk in enumerate(st.session_state.last_chunks, start=1):
            meta = chunk.get("metadata", {})
            st.markdown(
                f"**Chunk {i}** — *{meta.get('source_title', 'Unknown')}* "
                f"(chunk #{meta.get('chunk_index', '?')})  "
                f"distance: `{chunk.get('distance', 0):.4f}`"
            )
            st.text(chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else ""))
            st.markdown("---")

# Generate response for pending query (survives widget-triggered reruns)
if st.session_state.pending_query:
    query = st.session_state.pending_query

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                embedder = get_embedder()
                _, vector_store = get_stores()
                counts = vector_store.collection_counts()
                total_docs = sum(counts.values())

                if total_docs == 0:
                    answer = (
                        "The knowledge base is empty. "
                        'Please click "Ingest Data" in the sidebar first.'
                    )
                    st.session_state.last_chunks = []
                    st.session_state.last_category = ""
                else:
                    retriever = Retriever(vector_store=vector_store, embedder=embedder)
                    result = retriever.retrieve(query, k=top_k)

                    st.session_state.last_chunks = result["chunks"]
                    st.session_state.last_category = result["category"]

                    # Build history for multi-turn context
                    # Skip assistant "I don't know" turns — the LLM tends to
                    # copy previous refusals even when fresh context is adequate.
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                        if not (m["role"] == "assistant"
                                and "I don't know" in m.get("content", ""))
                    ]

                    generator = Generator(model=selected_model)
                    answer = generator.generate(
                        query=query,
                        chunks=result["chunks"],
                        chat_history=history,
                    )

            except Exception as exc:
                logger.exception("Chat error")
                answer = f"An unexpected error occurred: {exc}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.pending_query = None
    st.rerun()

