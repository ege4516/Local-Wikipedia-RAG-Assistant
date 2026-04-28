# Local Wikipedia RAG

A fully local, ChatGPT-style Retrieval Augmented Generation system that answers questions about famous people and places using Wikipedia data. No external APIs — everything runs on your machine.

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com), then start the server:

```bash
ollama serve
```

### 2. Pull the LLM model

```bash
# Pick at least one (llama3.2:3b is the default)
ollama pull llama3.2:3b
ollama pull mistral      # optional
ollama pull phi3          # optional
```

> **Note:** Embeddings are handled locally by `sentence-transformers` (`all-MiniLM-L6-v2`). The model is downloaded automatically on first run — no Ollama embedding model is needed.

### 3. Install Python dependencies

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

## How to run

### Step 1 — Make sure Ollama is running

In a separate terminal:

```bash
ollama serve    # leave this running
```

### Step 2 — Ingest Wikipedia data

Two equivalent options — pick one.

**Option A — standalone CLI script (recommended for first run):**

```bash
python ingest_all.py
```

This scrapes the 20 + 20 entities listed in `config.py`, chunks them, embeds the chunks via sentence-transformers, and writes everything to `data/wiki_rag.db` (SQLite) and `data/chroma_db/` (ChromaDB). Expect ~10 minutes on a GPU, ~20 minutes on CPU.

**Option B — from the Streamlit UI:** start the app (next step) and click the **Ingest Data** button in the sidebar.

### Step 3 — Start the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Example queries

### People

- "Who was Albert Einstein and what is he known for?"
- "What did Marie Curie discover?"
- "Why is Nikola Tesla famous?"
- "Compare Lionel Messi and Cristiano Ronaldo"
- "What is Frida Kahlo known for?"

### Places

- "Where is the Eiffel Tower located?"
- "Why is the Great Wall of China important?"
- "What is Machu Picchu?"
- "What was the Colosseum used for?"
- "Where is Mount Everest?"

### Mixed / Comparison

- "Which famous place is located in Turkey?"
- "Which person is associated with electricity?"
- "Compare Albert Einstein and Nikola Tesla"
- "Compare the Eiffel Tower and the Statue of Liberty"

### Failure cases (should return "I don't know")

- "Who is the president of Mars?"
- "Tell me about a random unknown person John Doe"

## Project structure

```
project/
├── app.py                    ← Streamlit UI entry point
├── ingest_all.py             ← Standalone ingestion CLI script
├── config.py                 ← All configuration constants
├── requirements.txt
├── .env.example
├── db/
│   ├── sqlite_store.py       ← Raw text + chunk metadata persistence
│   └── vector_store.py       ← ChromaDB wrapper (two collections)
├── ingest/
│   ├── wikipedia_scraper.py  ← Wikipedia API client
│   ├── chunker.py            ← Fixed-size token chunking with overlap
│   └── embedder.py           ← sentence-transformers (all-MiniLM-L6-v2)
├── retrieval/
│   ├── query_classifier.py   ← Rule-based person/place/both classifier
│   └── retriever.py          ← End-to-end retrieval pipeline
├── generation/
│   └── generator.py          ← Ollama LLM client + prompt construction
├── README.md
├── product_prd.md
└── recommendation.md
```

## Architecture and design choices

A full explanation of the chunking strategy, the two-collection vector-store layout (Option A from the assignment), the retrieval pipeline, and the grounded-generation prompt is in **[product_prd.md](product_prd.md)**.

## Technical decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | The Ollama `nomic-embed-text` produced degenerate (identical) vectors for short entity-name queries on Windows, making retrieval random. sentence-transformers is explicitly allowed by the assignment and produces well-discriminated 384-dim vectors locally. |
| Vector store layout | Option A — two collections | Physical separation prevents cross-category leakage; classifier routes queries to the correct collection |
| Chunking | 500 tokens, 50 overlap | Fits 5 chunks within the 4K context window of a 3B model with room for prompt and answer |
| LLM | llama3.2:3b via Ollama | Good balance of quality and speed on consumer hardware |

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE_TOKENS` | 500 | Tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | 50 | Overlap between consecutive chunks |
| `DEFAULT_TOP_K` | 5 | Chunks retrieved per query |
| `DEFAULT_LLM_MODEL` | llama3.2:3b | Default Ollama model |
| `EMBED_MODEL` | all-MiniLM-L6-v2 | Sentence-transformers embedding model |
| `EMBED_DIM` | 384 | Embedding vector dimension |
