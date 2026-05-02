# Product Requirements Document — Local Wikipedia RAG

## 1. Problem Statement

Large Language Models (LLMs) hallucinate facts. When users ask questions about specific people or places, a vanilla LLM may confidently produce plausible-sounding but incorrect information. This is especially problematic in educational settings where factual accuracy matters.

Existing solutions (ChatGPT, Gemini) either require cloud access, incur API costs, or raise data-privacy concerns. Students and researchers working in offline environments — or those who cannot send queries to third-party servers — have no good alternative today.

## 2. Target Users

| User | Motivation |
|------|-----------|
| University students | Research help that cites sources; works offline in libraries |
| Educators | Demonstrating RAG concepts with a tangible, runnable system |
| Privacy-conscious professionals | Query sensitive topics locally without cloud leakage |
| ML/NLP researchers | Baseline RAG implementation to experiment with chunking, retrieval, and prompting strategies |

## 3. Goals

- Provide accurate, source-grounded answers about famous people and places.
- Run 100% locally — no API keys, no internet required after initial data pull.
- Be deployable by any developer with a laptop in under 10 minutes.
- Degrade gracefully: clearly say "I don't know" rather than hallucinate.

## 4. Non-Goals

- Real-time Wikipedia updates (ingestion is a manual one-time step).
- Covering arbitrary Wikipedia topics (scoped to ~40 pre-defined entities).
- Mobile or native app packaging.
- Multi-user / multi-tenant support in v1.

## 5. Feature Requirements

### 5.1 Core Features (Must Have)

| ID | Feature | Description |
|----|---------|-------------|
| F1 | Wikipedia ingestion | Scrape, chunk, embed (sentence-transformers), and store 20+ people + 20+ places |
| F2 | Semantic retrieval | Query the correct ChromaDB collection based on query type |
| F3 | Grounded generation | LLM answers only from retrieved context; refuses to hallucinate |
| F4 | Chat interface | Persistent conversation with user/assistant message history |
| F5 | "I don't know" handling | Model explicitly says so when context is insufficient |

### 5.2 UI Features (Should Have)

| ID | Feature |
|----|---------|
| U1 | Sidebar: Ingest Data button with progress bar |
| U2 | Sidebar: Model selector (llama3.2:3b / mistral / phi3) |
| U3 | Sidebar: top-k slider |
| U4 | Toggle: show retrieved source chunks |
| U5 | Clear Chat button |
| U6 | Friendly error messages when Ollama is not running |
| U7 | Reset System button — clears all ingested data (SQLite + ChromaDB) and chat history |
| U8 | Source citations below each answer (📚 Sources: **Einstein** · **Tesla**) |
| U9 | Latency display — retrieval, generation, and total time per query |
| U10 | Response caching — identical queries return instantly from in-memory cache |

### 5.3 Nice to Have (Future)

- Incremental ingestion of new Wikipedia pages via URL input.
- Export conversation as PDF.
- Confidence scoring overlay on retrieved chunks.
- Cross-lingual support (non-English Wikipedia).

## 6. Success Metrics

| Metric | Target |
|--------|--------|
| Factual accuracy on 10 sample queries (manual eval) | ≥ 8/10 correct |
| "I don't know" correctly returned for out-of-KB queries | 100% |
| Time to first answer (after ingestion) | < 10 seconds on consumer hardware |
| Ingestion time (40 entities, all-MiniLM-L6-v2) | < 20 minutes |
| Zero external HTTP calls during query | Always |

## 7. Constraints

- **Local-only**: All computation on localhost; no cloud services.
- **No RAG frameworks**: LangChain, LlamaIndex, etc. are excluded — RAG logic is implemented from scratch.
- **Dependency footprint**: Only standard open-source Python libraries (chromadb, streamlit, tiktoken, requests, wikipedia, sentence-transformers).
- **Model size**: Default LLM ≤ 3B parameters so the app runs on CPUs with 8 GB RAM.

## 8. Data Model

```
entities(id, title, entity_type, url, raw_text, ingested_at)
chunks(id, entity_id, chunk_index, chunk_text, token_count, created_at)

ChromaDB: people_collection, places_collection
  — each document: {id, text, embedding, metadata{source_title, entity_type, chunk_index, url}}
```

## 9. Design Choices

### 9.1 Chunking strategy — fixed-size with overlap (500 / 50 tokens)

Implementation: `ingest/chunker.py`

- **Token-level boundaries** via tiktoken `cl100k_base` rather than character or word counts. This keeps chunk sizes meaningful regardless of whitespace/punctuation density.
- **Sliding window generator** (`_token_windows`) so very large documents are processed iteratively — never loaded fully into memory at once.
- **50-token overlap** preserves cross-boundary context: a sentence that straddles two chunks remains intact in at least one of them, so retrieval does not lose facts that happen to fall on a chunk edge.
- **Why 500 tokens?** Large enough to hold a meaningful biographical paragraph, small enough that a top-k of 5 fits comfortably (≤ 2 500 tokens of context) inside the 4 096-token window of a 3B-parameter model with room left for the system prompt and the answer.

### 9.1b Embedding model — sentence-transformers (`all-MiniLM-L6-v2`)

Implementation: `ingest/embedder.py`

The assignment allows either Ollama `nomic-embed-text` or `sentence-transformers`. We chose **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim) because the Ollama nomic-embed-text implementation on Windows produced degenerate (identical) embedding vectors for short, similarly-sized inputs (e.g., "Albert Einstein" vs "Nikola Tesla" had cosine similarity of 1.0). This made retrieval for entity-name queries essentially random.

`all-MiniLM-L6-v2` runs locally (CPU or GPU), requires no server, and produces well-discriminated vectors (Einstein vs Tesla cos=0.59, Einstein vs Eiffel cos=0.19).

### 9.2 Vector store — Option A (two collections), not Option B (one collection + metadata filter)

Implementation: `db/vector_store.py:21-30`

| | Option A (chosen) | Option B |
|---|---|---|
| Layout | `people_collection` + `places_collection` | One `entities_collection` with `where={'entity_type': 'person'}` filter |
| Cross-category leakage | Impossible — physically separated | Possible — a place chunk that happens to be cosine-similar to a person query can still bubble up if the filter is forgotten |
| HNSW tuning per category | Each collection can be tuned independently | Shared index parameters |
| Code path | Classifier picks the collection; query goes there directly | Every query carries metadata filter; one index serves both |
| Operational simplicity | Slightly more bookkeeping (two collections) | Slightly simpler |

**Why Option A was the right call here:**

1. The classifier outputs a hard category label (`person` / `place` / `both`). Routing by collection makes that label load-bearing — if classification is correct, the wrong category never even sees the query.
2. The "both" path explicitly merges results from the two collections by distance (`retrieval/retriever.py:111-116`), giving the *application* control over how cross-category results are combined rather than letting one shared index decide implicitly. For multi-entity queries ("Compare Einstein and Tesla") the retriever runs per-entity embeddings and deduplicates (`retrieval/retriever.py:69-103`).
3. The data set is small but heterogeneous (biographies vs. landmarks have very different vocabulary). Separate indexes mean cosine distances are calibrated within a category, not across them.

### 9.3 Retrieval — rule-based classifier + cosine similarity

Implementation: `retrieval/query_classifier.py`, `retrieval/retriever.py`

- Classifier first looks for **direct entity-name mentions** against the known `PEOPLE`/`PLACES` lists from `config.py`. If a known person appears and no known place does, the query is unambiguously "person".
- If no entity name matches, fall back to **keyword scoring** with two curated vocabulary sets. A 2:1 winning ratio is required to commit to a single category — otherwise the system queries both collections.
- The classifier is intentionally cheap and explainable. There is no ML model to train, version, or debug.

### 9.4 Generation — context-grounded system prompt

Implementation: `generation/generator.py:17-30`

The system prompt has ten guidelines. The most important are:
- Answer using **facts from the context passages** — no outside knowledge.
- Do **not** make up facts that are not in the context (Rule 6).
- If the context is only partially relevant, answer the parts that can be confirmed and state what is missing (Rule 7–8).
- Only return `"I don't know based on the available data."` when the passages contain **absolutely no** relevant information (Rule 9).

This is intentionally more flexible than a blanket "only from context" rule: partial answers are better than false refusals. Any refusal-variant the model produces is normalised to the canonical `_IDK` string by `_is_refusal()` (`generation/generator.py:74-82`), so callers always receive a consistent signal.

Rules 5–6 explicitly guide the model on "which person/place" queries: the source title of the retrieved passages is the answer candidate, and the model should use it directly. Rules 4 and 8–9 cover comparison and partial-answer scenarios. We deliberately do not use few-shot examples or chain-of-thought scaffolding — the prompt is short and easy to audit. The retrieved chunks are formatted with explicit source labels (`--- Passage N (Source: <title>) ---`) so the LLM is encouraged to cite, and so a human reviewer can spot-check whether an answer is grounded.

## 10. Risks

| Risk | Mitigation |
|------|-----------|
| Wikipedia API rate limits | 1.5s delay + retry with exponential backoff (up to 5 attempts) |
| Disambiguation pages | Auto-select first unambiguous option |
| Ollama not installed | Friendly error in UI with install instructions |
| Embedding model not available | Auto-downloaded by sentence-transformers on first run; no manual step |
| Context window overflow | Chunking keeps each input to LLM well within 4 096 tokens |
