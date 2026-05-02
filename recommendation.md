# Production Deployment Recommendation

## Current State vs. Production

The local prototype runs entirely on a developer's laptop. Scaling to a production-grade service requires rethinking every layer.

---

## 1. Containerisation (Docker)

### Single-node deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  rag-app:
    build: .
    ports: ["8501:8501"]
    volumes:
      - rag-data:/app/data
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    depends_on: [ollama]

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes:
      - ollama-models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  rag-data:
  ollama-models:
```

**Trade-off**: Docker Compose is simple but still single-host. Use Kubernetes for multi-replica deployments.

---

## 2. Cloud Vector Database Options

Replace local ChromaDB with a managed vector store for persistence, scalability, and multi-user support.

| Option | Strengths | Weaknesses | Cost Estimate |
|--------|-----------|-----------|--------------|
| **Pinecone** | Fully managed, fast ANN, generous free tier | Vendor lock-in, US-only free tier | $0–$70/mo (Starter) |
| **Weaviate Cloud** | Open-source core, GraphQL + REST, hybrid search | More complex setup | $0–$25/mo (Sandbox) |
| **Qdrant Cloud** | Rust-based, very fast, self-hostable | Smaller ecosystem | $0 self-host |
| **pgvector (PostgreSQL)** | No new infra if you already use Postgres | Slower ANN at scale | Same as DB cost |
| **Chroma Cloud** (if available) | Direct migration from local ChromaDB | Early product | TBD |

**Recommendation**: Qdrant self-hosted on a $10/mo VPS for cost control; Pinecone if managed simplicity is priority.

---

## 3. Model Serving at Scale

### Embedding service

The local prototype uses **sentence-transformers** (`all-MiniLM-L6-v2`) in-process — no separate embedding server. Under multi-user load this becomes a bottleneck because the model is loaded once per worker process.

Replace the in-process encode call with a dedicated embedding microservice:

- **Option A — Infinity (MIT)**: Lightweight embedding-only REST server, supports `all-MiniLM-L6-v2` and other sentence-transformers models natively. `docker run michaelf34/infinity`.
- **Option B — vLLM**: Serves any HuggingFace model with OpenAI-compatible API. Supports batching and tensor parallelism. Best for GPU servers.
- **Option C — Ollama behind a load balancer**: Run 2–4 Ollama replicas with `nomic-embed-text` behind nginx. Simple migration if switching back to Ollama embeddings; limited batching.

### LLM service

| Approach | Hardware | Throughput | Latency |
|----------|----------|------------|---------|
| Ollama (single) | 8-core CPU | ~1 req/s | 5–30s |
| vLLM on A10G GPU | 1× GPU | ~20 req/s | 1–3s |
| Llama.cpp server | Any CPU/GPU | 2–10 req/s | 3–15s |
| Groq API (cloud) | Groq LPU | 100+ req/s | <1s |

**Recommendation**: vLLM on a single A10G (GCP `g2-standard-4`, ~$0.90/hr spot) for a team of 20+ concurrent users.

---

## 4. Monitoring

### Application metrics (Prometheus + Grafana)

Instrument the following:
- `rag_query_latency_seconds` — histogram of end-to-end query time
- `rag_retrieval_latency_seconds` — retrieval-only time
- `rag_llm_latency_seconds` — generation-only time
- `rag_empty_context_total` — count of "I don't know" responses
- `rag_ingestion_entities_total` — entities successfully ingested

### LLM quality monitoring

- Log all (query, context, answer) triples to a structured store (S3 + Athena or BigQuery).
- Weekly human evaluation sample: pull 50 random logs, spot-check factual accuracy.
- Alert if `rag_empty_context_total / total_queries > 0.3` (knowledge gap signal).

### Infrastructure monitoring

- Ollama/vLLM GPU utilisation via DCGM Exporter.
- ChromaDB/vector store query latency percentiles.
- SQLite → PostgreSQL migration alert on write contention.

---

## 5. Cost Estimates (Monthly, 1 000 queries/day)

| Component | Local prototype | Cloud (mid-tier) |
|-----------|----------------|-----------------|
| Compute (LLM) | $0 (laptop) | $50–$150 (spot GPU) |
| Vector DB | $0 (local) | $0–$25 (Qdrant/Pinecone free) |
| Object storage (logs) | $0 | $2–$5 (S3) |
| CDN / ingress | $0 | $5–$10 |
| **Total** | **$0** | **~$60–$190/mo** |

---

## 6. Trade-offs vs. Current Local Setup

| Dimension | Local Prototype | Production |
|-----------|----------------|------------|
| Cost | Free | $60–$190/mo |
| Setup time | 10 minutes | 2–8 hours |
| Concurrent users | 1 | 20–100+ |
| Data privacy | Maximum | Depends on cloud provider |
| Reliability | Single point of failure | HA with replicas |
| Maintenance burden | None | Ongoing (patches, alerts) |
| GPU requirement | Optional | Recommended for latency |

---

## 7. Recommended Production Stack (Priority Order)

1. **Containerise** with Docker Compose — lowest effort, immediate reproducibility.
2. **Replace SQLite → PostgreSQL** — handles concurrent writes, better tooling.
3. **Replace local ChromaDB → Qdrant self-hosted** — persistent, queryable via REST.
4. **Move Ollama → vLLM on GPU** — 10× latency improvement.
5. **Add Prometheus + Grafana** — visibility before scaling further.
6. **Kubernetes (optional)** — only if > 50 concurrent users or 99.9% SLA needed.
