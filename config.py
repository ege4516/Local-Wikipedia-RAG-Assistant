"""
Central configuration for the Local Wikipedia RAG system.
All tuneable constants live here — import from this module everywhere.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "data", "wiki_rag.db")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model (384-dim)
EMBED_DIM = 384
LLM_TIMEOUT_SEC = 300        # per-generation HTTP timeout (CPU-friendly)

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TIKTOKEN_ENCODING = "cl100k_base"   # used for all token counting

# ── Retrieval ──────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# ── ChromaDB collection names ──────────────────────────────────────────────
PEOPLE_COLLECTION = "people_collection"
PLACES_COLLECTION = "places_collection"

# ── Wikipedia entities to ingest ───────────────────────────────────────────
PEOPLE = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Isaac Newton",
    "Charles Darwin",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Stephen Hawking",
    "Elon Musk",
    "Cleopatra",
    "Napoleon Bonaparte",
    "Wolfgang Amadeus Mozart",
    "Vincent van Gogh",
]

PLACES = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    "Niagara Falls",
    "Stonehenge",
    "Acropolis of Athens",
    "Angkor Wat",
    "Petra",
    "Sydney Opera House",
    "Big Ben",
    "Mount Fuji",
    "Venice",
    "Santorini",
]
