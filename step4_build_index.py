"""
QuIM-RAG Phase 2 - Step 4: Build Inverted Index
================================================
Reads:  data/corpus.jsonl           (output of step3_question_generation.py)
Writes: data/chroma_db/             (ChromaDB persistent store)
          └── question_collection   — question embeddings + chunk_id metadata
          └── chunk_collection      — raw chunk texts + source_url metadata

Paper spec:
  - Embedding model : BAAI/bge-large-en-v1.5  (top MTEB benchmark)
  - Vector DB       : ChromaDB
  - Quantization    : ChromaDB handles prototype-based quantization internally
  - Two collections : one for questions, one for chunks
  - Link key        : chunk_id (foreign key between collections)

SETUP:
  pip install chromadb sentence-transformers torch
"""

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_FILE          = Path("data/corpus.jsonl")
CHROMA_DIR           = Path("data/chroma_db")
LOG_FILE             = Path("logs/indexing.log")

EMBEDDING_MODEL      = "BAAI/bge-large-en-v1.5"   # paper's exact model
QUESTION_COLLECTION  = "question_collection"        # paper's question VectorDB
CHUNK_COLLECTION     = "chunk_collection"           # paper's chunk VectorDB

BATCH_SIZE           = 64    # embed this many items at once (GPU memory safe)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Embedding helper ──────────────────────────────────────────────────────────
class BGEEmbedder:
    """
    Wraps BAAI/bge-large-en-v1.5.
    The BGE model requires a specific instruction prefix for queries
    (not for documents/questions being indexed).
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        log.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        log.info("Model loaded.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts (questions or chunks)."""
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine similarity needs unit vectors
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query (with BGE instruction prefix)."""
        text = self.QUERY_PREFIX + query
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


# ── ChromaDB setup ────────────────────────────────────────────────────────────
def get_chroma_client() -> chromadb.Client:
    # Connect to remote ChromaDB server (containerized)
    client = chromadb.HttpClient(host="localhost", port=8000)
    log.info("ChromaDB HTTP client connected to http://localhost:8000")
    return client


def get_or_create_collection(
    client: chromadb.Client, name: str
) -> chromadb.Collection:
    """
    Get existing collection or create a new one.
    Uses cosine similarity — matches the paper's quantization metric.
    """
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},   # paper uses cosine similarity
    )
    log.info("Collection '%s' ready (count=%d)", name, collection.count())
    return collection


# ── Load corpus ───────────────────────────────────────────────────────────────
def load_corpus(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d corpus records from %s", len(records), path)
    return records


# ── Index questions ───────────────────────────────────────────────────────────
def index_questions(
    records: list[dict],
    embedder: BGEEmbedder,
    collection: chromadb.Collection,
) -> int:
    """
    For each chunk, embed every generated question and add to
    question_collection with chunk_id as metadata.

    This builds the inverted index:
      question_embedding → chunk_id → chunk_text
    """
    already_indexed = set(collection.get()["ids"]) if collection.count() > 0 else set()

    all_ids        : list[str]       = []
    all_texts      : list[str]       = []
    all_metadatas  : list[dict]      = []

    for rec in records:
        chunk_id = rec["chunk_id"]
        for q_idx, question in enumerate(rec["questions"]):
            q_id = f"{chunk_id}_q{q_idx:03d}"
            if q_id in already_indexed:
                continue
            all_ids.append(q_id)
            all_texts.append(question)
            all_metadatas.append({
                "chunk_id"  : chunk_id,
                "source_url": rec["source_url"],
            })

    if not all_ids:
        log.info("All questions already indexed in question_collection.")
        return 0

    log.info("Embedding %d questions...", len(all_ids))

    # Process in batches
    total_added = 0
    for start in range(0, len(all_ids), BATCH_SIZE):
        end        = min(start + BATCH_SIZE, len(all_ids))
        batch_ids  = all_ids[start:end]
        batch_txts = all_texts[start:end]
        batch_meta = all_metadatas[start:end]

        embeddings = embedder.embed_documents(batch_txts)

        collection.add(
            ids        = batch_ids,
            embeddings = embeddings,
            documents  = batch_txts,
            metadatas  = batch_meta,
        )
        total_added += len(batch_ids)
        log.info("  Questions indexed: %d / %d", total_added, len(all_ids))

    log.info("Question indexing complete. Total: %d", total_added)
    return total_added


# ── Index chunks ──────────────────────────────────────────────────────────────
def index_chunks(
    records: list[dict],
    embedder: BGEEmbedder,
    collection: chromadb.Collection,
) -> int:
    """
    Embed each chunk's raw text and add to chunk_collection.
    chunk_id is both the ChromaDB document ID and the foreign key
    used to retrieve chunks from question matches.
    """
    already_indexed = set(collection.get()["ids"]) if collection.count() > 0 else set()

    all_ids       : list[str]  = []
    all_texts     : list[str]  = []
    all_metadatas : list[dict] = []

    for rec in records:
        chunk_id = rec["chunk_id"]
        if chunk_id in already_indexed:
            continue
        all_ids.append(chunk_id)
        all_texts.append(rec["text"])
        all_metadatas.append({"source_url": rec["source_url"]})

    if not all_ids:
        log.info("All chunks already indexed in chunk_collection.")
        return 0

    log.info("Embedding %d chunks...", len(all_ids))

    total_added = 0
    for start in range(0, len(all_ids), BATCH_SIZE):
        end        = min(start + BATCH_SIZE, len(all_ids))
        batch_ids  = all_ids[start:end]
        batch_txts = all_texts[start:end]
        batch_meta = all_metadatas[start:end]

        embeddings = embedder.embed_documents(batch_txts)

        collection.add(
            ids        = batch_ids,
            embeddings = embeddings,
            documents  = batch_txts,
            metadatas  = batch_meta,
        )
        total_added += len(batch_ids)
        log.info("  Chunks indexed: %d / %d", total_added, len(all_ids))

    log.info("Chunk indexing complete. Total: %d", total_added)
    return total_added


# ── Main ──────────────────────────────────────────────────────────────────────
def run_indexing():
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"{CORPUS_FILE} not found. Run step3_question_generation.py first."
        )

    # Load corpus
    records = load_corpus(CORPUS_FILE)

    # Init embedder
    embedder = BGEEmbedder()

    # Init ChromaDB
    client = get_chroma_client()
    q_col  = get_or_create_collection(client, QUESTION_COLLECTION)
    c_col  = get_or_create_collection(client, CHUNK_COLLECTION)

    # Build both collections
    log.info("=== Indexing questions (inverted index) ===")
    q_count = index_questions(records, embedder, q_col)

    log.info("=== Indexing chunks (lookup store) ===")
    c_count = index_chunks(records, embedder, c_col)

    # Final stats
    log.info(
        "Indexing done. Questions in DB: %d | Chunks in DB: %d",
        q_col.count(), c_col.count()
    )
    print(f"\n✓ Phase 2 complete.")
    print(f"  Questions indexed : {q_col.count()}")
    print(f"  Chunks indexed    : {c_col.count()}")
    print(f"  ChromaDB saved to : {CHROMA_DIR}")
    print(f"\nNext → run Phase 3: step5_retrieval.py")


if __name__ == "__main__":
    run_indexing()
