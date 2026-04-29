"""
QuIM-RAG Phase 2 / Phase 3 - Step 5: Retrieval
================================================
This module is the CORE of QuIM-RAG.
Used at query time (Phase 3) but set up during Phase 2.

Paper spec:
  1. Encode user query with BAAI/bge-large-en-v1.5
  2. Search question_collection with cosine similarity
  3. Retrieve top k=3 most similar questions
  4. For each matched question, get its chunk_id
  5. Fetch chunk text + source_url from chunk_collection
  6. Return deduplicated chunks as context for the LLM

This file can be run standalone to test retrieval
before wiring in the LLM.
"""

import json
import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
LOG_FILE             = Path("logs/retrieval.log")
CHROMA_DIR           = Path("data/chroma_db")

EMBEDDING_MODEL      = "BAAI/bge-large-en-v1.5"
QUESTION_COLLECTION  = "question_collection"
CHUNK_COLLECTION     = "chunk_collection"

TOP_K                = 3    # paper: retrieve top 3 matching questions

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


# ── Retriever class ───────────────────────────────────────────────────────────
class QuIMRetriever:
    """
    Implements the Question-to-Question Inverted Index Matching retrieval.

    Usage:
        retriever = QuIMRetriever()
        results   = retriever.retrieve("What clubs are available at NDSU?")
        for r in results:
            print(r['matched_question'])
            print(r['chunk_text'])
            print(r['source_url'])
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self):
        log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {CHROMA_DIR.resolve()}"
            )

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.q_col = client.get_collection(QUESTION_COLLECTION)
        self.chunk_col = client.get_collection(CHUNK_COLLECTION)

        log.info(
            "Retriever ready on %s. Questions: %d | Chunks: %d",
            CHROMA_DIR,
            self.q_col.count(), self.chunk_col.count()
        )

    def _embed_query(self, query: str) -> list[float]:
        """Embed user query with BGE instruction prefix."""
        text = self.QUERY_PREFIX + query
        emb  = self.model.encode([text], normalize_embeddings=True)
        return emb[0].tolist()

    def retrieve(self, user_query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Core QuIM-RAG retrieval:
          query → embed → match against question_collection
                        → get chunk_ids → fetch chunks

        Returns list of dicts (deduplicated by chunk_id):
          [
            {
              "chunk_id"        : "chunk_00042",
              "matched_question": "What is the mission of the NDSU career center?",
              "similarity_score": 0.91,
              "chunk_text"      : "The NDSU Career and Advising Center...",
              "source_url"      : "https://career-advising.ndsu.edu/...",
            },
            ...
          ]
        """
        log.info("Query: %s", user_query)

        # Step 1 — embed the query
        query_embedding = self._embed_query(user_query)

        # Step 2 — Q-to-Q matching (the inverted index lookup)
        q_results = self.q_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        matched_questions = q_results["documents"][0]
        metadatas         = q_results["metadatas"][0]
        distances         = q_results["distances"][0]

        # Step 3 — collect unique chunk_ids (multiple questions may map
        #           to the same chunk — the paper observed this)
        seen_chunk_ids = set()
        ordered_pairs  = []   # preserve ranking order

        for question, meta, dist in zip(
            matched_questions, metadatas, distances
        ):
            chunk_id = meta["chunk_id"]
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                # Convert distance to similarity (ChromaDB cosine distance = 1 - similarity)
                similarity = 1.0 - dist
                ordered_pairs.append((chunk_id, question, similarity))

        # Step 4 — fetch chunk texts from chunk_collection
        chunk_ids = [p[0] for p in ordered_pairs]
        if not chunk_ids:
            log.warning("No chunks retrieved for query: %s", user_query)
            return []

        chunk_results = self.chunk_col.get(
            ids     = chunk_ids,
            include = ["documents", "metadatas"],
        )

        # Build a lookup from chunk_id → text + url
        chunk_lookup = {}
        for cid, doc, meta in zip(
            chunk_results["ids"],
            chunk_results["documents"],
            chunk_results["metadatas"],
        ):
            chunk_lookup[cid] = {"text": doc, "source_url": meta["source_url"]}

        # Step 5 — assemble final results
        results = []
        for chunk_id, matched_q, similarity in ordered_pairs:
            if chunk_id not in chunk_lookup:
                continue
            results.append({
                "chunk_id"        : chunk_id,
                "matched_question": matched_q,
                "similarity_score": round(similarity, 4),
                "chunk_text"      : chunk_lookup[chunk_id]["text"],
                "source_url"      : chunk_lookup[chunk_id]["source_url"],
            })

        log.info("Retrieved %d unique chunks.", len(results))
        return results


# ── Standalone test ───────────────────────────────────────────────────────────
def test_retrieval():
    """Run a few sample queries to verify the index is working."""
    retriever = QuIMRetriever()

    test_queries = [
        "What is the location of the NDSU Career and Advising Center?",
        "What clubs are available for students at NDSU?",
        "What are the graduation requirements for computer science?",
        "How do I apply for internships through NDSU career advising?",
    ]

    for query in test_queries:
        print("\n" + "="*70)
        print(f"QUERY: {query}")
        print("="*70)

        results = retriever.retrieve(query)

        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] Matched Question : {r['matched_question']}")
            print(f"      Similarity Score : {r['similarity_score']}")
            print(f"      Source URL       : {r['source_url']}")
            print(f"      Chunk preview    : {r['chunk_text'][:200]}...")


if __name__ == "__main__":
    test_retrieval()
