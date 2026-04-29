"""
QuIM-RAG Phase 3 & 4 - Step 6, 7, 8: Generation Pipeline
==========================================================
Implements the full QuIM-RAG pipeline:
    User query → Retrieve chunks → Generate answer via local Llama

Paper spec:
    - LLM        : Meta-LLaMA3-8B-Instruct
    - Context    : top 3 retrieved chunks
    - Prompt     : custom prompt (paper Figure 4 lower section)
    - OOD guard  : if answer not in context, say so explicitly
    - Output     : answer + source links

This version runs fully locally with Ollama instead of Groq.

SETUP:
    1. pip install chromadb sentence-transformers python-dotenv

    2. Make sure the local ChromaDB store exists:
             data/chroma_db/

    3. Pull a model in Ollama and set the model name in .env:
             OLLAMA_MODEL=llama3:8b

    4. Run:
             python step6_pipeline.py
"""

import os
import sys
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import requests

# ── Load environment variables from .env file ─────────────────────────────────
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────


OLLAMA_HOST          = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_TIMEOUT_SEC   = int(os.getenv("OLLAMA_TIMEOUT_SEC", "300"))
OLLAMA_RETRIES       = int(os.getenv("OLLAMA_RETRIES", "2"))
OLLAMA_NUM_PREDICT   = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

# Local ChromaDB persistent store
CHROMA_DIR           = Path("data/chroma_db")

# Collections
QUESTION_COLLECTION  = "question_collection"
CHUNK_COLLECTION     = "chunk_collection"

# Embedding model (same as used in step4_build_index.py)
EMBEDDING_MODEL      = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX         = "Represent this sentence for searching relevant passages: "

# LLM settings
TOP_K                = 3       # paper: top 3 chunks
MAX_TOKENS           = 1024
TEMPERATURE          = 0.1     # low = more factual, less creative


# ── Step 6: Retriever ─────────────────────────────────────────────────────────
class QuIMRetriever:
    """
    Question-to-Question Inverted Index Matching retrieval.
    Connects to ChromaDB running in Docker container via HTTP.
    Exact implementation from paper Section III-B.
    """

    def __init__(self):
        print("Loading embedding model (BAAI/bge-large-en-v1.5)...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {CHROMA_DIR.resolve()}"
            )

        print(f"Opening local ChromaDB store at {CHROMA_DIR}...")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        self.q_col     = client.get_collection(QUESTION_COLLECTION)
        self.chunk_col = client.get_collection(CHUNK_COLLECTION)

        print(f"Questions in DB : {self.q_col.count()}")
        print(f"Chunks in DB    : {self.chunk_col.count()}")

    def retrieve(self, user_query: str, top_k: int = TOP_K) -> list[dict]:
        # Embed query with BGE instruction prefix
        query_emb = self.model.encode(
            [QUERY_PREFIX + user_query],
            normalize_embeddings=True,
        )[0].tolist()

        # Q-to-Q matching via inverted index
        q_results = self.q_col.query(
            query_embeddings = [query_emb],
            n_results        = top_k,
            include          = ["documents", "metadatas", "distances"],
        )

        matched_questions = q_results["documents"][0]
        metadatas         = q_results["metadatas"][0]
        distances         = q_results["distances"][0]

        # Deduplicate by chunk_id
        seen, pairs = set(), []
        for q, meta, dist in zip(matched_questions, metadatas, distances):
            cid = meta["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                pairs.append((cid, q, round(1.0 - dist, 4)))

        # Fetch chunk texts
        chunk_ids    = [p[0] for p in pairs]
        chunk_result = self.chunk_col.get(
            ids     = chunk_ids,
            include = ["documents", "metadatas"],
        )

        lookup = {
            cid: {"text": doc, "source_url": meta["source_url"]}
            for cid, doc, meta in zip(
                chunk_result["ids"],
                chunk_result["documents"],
                chunk_result["metadatas"],
            )
        }

        results = []
        for chunk_id, matched_q, score in pairs:
            if chunk_id in lookup:
                results.append({
                    "chunk_id"         : chunk_id,
                    "matched_question" : matched_q,
                    "similarity_score" : score,
                    "chunk_text"       : lookup[chunk_id]["text"],
                    "source_url"       : lookup[chunk_id]["source_url"],
                })

        return results


class LocalLlamaGenerator:
    """Calls a local Ollama model via HTTP."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model
        print(f"Using Ollama at {self.host} with model '{self.model}'.")

    def generate(self, user_query: str, retrieved_chunks: list[dict]) -> str:
        url = f"{self.host}/api/generate"
        last_error = "Unknown generation error"

        for attempt in range(1, max(1, OLLAMA_RETRIES) + 1):
            # On retry, reduce model pressure and shorten context to improve stability.
            slim_chunks = retrieved_chunks if attempt == 1 else retrieved_chunks[:1]
            prompt = build_rag_prompt(user_query, slim_chunks)
            num_predict = max(128, OLLAMA_NUM_PREDICT // (2 ** (attempt - 1)))

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": num_predict,
                },
            }

            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=OLLAMA_TIMEOUT_SEC,
                )
            except requests.RequestException as exc:
                last_error = f"Failed to reach Ollama at {self.host}: {exc}"
                if attempt < OLLAMA_RETRIES:
                    time.sleep(1.0)
                    continue
                raise RuntimeError(last_error) from exc

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()

            body = response.text.strip()
            last_error = f"Ollama error {response.status_code}: {body}"

            # Ollama runner termination is usually transient or memory-related.
            is_runner_terminated = (
                response.status_code >= 500
                and "runner process has terminated" in body.lower()
            )
            if is_runner_terminated and attempt < OLLAMA_RETRIES:
                time.sleep(1.5)
                continue

            break

        raise RuntimeError(
            f"{last_error}\n"
            "Tip: reduce model load (smaller model or lower OLLAMA_NUM_PREDICT), "
            "or verify Ollama can run a quick prompt with this model."
        )


# ── Step 7: Prompt + Answer generation ───────────────────────────────────────
def build_rag_prompt(user_query: str, retrieved_chunks: list[dict]) -> str:
    """
    Implements the RAG prompt from paper Figure 4 (lower section).
    Two key behaviours from paper:
      1. Answer strictly from provided context
      2. OOD guard — if not in context, say so explicitly
    """
    context_blocks = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_blocks += (
            f"[Context {i}]\n"
            f"{chunk['chunk_text']}\n"
            f"Source: {chunk['source_url']}\n\n"
        )

    prompt = (
        "You are a helpful assistant that answers questions strictly based "
        "on the provided context from NDSU (North Dakota State University) documents.\n\n"
        "RULES:\n"
        "1. Answer ONLY using information from the context below\n"
        "2. If the context does not contain enough information to answer "
        "the question, say: 'I don't have enough information in my knowledge "
        "base to answer this question accurately.'\n"
        "3. Always include the source link(s) at the end of your answer\n"
        "4. Be concise and direct\n\n"
        f"CONTEXT:\n{context_blocks}"
        f"QUESTION: {user_query}\n\n"
        "ANSWER:"
    )

    return prompt


def generate_answer(generator: LocalLlamaGenerator, user_query: str,
                    retrieved_chunks: list[dict]) -> str:
    """
    Step 7: Generate answer using a local Llama model.
    Paper Section III-C.
    """
    return generator.generate(user_query, retrieved_chunks)


# ── Step 8: Full pipeline + terminal UI ──────────────────────────────────────
def run_pipeline():
    # Init components
    print("\n" + "="*60)
    print("  QuIM-RAG — Question Inverted Index Matching RAG")
    print("="*60)

    retriever = QuIMRetriever()
    generator = LocalLlamaGenerator()

    print(f"\nLLM : Ollama {OLLAMA_MODEL}")
    print(f"DB  : ChromaDB at {CHROMA_DIR}")
    print("\nType your question below. Type 'quit' to exit.\n")

    # Terminal chat loop
    while True:
        print("-" * 60)
        user_query = input("You: ").strip()

        if not user_query:
            continue

        if user_query.lower() in ("quit", "exit", "q"):
            print("Exiting QuIM-RAG. Goodbye!")
            break

        # Retrieve
        print("\nRetrieving relevant chunks...")
        retrieved = retriever.retrieve(user_query)

        if not retrieved:
            print("No relevant chunks found for your query.\n")
            continue

        # Show matched questions (useful for research/debugging)
        print(f"\nMatched {len(retrieved)} chunk(s):")
        for i, r in enumerate(retrieved, 1):
            print(f"  [{i}] Q     : {r['matched_question']}")
            print(f"       Score : {r['similarity_score']}")
            print(f"       URL   : {r['source_url']}")

        # Generate answer
        print("\nGenerating answer...\n")
        answer = generate_answer(generator, user_query, retrieved)

        # Display
        print("=" * 60)
        print(f"Answer:\n{answer}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()