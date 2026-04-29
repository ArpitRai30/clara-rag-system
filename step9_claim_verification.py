"""
QuIM-RAG Extension - Step 9: Per-Claim Semantic Verification
=============================================================
Verifies each atomic claim against retrieved context chunks.

Module mapping:
  - Module 4 (primary novel):
      claim -> semantic similarity -> LLM judge label
      labels: SUPPORTED / CONTRADICTED / NOT_FOUND

Scoring:
  - confidence_score          = supported_claims / total_claims
  - weighted_confidence_score = weighted support by similarity

SETUP:
  pip install sentence-transformers requests python-dotenv

Run:
  python step9_claim_verification.py
"""

import math
import os
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer

# Load env
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
TOP_EVIDENCE = int(os.getenv("CLAIM_TOP_EVIDENCE", "3"))


class ClaimVerifier:
    """Verifies claims using semantic retrieval and an LLM judge."""

    def __init__(self):
        print(f"Loading verifier embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _parse_verdict(text: str) -> str:
        t = text.upper()
        if "SUPPORTED" in t:
            return "SUPPORTED"
        if "CONTRADICTED" in t:
            return "CONTRADICTED"
        return "NOT_FOUND"

    def _judge_claim(self, claim: str, evidence_text: str) -> tuple[str, str]:
        prompt = (
            "You are a strict verification judge.\n"
            "Classify the claim against context as exactly one of:\n"
            "SUPPORTED, CONTRADICTED, NOT_FOUND\n\n"
            "Return in this format:\n"
            "VERDICT: <SUPPORTED|CONTRADICTED|NOT_FOUND>\n"
            "REASON: <one short sentence>\n\n"
            f"CLAIM: {claim}\n\n"
            f"CONTEXT:\n{evidence_text}\n"
        )

        url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 200,
            },
        }

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        raw = response.json().get("response", "")

        verdict = self._parse_verdict(raw)
        reason = raw.strip().splitlines()[-1] if raw.strip() else "No reason returned."
        return verdict, reason

    def verify_claims(
        self,
        claims: list[str],
        retrieved_chunks: list[dict],
        top_evidence: int = TOP_EVIDENCE,
    ) -> dict:
        """
        Verify each claim against retrieved chunks.

        Returns:
          {
            "verifications": [...],
            "confidence_score": 0.0-1.0,
            "weighted_confidence_score": 0.0-1.0,
          }
        """
        if not claims:
            return {
                "verifications": [],
                "confidence_score": 0.0,
                "weighted_confidence_score": 0.0,
            }

        if not retrieved_chunks:
            return {
                "verifications": [
                    {
                        "claim": c,
                        "label": "NOT_FOUND",
                        "similarity": 0.0,
                        "source_url": "",
                        "chunk_id": "",
                        "reason": "No retrieved context available.",
                    }
                    for c in claims
                ],
                "confidence_score": 0.0,
                "weighted_confidence_score": 0.0,
            }

        chunk_texts = [c["chunk_text"] for c in retrieved_chunks]
        chunk_embeddings = self.embedder.encode(
            chunk_texts,
            normalize_embeddings=True,
        ).tolist()

        claim_embeddings = self.embedder.encode(
            claims,
            normalize_embeddings=True,
        ).tolist()

        verifications = []
        for claim, claim_emb in zip(claims, claim_embeddings):
            scored = []
            for idx, chunk_emb in enumerate(chunk_embeddings):
                sim = self._cosine(claim_emb, chunk_emb)
                scored.append((sim, idx))

            scored.sort(reverse=True, key=lambda x: x[0])
            top = scored[: max(1, top_evidence)]

            evidence_chunks = [retrieved_chunks[idx] for _, idx in top]
            evidence_text = "\n\n".join(
                f"[Evidence {i}]\n{c['chunk_text']}\nSource: {c['source_url']}"
                for i, c in enumerate(evidence_chunks, 1)
            )

            best_sim, best_idx = top[0]
            best_chunk = retrieved_chunks[best_idx]

            try:
                label, reason = self._judge_claim(claim, evidence_text)
            except requests.RequestException:
                label = "NOT_FOUND"
                reason = "Verification call failed; defaulted to NOT_FOUND."

            verifications.append(
                {
                    "claim": claim,
                    "label": label,
                    "similarity": round(float(best_sim), 4),
                    "source_url": best_chunk.get("source_url", ""),
                    "chunk_id": best_chunk.get("chunk_id", ""),
                    "matched_question": best_chunk.get("matched_question", ""),
                    "reason": reason,
                }
            )

        supported = [v for v in verifications if v["label"] == "SUPPORTED"]
        confidence = len(supported) / len(verifications)

        total_weight = sum(max(v["similarity"], 0.0) for v in verifications)
        supported_weight = sum(
            max(v["similarity"], 0.0)
            for v in verifications
            if v["label"] == "SUPPORTED"
        )
        weighted_confidence = (
            supported_weight / total_weight if total_weight > 0 else 0.0
        )

        return {
            "verifications": verifications,
            "confidence_score": round(confidence, 4),
            "weighted_confidence_score": round(weighted_confidence, 4),
        }


def _demo() -> None:
    verifier = ClaimVerifier()
    claims = [
        "The NDSU Career Center is in Ceres Hall.",
        "The center is open on weekends.",
    ]
    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "matched_question": "Where is the career center located?",
            "chunk_text": "The NDSU Career and Advising Center is located on the second floor of Ceres Hall.",
            "source_url": "https://example.edu/career",
        }
    ]

    out = verifier.verify_claims(claims, retrieved_chunks)
    print("Verification summary:")
    print(out)


if __name__ == "__main__":
    _demo()
