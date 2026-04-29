"""
QuIM-RAG Extension - Step 8: Atomic Claim Decomposition
========================================================
Converts a generated answer into atomic, verifiable claims.

Module mapping:
  - Module 3 (novel): split one answer into one-fact-per-claim sentences.

Approach:
  1. Ask local Ollama model to rewrite answer as atomic claims
  2. Parse numbered output into a clean list
  3. Fallback to deterministic sentence splitting if model output is noisy

SETUP:
  pip install requests python-dotenv

Run:
  python step8_claim_decomposition.py
"""

import os
import re
from dotenv import load_dotenv
import requests

# Load env
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
MAX_CLAIMS = int(os.getenv("MAX_ATOMIC_CLAIMS", "12"))


def _fallback_sentence_split(answer: str) -> list[str]:
    """Conservative fallback when LLM output cannot be parsed reliably."""
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = []
    for p in parts:
        clean = p.strip().strip("-• ")
        if clean:
            claims.append(clean)
    return claims[:MAX_CLAIMS]


def _parse_claims(raw_text: str) -> list[str]:
    """Parse claims from numbered or bulleted LLM output."""
    claims = []
    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            continue
        text = re.sub(r"^\d+[\).:-]?\s*", "", text)
        text = re.sub(r"^[\-*•]\s*", "", text)
        text = text.strip().strip('"')
        if text:
            claims.append(text)

    unique = []
    seen = set()
    for claim in claims:
        key = claim.lower()
        if key not in seen:
            seen.add(key)
            unique.append(claim)

    return unique[:MAX_CLAIMS]


def decompose_answer_into_claims(
    answer: str,
    host: str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
) -> list[str]:
    """
    Module 3: Decompose answer into atomic claims.

    Returns list of one-fact-per-claim strings.
    """
    answer = answer.strip()
    if not answer:
        return []

    prompt = (
        "You are a fact decomposition assistant. "
        "Rewrite the answer into atomic factual claims.\n\n"
        "Rules:\n"
        "1. Each claim must contain exactly one checkable fact\n"
        "2. Keep original meaning; do not add new facts\n"
        "3. Output plain numbered lines only (no commentary)\n"
        f"4. Maximum {MAX_CLAIMS} claims\n\n"
        f"Answer:\n{answer}\n\n"
        "Atomic claims:"
    )

    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        raw = response.json().get("response", "")
        claims = _parse_claims(raw)
        if claims:
            return claims
    except requests.RequestException:
        pass

    return _fallback_sentence_split(answer)


def _demo() -> None:
    print("=" * 70)
    print("Step 8 - Atomic Claim Decomposition")
    print("=" * 70)

    answer = input("Paste an answer to decompose:\n\n").strip()
    claims = decompose_answer_into_claims(answer)

    print("\nAtomic claims:\n")
    for i, claim in enumerate(claims, 1):
        print(f"  [{i}] {claim}")


if __name__ == "__main__":
    _demo()
