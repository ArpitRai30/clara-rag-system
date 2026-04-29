"""
QuIM-RAG Extension - Step 10: Confidence Gate + Output Router
===============================================================
Full research pipeline with five modules:
  1) QuIM-RAG enhanced retrieval (existing)
  2) Constrained answer generation (existing, strict prompt)
  3) Atomic claim decomposition (new)
  4) Per-claim semantic verification (new)
  5) Confidence gate and routing (new)

Decision rule:
  - confidence_score >= threshold: return verified answer + per-claim citations
  - confidence_score < threshold : abstain + failed claim reasons

SETUP:
  pip install sentence-transformers chromadb requests python-dotenv

Run:
  python step10_confidence_router.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from step6_pipeline import QuIMRetriever, LocalLlamaGenerator, generate_answer
from step8_claim_decomposition import decompose_answer_into_claims
from step9_claim_verification import ClaimVerifier

# Load env
load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", "0.75"))
USE_WEIGHTED_SCORE = os.getenv("USE_WEIGHTED_SCORE", "true").lower() == "true"
OUTPUT_DIR = Path("data/verification_runs")


class ConfidenceRouter:
    """Routes final output based on claim-verification confidence."""

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def route(self, answer: str, verification: dict) -> dict:
        score_key = "weighted_confidence_score" if USE_WEIGHTED_SCORE else "confidence_score"
        score = float(verification[score_key])

        unsupported = [
            v for v in verification["verifications"]
            if v["label"] != "SUPPORTED"
        ]

        if score >= self.threshold:
            return {
                "status": "VERIFIED",
                "score_key": score_key,
                "score": round(score, 4),
                "answer": answer,
                "citations": [
                    {
                        "claim": v["claim"],
                        "source_url": v["source_url"],
                        "chunk_id": v["chunk_id"],
                        "similarity": v["similarity"],
                        "label": v["label"],
                    }
                    for v in verification["verifications"]
                ],
                "failed_claims": [],
            }

        return {
            "status": "ABSTAIN",
            "score_key": score_key,
            "score": round(score, 4),
            "answer": (
                "I cannot confidently return this answer because one or more "
                "claims were not verified against retrieved context."
            ),
            "citations": [],
            "failed_claims": [
                {
                    "claim": v["claim"],
                    "label": v["label"],
                    "reason": v["reason"],
                    "source_url": v["source_url"],
                    "similarity": v["similarity"],
                }
                for v in unsupported
            ],
        }


def run_confidence_routed_pipeline() -> None:
    print("\n" + "=" * 70)
    print("Trust-RAG Pipeline - Retrieve -> Generate -> Verify -> Route")
    print("=" * 70)

    retriever = QuIMRetriever()
    generator = LocalLlamaGenerator()
    verifier = ClaimVerifier()
    router = ConfidenceRouter()

    print(f"Confidence threshold: {router.threshold:.2f}")
    print("Type your question below. Type 'quit' to exit.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        print("-" * 70)
        user_query = input("You: ").strip()

        if not user_query:
            continue

        if user_query.lower() in ("quit", "exit", "q"):
            print("Exiting Trust-RAG pipeline. Goodbye!")
            break

        print("\n[1/5] Retrieving context chunks...")
        retrieved = retriever.retrieve(user_query)
        if not retrieved:
            print("No relevant chunks found.\n")
            continue

        print("[2/5] Generating constrained answer...")
        try:
            answer = generate_answer(generator, user_query, retrieved)
        except RuntimeError as exc:
            print(f"Generation failed: {exc}")
            print("Routing this query to ABSTAIN due to generation failure.")

            verification = {
                "verifications": [
                    {
                        "claim": "Answer generation failed.",
                        "label": "NOT_FOUND",
                        "similarity": 0.0,
                        "source_url": "",
                        "chunk_id": "",
                        "matched_question": "",
                        "reason": str(exc),
                    }
                ],
                "confidence_score": 0.0,
                "weighted_confidence_score": 0.0,
            }

            routed = {
                "status": "ABSTAIN",
                "score_key": "confidence_score",
                "score": 0.0,
                "answer": "I could not generate a stable answer because the local model failed during inference.",
                "citations": [],
                "failed_claims": [
                    {
                        "claim": "Answer generation failed.",
                        "label": "NOT_FOUND",
                        "reason": str(exc),
                        "source_url": "",
                        "similarity": 0.0,
                    }
                ],
            }

            run_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "question": user_query,
                "retrieved_count": len(retrieved),
                "answer": "",
                "claims": [],
                "verification": verification,
                "decision": routed,
            }
            output_file = OUTPUT_DIR / "last_run.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(run_record, f, indent=2, ensure_ascii=False)

            print("\n" + "=" * 70)
            print(f"Status: {routed['status']}")
            print(f"{routed['score_key']}: {routed['score']}")
            print("\nAbstention reason:")
            print(routed["answer"])
            print("\nFailed claims:")
            for i, f in enumerate(routed["failed_claims"], 1):
                print(
                    f"  [{i}] {f['label']} | sim={f['similarity']} | "
                    f"{f['source_url']}"
                )
                print(f"      Claim : {f['claim']}")
                print(f"      Reason: {f['reason']}")

            print(f"\nSaved run report to {output_file}\n")
            continue

        print("[3/5] Decomposing answer into atomic claims...")
        claims = decompose_answer_into_claims(answer)

        print("[4/5] Verifying each claim against context...")
        verification = verifier.verify_claims(claims, retrieved)

        print("[5/5] Applying confidence gate + output router...")
        routed = router.route(answer, verification)

        print("\n" + "=" * 70)
        print(f"Status: {routed['status']}")
        print(f"{routed['score_key']}: {routed['score']}")

        if routed["status"] == "VERIFIED":
            print("\nAnswer:")
            print(routed["answer"])
            print("\nPer-claim citations:")
            for i, c in enumerate(routed["citations"], 1):
                print(
                    f"  [{i}] {c['label']} | sim={c['similarity']} | "
                    f"{c['source_url']}"
                )
                print(f"      Claim: {c['claim']}")
        else:
            print("\nAbstention reason:")
            print(routed["answer"])
            print("\nFailed claims:")
            for i, f in enumerate(routed["failed_claims"], 1):
                print(
                    f"  [{i}] {f['label']} | sim={f['similarity']} | "
                    f"{f['source_url']}"
                )
                print(f"      Claim : {f['claim']}")
                print(f"      Reason: {f['reason']}")

        run_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": user_query,
            "retrieved_count": len(retrieved),
            "answer": answer,
            "claims": claims,
            "verification": verification,
            "decision": routed,
        }
        output_file = OUTPUT_DIR / "last_run.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2, ensure_ascii=False)

        print(f"\nSaved run report to {output_file}\n")


if __name__ == "__main__":
    run_confidence_routed_pipeline()
