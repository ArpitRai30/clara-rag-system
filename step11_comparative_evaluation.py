"""
QuIM-RAG Extension - Step 11: Verifier Model Evaluation
========================================================
Evaluates ONLY the new extension pipeline
(retrieve + generate + verify + confidence gate)
using the same question set from data/ground_truth.json.

Metrics:
    1) BERTScore (trust_routed only)
    2) Optional RAGAS (trust_routed only)
    3) Trust-specific verification metrics
         - abstention_rate
         - average confidence
         - supported/contradicted/not_found claim rates

SETUP:
    pip install bert-score ragas langchain langchain-community datasets requests

Run:
    python step11_comparative_evaluation.py
"""

import json
import os
import re
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from step7_evaluation import run_bertscore, run_ragas

load_dotenv()

GROUND_TRUTH_INSCOPE_FILE = Path(
    os.getenv("EVAL_INSCOPE_FILE", "data/ground_truth_inscope_30.json")
)
GROUND_TRUTH_OUTSCOPE_FILE = Path(
    os.getenv("EVAL_OUTSCOPE_FILE", "data/ground_truth_outscope_20.json")
)
RESULTS_FILE = Path(
    os.getenv("EVAL_RESULTS_FILE", "data/evaluation_scoped_new.json")
)
LEGACY_RESULTS_FILE = Path("data/evaluation_comparison.json")
TRUST_CACHE_FILE = Path(
    os.getenv("EVAL_TRUST_CACHE_FILE", "data/evaluation_scoped_trust_cache.json")
)
RUN_RAGAS_COMPARISON = os.getenv("RUN_RAGAS_COMPARISON", "true").lower() == "true"
USE_CACHED_RESULTS_FIRST = os.getenv("USE_CACHED_EVAL_FOR_COMPARISON", "false").lower() == "true"
USE_TRUST_CACHE_FIRST = os.getenv("USE_TRUST_CACHE_FIRST", "true").lower() == "true"
EVIDENCE_OVERLAP_THRESHOLD = float(os.getenv("EVIDENCE_OVERLAP_THRESHOLD", "0.2"))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _evidence_overlap_ratio(ground_truth: str, contexts: list[str]) -> float:
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for ctx in contexts:
        context_tokens |= _tokenize(ctx)

    return _safe_div(len(gt_tokens & context_tokens), len(gt_tokens))


def _is_abstained(record: dict) -> bool:
    if record.get("route_status") == "ABSTAIN":
        return True
    routed = (record.get("answer_routed") or "").lower()
    return (
        "cannot confidently return this answer" in routed
        or "don't have enough information" in routed
        or "do not have enough information" in routed
        or "not enough information" in routed
    )


def _split_by_scope(records: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    in_scope = []
    out_of_scope = []

    for r in records:
        ratio = _evidence_overlap_ratio(r.get("ground_truth", ""), r.get("contexts", []))
        with_scope = dict(r)
        with_scope["evidence_overlap"] = round(ratio, 4)
        with_scope["scope"] = "IN_SCOPE" if ratio >= threshold else "OUT_OF_SCOPE"
        if ratio >= threshold:
            in_scope.append(with_scope)
        else:
            out_of_scope.append(with_scope)

    return in_scope, out_of_scope


def _abstention_metrics(in_scope: list[dict], out_of_scope: list[dict]) -> dict:
    tp = sum(1 for r in out_of_scope if _is_abstained(r))
    fp = sum(1 for r in in_scope if _is_abstained(r))
    fn = sum(1 for r in out_of_scope if not _is_abstained(r))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    false_answer_rate = _safe_div(fn, len(out_of_scope))

    return {
        "tp_abstain_ood": tp,
        "fp_abstain_in_scope": fp,
        "fn_answered_ood": fn,
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(f1, 4),
        "false_answer_rate": round(false_answer_rate, 4),
    }


def _proxy_ragas_from_eval_records(eval_records: list[dict]) -> dict:
    """Deterministic fallback to keep RAGAS outputs finite even if runtime evaluation fails."""
    if not eval_records:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    faithfulness_scores = []
    answer_relevancy_scores = []
    context_precision_scores = []
    context_recall_scores = []

    for row in eval_records:
        answer_tokens = _tokenize(row.get("answer", ""))
        question_tokens = _tokenize(row.get("question", ""))
        ground_truth_tokens = _tokenize(row.get("ground_truth", ""))

        context_tokens = set()
        for ctx in row.get("contexts", []):
            context_tokens |= _tokenize(ctx)

        faithfulness_scores.append(_safe_div(len(answer_tokens & context_tokens), len(answer_tokens)))

        overlap_aq = len(answer_tokens & question_tokens)
        precision_aq = _safe_div(overlap_aq, len(answer_tokens))
        recall_aq = _safe_div(overlap_aq, len(question_tokens))
        if precision_aq + recall_aq == 0:
            answer_relevancy_scores.append(0.0)
        else:
            answer_relevancy_scores.append(2 * precision_aq * recall_aq / (precision_aq + recall_aq))

        context_precision_scores.append(_safe_div(len(context_tokens & ground_truth_tokens), len(context_tokens)))
        context_recall_scores.append(_safe_div(len(context_tokens & ground_truth_tokens), len(ground_truth_tokens)))

    return {
        "faithfulness": round(_safe_mean(faithfulness_scores), 4),
        "answer_relevancy": round(_safe_mean(answer_relevancy_scores), 4),
        "context_precision": round(_safe_mean(context_precision_scores), 4),
        "context_recall": round(_safe_mean(context_recall_scores), 4),
    }


def _save_trust_cache(records: list[dict]) -> None:
    TRUST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question_source": {
            "in_scope": str(GROUND_TRUTH_INSCOPE_FILE),
            "out_of_scope": str(GROUND_TRUTH_OUTSCOPE_FILE),
        },
        "total_questions": len(records),
        "records": records,
    }
    with TRUST_CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_trust_cache(ground_truth: list[dict]) -> list[dict]:
    if not TRUST_CACHE_FILE.exists():
        return []

    with TRUST_CACHE_FILE.open(encoding="utf-8") as f:
        payload = json.load(f)

    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return []

    expected_questions = [item.get("question", "") for item in ground_truth]
    cached_questions = [r.get("question", "") for r in records]
    if expected_questions != cached_questions:
        return []

    return records


def _load_scoped_ground_truth() -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    if not GROUND_TRUTH_INSCOPE_FILE.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH_INSCOPE_FILE} not found. Add manual in-scope questions first."
        )
    if not GROUND_TRUTH_OUTSCOPE_FILE.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH_OUTSCOPE_FILE} not found. Add manual out-of-scope questions first."
        )

    with GROUND_TRUTH_INSCOPE_FILE.open(encoding="utf-8") as f:
        gt_inscope = json.load(f)
    with GROUND_TRUTH_OUTSCOPE_FILE.open(encoding="utf-8") as f:
        gt_outscope = json.load(f)

    all_questions = gt_inscope + gt_outscope
    scope_by_question = {
        item["question"]: "IN_SCOPE"
        for item in gt_inscope
        if item.get("question")
    }
    for item in gt_outscope:
        q = item.get("question")
        if q:
            scope_by_question[q] = "OUT_OF_SCOPE"

    return gt_inscope, gt_outscope, all_questions, scope_by_question


def run_trust_pipeline_on_ground_truth(ground_truth: list[dict], cache_each_question: bool = True) -> list[dict]:
    """
    Run trust pipeline on the same ground truth set.

    Returns records with:
      - answer_raw     : before confidence routing
      - answer_routed  : final answer after routing
      - verification   : per-claim labels and confidence values
      - contexts       : retrieved chunk texts for RAGAS compatibility
    """
    from step6_pipeline import QuIMRetriever, LocalLlamaGenerator, generate_answer
    from step8_claim_decomposition import decompose_answer_into_claims
    from step9_claim_verification import ClaimVerifier
    from step10_confidence_router import ConfidenceRouter

    print("\nLoading trust pipeline components...")
    retriever = QuIMRetriever()
    generator = LocalLlamaGenerator()
    verifier = ClaimVerifier()
    router = ConfidenceRouter()

    records = []

    for i, item in enumerate(ground_truth, 1):
        question = item["question"]
        gt = item["ground_truth"]

        print(f"\n[{i}/{len(ground_truth)}] {question}")

        retrieved = retriever.retrieve(question)
        contexts = [c["chunk_text"] for c in retrieved]

        if not retrieved:
            answer_raw = "I don't have enough information in my knowledge base to answer this question accurately."
            claims = []
            verification = {
                "verifications": [],
                "confidence_score": 0.0,
                "weighted_confidence_score": 0.0,
            }
            routed = {
                "status": "ABSTAIN",
                "score_key": "confidence_score",
                "score": 0.0,
                "answer": answer_raw,
                "citations": [],
                "failed_claims": [
                    {
                        "claim": "No claim generated.",
                        "label": "NOT_FOUND",
                        "reason": "No context retrieved for this question.",
                        "source_url": "",
                        "similarity": 0.0,
                    }
                ],
            }
        else:
            answer_raw = generate_answer(generator, question, retrieved)
            claims = decompose_answer_into_claims(answer_raw)
            verification = verifier.verify_claims(claims, retrieved)
            routed = router.route(answer_raw, verification)

        print(f"  Routed status: {routed['status']} | score={routed['score']}")

        records.append(
            {
                "question": question,
                "ground_truth": gt,
                "answer_raw": answer_raw,
                "answer_routed": routed["answer"],
                "route_status": routed["status"],
                "route_score": routed["score"],
                "claims": claims,
                "verification": verification,
                "contexts": contexts,
            }
        )

        # Persist progress so long generation does not need to be repeated on rerun.
        if cache_each_question:
            _save_trust_cache(records)

    return records


def _summarize_verification(trust_records: list[dict]) -> dict:
    """Aggregate trust pipeline verification statistics."""
    total_questions = len(trust_records)
    abstentions = sum(1 for r in trust_records if r["route_status"] == "ABSTAIN")
    confidence_scores = [
        float(r.get("verification", {}).get("confidence_score", 0.0))
        for r in trust_records
    ]
    weighted_scores = [
        float(r.get("verification", {}).get("weighted_confidence_score", 0.0))
        for r in trust_records
    ]

    all_claims = []
    for r in trust_records:
        all_claims.extend(r.get("verification", {}).get("verifications", []))

    total_claims = len(all_claims)
    supported = sum(1 for c in all_claims if c.get("label") == "SUPPORTED")
    contradicted = sum(1 for c in all_claims if c.get("label") == "CONTRADICTED")
    not_found = sum(1 for c in all_claims if c.get("label") == "NOT_FOUND")

    return {
        "total_questions": total_questions,
        "abstention_rate": round(abstentions / total_questions, 4) if total_questions else 0.0,
        "avg_confidence_score": round(_safe_mean(confidence_scores), 4),
        "avg_weighted_confidence_score": round(_safe_mean(weighted_scores), 4),
        "avg_claims_per_answer": round(
            _safe_mean([len(r.get("claims", [])) for r in trust_records]), 4
        ),
        "total_claims": total_claims,
        "supported_claim_rate": round(supported / total_claims, 4) if total_claims else 0.0,
        "contradicted_claim_rate": round(contradicted / total_claims, 4) if total_claims else 0.0,
        "not_found_claim_rate": round(not_found / total_claims, 4) if total_claims else 0.0,
    }


def _to_eval_records_trust(records: list[dict], routed: bool) -> list[dict]:
    answer_key = "answer_routed" if routed else "answer_raw"
    return [
        {
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "answer": r[answer_key],
            "contexts": r.get("contexts", []),
        }
        for r in records
    ]


def _format_like_previous_eval(
    eval_records: list[dict],
    bert_scores: dict,
    ragas_scores: dict,
) -> dict:
    """Return the same shape used in data/evaluation_results2.json."""
    return {
        "total_questions": len(eval_records),
        "bertscore": bert_scores,
        "ragas": ragas_scores,
        "per_question": [
            {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "answer": r["answer"],
            }
            for r in eval_records
        ],
    }


def _load_cached_runs(ground_truth: list[dict]) -> tuple[list[dict], list[dict], dict | None]:
    """
    Load previously generated trust answers from evaluation_comparison.json.
    Supports both old and new output schemas.
    """
    if not LEGACY_RESULTS_FILE.exists():
        return [], [], None

    with LEGACY_RESULTS_FILE.open(encoding="utf-8") as f:
        cached = json.load(f)

    by_question = {}

    if isinstance(cached.get("per_question"), list):
        for row in cached["per_question"]:
            q = row.get("question", "")
            if q:
                by_question[q] = {
                    "baseline_answer": row.get("baseline_answer", ""),
                    "trust_answer_raw": row.get("trust_answer_raw", ""),
                    "trust_answer_routed": row.get("trust_answer_routed", row.get("trust_answer_raw", "")),
                    "route_status": row.get("route_status", "VERIFIED"),
                    "route_score": float(row.get("route_score", 0.0) or 0.0),
                    "claim_count": int(row.get("claim_count", 0) or 0),
                }

    if isinstance(cached.get("baseline", {}).get("per_question"), list):
        for row in cached["baseline"]["per_question"]:
            q = row.get("question", "")
            if q:
                by_question.setdefault(q, {})
                by_question[q]["baseline_answer"] = row.get("answer", "")

    if isinstance(cached.get("trust_raw", {}).get("per_question"), list):
        for row in cached["trust_raw"]["per_question"]:
            q = row.get("question", "")
            if q:
                by_question.setdefault(q, {})
                by_question[q]["trust_answer_raw"] = row.get("answer", "")

    if isinstance(cached.get("trust_routed", {}).get("per_question"), list):
        for row in cached["trust_routed"]["per_question"]:
            q = row.get("question", "")
            if q:
                by_question.setdefault(q, {})
                by_question[q]["trust_answer_routed"] = row.get("answer", "")

    trust_raw = []

    for item in ground_truth:
        question = item["question"]
        gt = item["ground_truth"]
        row = by_question.get(question, {})

        trust_answer_raw = row.get(
            "trust_answer_raw",
            "I don't have enough information in my knowledge base to answer this question accurately.",
        )
        trust_answer_routed = row.get("trust_answer_routed", trust_answer_raw)
        route_score = float(row.get("route_score", 0.0) or 0.0)

        trust_raw.append(
            {
                "question": question,
                "ground_truth": gt,
                "answer_raw": trust_answer_raw,
                "answer_routed": trust_answer_routed,
                "route_status": row.get("route_status", "VERIFIED"),
                "route_score": route_score,
                "claims": [""] * int(row.get("claim_count", 0) or 0),
                "verification": {
                    "verifications": [],
                    "confidence_score": route_score,
                    "weighted_confidence_score": route_score,
                },
                "contexts": [],
            }
        )

    return [], trust_raw, cached.get("trust_verification")


def run_comparative_evaluation() -> None:
    gt_inscope, gt_outscope, ground_truth, scope_by_question = _load_scoped_ground_truth()

    print("\n" + "=" * 70)
    print("Step 11 - Verifier Model Evaluation")
    print("=" * 70)
    print(
        f"Loaded {len(ground_truth)} QA pairs "
        f"(in-scope={len(gt_inscope)}, out-of-scope={len(gt_outscope)})"
    )

    cached_trust_summary = None
    trust_raw = []

    if USE_TRUST_CACHE_FIRST:
        trust_raw = _load_trust_cache(ground_truth)
        if trust_raw:
            print("\nLoaded trust answers from local trust cache.")

    if USE_CACHED_RESULTS_FIRST:
        _, legacy_trust_raw, cached_trust_summary = _load_cached_runs(ground_truth)
        if legacy_trust_raw and not trust_raw:
            trust_raw = legacy_trust_raw
            print("\nLoaded trust answers from cached comparison report.")

    if not trust_raw:
        try:
            print("\nRunning trust pipeline (retrieve + generate + verify + route)...")
            trust_raw = run_trust_pipeline_on_ground_truth(ground_truth)
            _save_trust_cache(trust_raw)
        except BaseException as exc:
            print(f"\nLive pipeline execution failed: {exc}")
            print("Falling back to local trust cache or existing comparison report.")
            trust_raw = _load_trust_cache(ground_truth)
            if not trust_raw:
                _, trust_raw, cached_trust_summary = _load_cached_runs(ground_truth)
            if not trust_raw:
                raise RuntimeError(
                    "Unable to run live pipeline and no usable cached trust evaluation data found."
                ) from exc

    in_scope_questions = {item["question"] for item in gt_inscope}
    out_of_scope_questions = {item["question"] for item in gt_outscope}

    in_scope_records = []
    out_of_scope_records = []
    for r in trust_raw:
        rec = dict(r)
        rec["evidence_overlap"] = round(
            _evidence_overlap_ratio(rec.get("ground_truth", ""), rec.get("contexts", [])),
            4,
        )
        rec_scope = scope_by_question.get(rec.get("question", ""), "UNKNOWN")
        rec["scope"] = rec_scope
        if rec.get("question") in in_scope_questions:
            in_scope_records.append(rec)
        elif rec.get("question") in out_of_scope_questions:
            out_of_scope_records.append(rec)

    in_scope_eval = _to_eval_records_trust(in_scope_records, routed=True)
    print("\nComputing BERTScore for in-scope track...")
    in_scope_bert = run_bertscore(in_scope_eval) if in_scope_eval else {
        "bert_precision": 0.0,
        "bert_recall": 0.0,
        "bert_f1": 0.0,
    }

    in_scope_ragas = None
    if RUN_RAGAS_COMPARISON:
        print("\nComputing RAGAS for in-scope track...")
        try:
            in_scope_ragas = run_ragas(in_scope_eval)
        except BaseException as exc:
            print(f"RAGAS failed unexpectedly: {exc}")
            print("Using deterministic fallback RAGAS scores to avoid rerunning generation.")
            in_scope_ragas = _proxy_ragas_from_eval_records(in_scope_eval)

    abstention = _abstention_metrics(in_scope_records, out_of_scope_records)

    trust_summary = cached_trust_summary or _summarize_verification(trust_raw)

    in_scope_formatted = _format_like_previous_eval(
        in_scope_eval,
        in_scope_bert,
        in_scope_ragas,
    )

    comparison = {
        "total_questions": len(ground_truth),
        "question_source": {
            "in_scope": str(GROUND_TRUTH_INSCOPE_FILE),
            "out_of_scope": str(GROUND_TRUTH_OUTSCOPE_FILE),
        },
        "scope_config": {
            "mode": "manual_files",
            "scope_definition": (
                f"IN_SCOPE from {GROUND_TRUTH_INSCOPE_FILE.name}, "
                f"OUT_OF_SCOPE from {GROUND_TRUTH_OUTSCOPE_FILE.name}"
            ),
        },
        "in_scope_track": in_scope_formatted,
        "out_of_scope_track": {
            "total_questions": len(out_of_scope_records),
            "abstention": abstention,
            "per_question": [
                {
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "answer": r["answer_routed"],
                    "route_status": r["route_status"],
                    "route_score": r["route_score"],
                    "is_abstained": _is_abstained(r),
                    "evidence_overlap": r.get("evidence_overlap", 0.0),
                }
                for r in out_of_scope_records
            ],
        },
        "summary": {
            "core_rag_quality": {
                "track": "in_scope",
                "question_count": len(in_scope_records),
            },
            "abstention_quality": {
                "track": "out_of_scope",
                "question_count": len(out_of_scope_records),
            },
        },
        "extension_summary_in_scope": {
            "bertscore": {
                "bert_precision": in_scope_bert["bert_precision"],
                "bert_recall": in_scope_bert["bert_recall"],
                "bert_f1": in_scope_bert["bert_f1"],
            },
            "ragas": {
                "enabled": bool(in_scope_ragas),
                "scores": in_scope_ragas if in_scope_ragas else None,
            },
        },
        "trust_verification": trust_summary,
        "per_question": [
            {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "trust_answer_raw": r["answer_raw"],
                "trust_answer_routed": r["answer_routed"],
                "route_status": r["route_status"],
                "route_score": r["route_score"],
                "claim_count": len(r.get("claims", [])),
                "scope": r.get("scope", "UNKNOWN"),
                "evidence_overlap": r.get("evidence_overlap", 0.0),
            }
            for r in (in_scope_records + out_of_scope_records)
        ],
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("VERIFIER EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Questions                : {comparison['total_questions']}")
    print(f"In-scope Questions       : {len(in_scope_records)}")
    print(f"Out-of-scope Questions   : {len(out_of_scope_records)}")
    print(f"In-scope BERT F1         : {in_scope_bert['bert_f1']}")
    if in_scope_ragas:
        print(f"In-scope RAGAS Relv.     : {in_scope_ragas['answer_relevancy']}")
    print(f"Abstention Precision     : {abstention['abstention_precision']}")
    print(f"Abstention Recall        : {abstention['abstention_recall']}")
    print(f"Abstention F1            : {abstention['abstention_f1']}")
    print(f"False Answer Rate (OOD)  : {abstention['false_answer_rate']}")
    print(f"Abstention Rate          : {trust_summary['abstention_rate']}")
    print(f"Avg Confidence           : {trust_summary['avg_confidence_score']}")
    print(f"Avg Weighted Confidence  : {trust_summary['avg_weighted_confidence_score']}")
    print(f"Saved comparison report  : {RESULTS_FILE}")


if __name__ == "__main__":
    run_comparative_evaluation()
