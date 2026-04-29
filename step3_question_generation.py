"""
QuIM-RAG Phase 1 - Step 3: Question Generation
===============================================
Reads:  data/chunks.jsonl           (output of step2_chunker.py)
Writes: data/corpus.jsonl           (final custom corpus)

Paper spec:
  - Model  : GPT-3.5-turbo-instruct
  - Prompt : generate ALL questions covering key info in chunk,
             no redundancy, each question unique and relevant
  - Output : list of questions per chunk

Each saved record:
  {
    "chunk_id"   : "chunk_00001",
    "text"       : "...",
    "source_url" : "https://...",
    "questions"  : ["Q1?", "Q2?", ...]
  }

SETUP:
  export OPENAI_API_KEY="sk-..."
  pip install openai tiktoken
"""

import json
import logging
import os
import time
from pathlib import Path

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE    = Path("data/chunks.jsonl")
OUTPUT_FILE   = Path("data/corpus.jsonl")
FAILED_FILE   = Path("data/failed_chunks.jsonl")   # retry list
LOG_FILE      = Path("logs/question_gen.log")

MODEL         = "gpt-3.5-turbo-instruct"
MAX_TOKENS    = 1024       # per API call
TEMPERATURE   = 0.3        # low = more focused, less random
DELAY_SECONDS = 1.0        # between API calls (rate limit safety)
MAX_RETRIES   = 3

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


# ── Prompt (paper Figure 4 — upper section) ───────────────────────────────────
def build_prompt(chunk_text: str) -> str:
    """
    Replicates the custom prompt described in Figure 4 of the paper.
    Instructs the model to:
      1. Generate questions covering ALL key information in the chunk
      2. Avoid redundancy — each question must be unique
      3. Be contextually relevant to the chunk
      4. Return one question per line (for easy parsing)
    """
    return (
        "You are an expert at creating comprehensive question sets from text passages.\n\n"
        "Read the following text chunk carefully and generate a set of questions that:\n"
        "1. Cover ALL key information and concepts present in the chunk\n"
        "2. Are unique — do not generate redundant or overlapping questions\n"
        "3. Are specific and directly relevant to the content of the chunk\n"
        "4. Range from factual to conceptual to ensure full coverage\n\n"
        "Return ONLY the questions, one per line, with no numbering, "
        "no bullet points, and no extra commentary.\n\n"
        f"TEXT CHUNK:\n{chunk_text}\n\n"
        "QUESTIONS:"
    )


# ── API call with retry ───────────────────────────────────────────────────────
def generate_questions(client: OpenAI, chunk_text: str) -> list[str]:
    """
    Call GPT-3.5-turbo-instruct and parse the response into a list of questions.
    Returns empty list on failure after MAX_RETRIES.
    """
    prompt = build_prompt(chunk_text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.completions.create(
                model       = MODEL,
                prompt      = prompt,
                max_tokens  = MAX_TOKENS,
                temperature = TEMPERATURE,
            )
            raw_text = response.choices[0].text.strip()

            # Parse: split on newlines, clean each line
            questions = []
            for line in raw_text.split("\n"):
                line = line.strip()
                # Remove leading numbering like "1." "1)" "- " if present
                line = line.lstrip("0123456789.-) ").strip()
                if len(line) > 10 and line.endswith("?"):
                    questions.append(line)

            if questions:
                return questions

            log.warning("Empty question list on attempt %d for chunk", attempt)

        except Exception as exc:
            log.error("API error on attempt %d: %s", attempt, exc)
            time.sleep(2 ** attempt)   # exponential backoff

    return []


# ── Load already-processed chunk IDs (for resume) ────────────────────────────
def load_done_ids(output_path: Path) -> set[str]:
    done = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["chunk_id"])
                except Exception:
                    pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────
def run_question_generation():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Run:  export OPENAI_API_KEY='sk-...'"
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run step2_chunker.py first."
        )

    client   = OpenAI(api_key=api_key)
    done_ids = load_done_ids(OUTPUT_FILE)

    if done_ids:
        log.info("Resuming — %d chunks already processed.", len(done_ids))

    total_chunks     = 0
    processed        = 0
    failed_chunks    = []

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with (INPUT_FILE.open(encoding="utf-8") as in_f,
          OUTPUT_FILE.open("a", encoding="utf-8") as out_f):

        for line in in_f:
            line = line.strip()
            if not line:
                continue

            chunk        = json.loads(line)
            chunk_id     = chunk["chunk_id"]
            total_chunks += 1

            # Skip already processed
            if chunk_id in done_ids:
                continue

            questions = generate_questions(client, chunk["text"])

            if not questions:
                log.error("FAILED to generate questions for %s", chunk_id)
                failed_chunks.append(chunk_id)
                continue

            record = {
                "chunk_id"  : chunk_id,
                "text"      : chunk["text"],
                "source_url": chunk["source_url"],
                "questions" : questions,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            log.info(
                "[%d/%d] %s → %d questions",
                processed, total_chunks, chunk_id, len(questions)
            )

            time.sleep(DELAY_SECONDS)

    # Save failed chunk IDs for manual retry
    if failed_chunks:
        FAILED_FILE.write_text(
            "\n".join(failed_chunks), encoding="utf-8"
        )
        log.warning("%d chunks failed — see %s", len(failed_chunks), FAILED_FILE)

    log.info(
        "Question generation complete. Processed: %d | Failed: %d | Total: %d",
        processed, len(failed_chunks), total_chunks
    )
    return processed


if __name__ == "__main__":
    total = run_question_generation()
    print(f"\nDone. {total} corpus records saved to {OUTPUT_FILE}")
    print("Next → run Phase 2: step4_build_index.py")
