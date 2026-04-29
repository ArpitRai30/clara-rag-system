"""
QuIM-RAG Phase 1 - Step 2: Chunking
=====================================
Reads:  data/raw_pages.jsonl        (output of step1_crawler.py)
Writes: data/chunks.jsonl           (one chunk per line)

Paper spec:
  - Chunk size  : 1000 tokens  (using TikToken, GPT-3.5 tokenizer)
  - Overlap     : 200 characters (character-level, not token-level)
  - Each chunk carries its source URL as metadata

Each saved record:
  {
    "chunk_id"   : "chunk_0001",
    "text"       : "...",
    "char_count" : N,
    "token_count": N,
    "source_url" : "https://..."
  }
"""

import json
import logging
from pathlib import Path

import tiktoken

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE     = Path("data/raw_pages.jsonl")
OUTPUT_FILE    = Path("data/chunks.jsonl")
LOG_FILE       = Path("logs/chunking.log")

CHUNK_SIZE_TOKENS = 1000   # paper: 1000 tokens per chunk
OVERLAP_CHARS     = 200    # paper: 200 character overlap between chunks
ENCODING_NAME     = "cl100k_base"  # GPT-3.5/GPT-4 tokenizer

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


# ── Chunker ───────────────────────────────────────────────────────────────────
class TikTokenChunker:
    """
    Splits text into chunks of at most `max_tokens` tokens.
    Between consecutive chunks there is a character-level overlap
    of `overlap_chars` characters (paper spec).
    """

    def __init__(self, max_tokens: int = CHUNK_SIZE_TOKENS,
                 overlap_chars: int = OVERLAP_CHARS,
                 encoding_name: str = ENCODING_NAME):
        self.enc           = tiktoken.get_encoding(encoding_name)
        self.max_tokens    = max_tokens
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> list[str]:
        """Return a list of text chunks from `text`."""
        tokens     = self.enc.encode(text)
        chunks     = []
        start_tok  = 0

        while start_tok < len(tokens):
            end_tok = min(start_tok + self.max_tokens, len(tokens))

            # Decode this window back to text
            chunk_text = self.enc.decode(tokens[start_tok:end_tok])
            chunks.append(chunk_text)

            if end_tok == len(tokens):
                break   # last chunk — done

            # Next chunk starts so that the last `overlap_chars`
            # characters of this chunk are re-included.
            # Find the token boundary closest to that overlap point.
            overlap_text  = chunk_text[-self.overlap_chars:]
            overlap_toks  = len(self.enc.encode(overlap_text))
            start_tok     = end_tok - overlap_toks

        return [c.strip() for c in chunks if c.strip()]


# ── Main ──────────────────────────────────────────────────────────────────────
def run_chunking():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run step1_crawler.py first."
        )

    chunker   = TikTokenChunker()
    enc       = tiktoken.get_encoding(ENCODING_NAME)
    chunk_idx = 0
    page_idx  = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("")   # clear

    with (INPUT_FILE.open(encoding="utf-8") as in_f,
          OUTPUT_FILE.open("a", encoding="utf-8") as out_f):

        for line in in_f:
            line = line.strip()
            if not line:
                continue

            page       = json.loads(line)
            url        = page["url"]
            text       = page["text"]
            page_idx  += 1

            page_chunks = chunker.chunk(text)

            for chunk_text in page_chunks:
                token_count = len(enc.encode(chunk_text))
                record = {
                    "chunk_id"   : f"chunk_{chunk_idx:05d}",
                    "text"       : chunk_text,
                    "char_count" : len(chunk_text),
                    "token_count": token_count,
                    "source_url" : url,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_idx += 1

            log.info("Page %d → %d chunks | url: %s",
                     page_idx, len(page_chunks), url)

    log.info("Chunking complete. Pages: %d | Chunks: %d", page_idx, chunk_idx)
    return chunk_idx


if __name__ == "__main__":
    total_chunks = run_chunking()
    print(f"\nDone. {total_chunks} chunks saved to {OUTPUT_FILE}")
    print("Next → run step3_question_generation.py")
