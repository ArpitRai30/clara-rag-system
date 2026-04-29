"""
QuIM-RAG Phase 1 - Step 1: Web Crawler
=======================================
Crawls NDSU Career Advising and NDSU Catalog websites.
Uses BeautifulSoup + requests (paper used Scrapy+BS4).
Saves: data/raw_pages.jsonl  — one JSON object per page.

Each saved record:
  { "url": "...", "text": "...", "char_count": N }
"""

import json
import time
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
SEED_URLS = [
    "https://career-advising.ndsu.edu",
    "https://catalog.ndsu.edu",
]

ALLOWED_DOMAINS = {
    "career-advising.ndsu.edu",
    "catalog.ndsu.edu",
}

OUTPUT_FILE   = Path("data/raw_pages.jsonl")
LOG_FILE      = Path("logs/crawl.log")
MIN_CHARS     = 250          # paper: filter pages < 250 characters
DELAY_SECONDS = 1.0          # polite delay between requests
MAX_PAGES     = 5000         # safety cap — remove if you want full crawl
REQUEST_TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; QuIMRAG-Research-Crawler/1.0; "
        "+https://github.com/your-repo)"
    )
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_allowed(url: str) -> bool:
    """Return True if URL belongs to one of the allowed domains."""
    try:
        domain = urlparse(url).netloc.lstrip("www.")
        return domain in ALLOWED_DOMAINS
    except Exception:
        return False


def clean_text(soup: BeautifulSoup) -> str:
    """
    Remove header, footer, nav, script, style tags — same cleanup
    described in the paper — then return plain text.
    """
    for tag in soup(["header", "footer", "nav", "script",
                     "style", "noscript", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_404_page(soup: BeautifulSoup) -> bool:
    """Detect pages that are effectively '404 not found'."""
    title = soup.title.string if soup.title else ""
    return "404" in (title or "").lower() or "not found" in (title or "").lower()


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Return all absolute HTTP(S) links found on the page."""
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        absolute = urljoin(base_url, href)
        parsed   = urlparse(absolute)
        # Keep only http/https, drop anchors and query strings
        clean = parsed._replace(fragment="", query="").geturl()
        if parsed.scheme in ("http", "https"):
            links.append(clean)
    return links


# ── Main crawl ────────────────────────────────────────────────────────────────
def crawl():
    visited: set[str] = set()
    queue:   deque     = deque(SEED_URLS)
    saved   = 0
    skipped = 0

    OUTPUT_FILE.write_text("")   # clear / create file

    with OUTPUT_FILE.open("a", encoding="utf-8") as out_f:
        while queue and saved < MAX_PAGES:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if not is_allowed(url):
                continue

            try:
                resp = requests.get(url, headers=HEADERS,
                                    timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException as exc:
                log.warning("SKIP %s — %s", url, exc)
                skipped += 1
                continue

            # Only process HTML pages
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # ── Paper filters ──────────────────────────────────────────────
            if is_404_page(soup):
                log.info("404-title page, skipping: %s", url)
                skipped += 1
                continue

            text = clean_text(soup)

            if len(text) < MIN_CHARS:
                log.info("Too short (%d chars), skipping: %s", len(text), url)
                skipped += 1
                continue
            # ──────────────────────────────────────────────────────────────

            record = {"url": url, "text": text, "char_count": len(text)}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            log.info("[%d saved] %s (%d chars)", saved, url, len(text))

            # Enqueue new links
            for link in extract_links(soup, url):
                if link not in visited and is_allowed(link):
                    queue.append(link)

            time.sleep(DELAY_SECONDS)

    log.info("Crawl complete. Saved: %d | Skipped: %d | Visited: %d",
             saved, skipped, len(visited))
    return saved


if __name__ == "__main__":
    total = crawl()
    print(f"\nDone. {total} pages saved to {OUTPUT_FILE}")
