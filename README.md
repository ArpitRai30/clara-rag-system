# 🕵️‍♀️ CLARA: Claim-Level Atomic Reasoning and Abstention

> **For Trustworthy Retrieval Augmented Generation**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![LLaMA-3](https://img.shields.io/badge/LLaMA--3-8B-orange?style=for-the-badge&logo=meta)
![ChromaDB](https://img.shields.io/badge/Vector_DB-Chroma-1B2F3A?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research_Implementation-success?style=for-the-badge)

Retrieval Augmented Generation (RAG) significantly improves factual answer generation in Large Language Models (LLMs), yet these systems remain susceptible to hallucination. **CLARA** introduces a novel evaluation and verification pipeline structured to evaluate text at the granular, atomic claim level.

---

## 🚀 Overview

Large Language Models often struggle with domain-specific knowledge, prompting the use of Retrieval Augmented Generation (RAG). However, standard RAG lacks mechanisms to verify if retrieved evidence actually supports the generated output. 

**CLARA** provides a two-phase framework designed for **claim-level factual verification** and **self-regulation**:
1. **Phase I (QuIM-RAG):** Uses an inverted question-matching pipeline to retrieve chunks accurately.
2. **Phase II (Trust Layer):** Decomposes generated answers into atomic claims, verifies each one mathematically against retrieved chunks, and routes the answer based on a confidence threshold (safe abstention).

---

## ✨ Key Features

* 🧩 **Atomic Claim Decomposition:** Answers are broken down into granular, verifiable factual claims rather than evaluating entire paragraphs at once.
* ⚖️ **Per-Claim Verification:** Every claim is checked against source chunks and tagged as `SUPPORTED`, `CONTRADICTED`, or `NOT_FOUND`.
* 🛡️ **Confidence-Gated Router:** The system calculates a weighted confidence score. If the score falls below a set threshold, the system **safely abstains** instead of hallucinating.
* 🔍 **Inverted Question Matching:** Retrieval operates in the *question semantic space* rather than raw text gaps, drastically improving contextual relevance.

---

## 🏗️ System Architecture

### 🔹 Phase I: Inverted Question Matching RAG (QuIM-RAG)
* **Web Crawling (`step1_crawler.py`):** Traverses institutional portals to build a clean HTML-extracted corpus.
* **Text Chunking (`step2_chunker.py`):** Transforms content into continuous 1,000-token blocks utilizing a TikToken `cl100k_base` tokenizer.
* **Hypothetical Question Generation (`step3_question_generation.py`):** Utilizing LLaMA3-8B (via Ollama), maps textual chunks into multiple hypothetical natural language questions.
* **Dual-Collection Indexing (`step4_build_index.py`):** BAAI/bge-large-en-v1.5 embeddings for both questions and raw chunks inside a ChromaDB database.
* **Retrieval & Pipeline (`step5` & `step6`):** Encodes user prompts and matches them against the *Question Collection* to fetch grounded context, serving a preliminary LLM answer.

### 🔹 Phase II: CLARA Post-Generation Verification
* **Decomposition (`step8_claim_decomposition.py`):** LLM deterministically splits the Phase I answer into isolated, verifiable bullet points.
* **Verification (`step9_claim_verification.py`):** The core trust judge. Computes similarity rankings between claims and text context, labelling validity context.
* **Output Routing (`step10_confidence_router.py`):** Calculates Base Confidence and Weighted Confidence score. Uses `τ = 0.75` to decide between Verified Output or logging an Audit Trail Abstention.

---

## 📊 Performance & Evaluation

Extensively evaluated on custom ground datasets partitioning both in-scope queries and deliberate out-of-scope institutional anomalies.

| Evaluation Metric | Score | Description |
| :--- | :--- | :--- |
| **BERTScore F1** | `0.8675` | High semantic fidelity to source text |
| **Abstention Precision** | `94.12%` | Effectively catches and abstains unanswerable queries |
| **Supported Claim Rate** | `94.37%` | Out of 515 verified atomic claims |
| **Contradicted Claims** | `0.39%` | Minimal active hallucination leakage |

> *"Our system achieved an abstention precision of 94.12%, correctly withholding an answer for out-of-scope questions, indicating strong robustness against hallucinations."*

---

## ⚙️ Installation & Usage

### Prerequisites
Instantiate a Python environment (Python 3.10+) and ensure you have a local instance of [Ollama](https://ollama.com/) serving the `llama3:8b` model.

```bash
pip install -r requirements.txt
```

### Execution Pipeline
Follow the steps in numerical order to replicate the framework:

```bash
# Phase I: Data Prep & Indexing
python step1_crawler.py               # 1. Scrape Pages
python step2_chunker.py               # 2. Extract Token Chunks
python step3_question_generation.py   # 3. Formulate Q&A pairs
python step4_build_index.py           # 4. Generate ChromaDB Vector Space
python step5_retrieval.py             # 5. Baseline Q-to-Q Validation
python step6_pipeline.py              # 6. Constrained Base Generation

# Phase II: Verification & Abstention
python step8_claim_decomposition.py   # 7. Atomic Breakdowns
python step9_claim_verification.py    # 8. Fact-Checking Evidence
python step10_confidence_router.py    # 9. End-to-end framework w/ Trust Router
python step7_evaluation.py            # 10. Metric Computation
```

---

