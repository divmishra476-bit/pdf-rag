# PDF RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about any uploaded PDF, grounded strictly in the document's content. Built from scratch — no LangChain/LlamaIndex — to understand every layer of the pipeline: extraction, chunking, embeddings, vector search, and grounded generation.

**Live demo:** https://pdf-rag-bfivgarhe9ab4h89sljcuj.streamlit.app

## What it does

Upload any PDF, ask questions about it in plain English, and get answers grounded in the actual document — not the model's general knowledge. If the answer isn't in the document, the system says so instead of guessing.

## Demo

<img width="700" alt="Chat history showing two grounded Q&A exchanges" src="https://github.com/user-attachments/assets/4e512df6-82a1-4b83-ab4c-1d728d8d475c" />

*Example: asked about "cloud model family" (not explicitly in the document) — the system correctly clarified what it does know (AWS/GCP training, enterprise availability) instead of guessing. The follow-up question was answered using retrieved context, with both exchanges preserved in the chat history.*




## Why this exists

Most beginner RAG demos work fine on the happy path and fall apart the moment you ask something the document doesn't cover. This project was built and tested specifically to check for that failure mode — see **Evaluation** below.

## Architecture

```
PDF upload
   │
   ▼
Text extraction (PyMuPDF)
   │
   ▼
Chunking (1000 chars, 200 char overlap)
   │
   ▼
Embeddings (sentence-transformers, all-MiniLM-L6-v2 — runs locally, free)
   │
   ▼
Vector storage (ChromaDB, persistent)
   │
   ▼
User question → embedded → top-3 similar chunks retrieved
   │
   ▼
Chunks + question → prompt → Groq (Llama 3.3 70B) → grounded answer
```

## Tech stack

- **Python**
- **PyMuPDF (fitz)** — PDF text extraction (switched from `pypdf` after hitting decompression errors on certain PDFs)
- **sentence-transformers** — local, free embeddings (no API cost)
- **ChromaDB** — persistent vector storage
- **Groq (Llama 3.3 70B)** — free LLM inference for answer generation
- **Streamlit** — UI

## Evaluation

Tested with 9 questions spanning four categories, specifically to probe for weaknesses rather than just confirm the happy path:

| Category | Result |
|---|---|
| Directly answerable questions | Accurate, detailed answers |
| Deliberately unrelated questions (e.g. "capital of France") | Correctly refused to answer instead of hallucinating |
| Multi-chunk reasoning (comparing two models in the doc) | Correctly synthesized information spread across multiple chunks |
| Ambiguous/adjacent questions (e.g. "is it available on AWS?") | Gave a nuanced answer — distinguished "trained using AWS infrastructure" from "available as an AWS product," rather than overclaiming either way |

This is the result I consider most important: the system's honesty under ambiguity, not just its accuracy on easy questions.

**Known limitation found during live testing (n_results=3):**
Enumeration and comparison-style questions (e.g., "what are the three models in the family," "compare X vs Y") occasionally failed — the system said "I don't know" even though the document contained the answer. Root cause: the 3 nearest-neighbor chunks sometimes clustered around only one side of a comparison, so the model genuinely didn't have both halves of the answer in context, and correctly refused rather than guessing. Single-topic and section-summary questions were unaffected (6/8 answerable questions passed).

**Edge cases handled:**
- Corrupted/incomplete PDF downloads (caught via cross-checking two different PDF libraries)
- Empty retrieval results (returns a clear fallback message instead of crashing or sending an empty prompt to the LLM)
- Re-ingesting a new document cleanly replaces the old one (no stale data or ID collisions)

## Project structure

```
pdf-rag/
├── ingest.py      # PDF → chunks → embeddings → ChromaDB
├── query.py       # retrieval + grounded answer generation
├── app.py         # Streamlit UI
├── requirements.txt
└── .gitignore
```

## Running it locally

```bash
git clone https://github.com/divmishra476-bit<img width="1920" height="1080" alt="Screenshot (368)" src="https://github.com/user-attachments/assets/acf92c05-7f81-4faf-b0c3-672ea6d7f731" />
/pdf-rag.git
cd pdf-rag
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file with:
```
GROQ_API_KEY=your_key_here
```

Then run:
```bash
streamlit run app.py
```

