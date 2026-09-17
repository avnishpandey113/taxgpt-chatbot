# TaxGPT Financial Chatbot

**Author:** Avnish Pandey / https://www.linkedin.com/in/avnishpandey/

## Demo Video
[▶ Watch Demo](https://drive.google.com/file/d/1KWpgP5tM0Xrn1mhnxPlPmiquNl9bDb81/view?usp=drive_link)

A financial Q&A chatbot that ingests structured (CSV) and unstructured (PDF, PPT) data, performs hybrid semantic + graph-based retrieval, and answers user questions via a local LLM (Ollama).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Data Sources                        │
│   Financial_Information.csv │ PDFs │ PPT (via PDF)       │
└────────────────┬────────────────────────────────────────┘
                 │ Ingestion Pipeline
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Dual Retrieval Layer                         │
│                                                          │
│  ┌─────────────────────┐   ┌──────────────────────────┐ │
│  │  ChromaDB (Vector)  │   │  Neo4j (Graph)           │ │
│  │  Semantic search    │   │  Entity relationships    │ │
│  │  sentence-xformers  │   │  (taxpayer→state→year)  │ │
│  └──────────┬──────────┘   └───────────┬──────────────┘ │
│             └──────────┬───────────────┘                 │
│                        ▼                                 │
│              Hybrid Retrieval Merger                     │
└────────────────────────┬────────────────────────────────┘
                         │ Context
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Ollama (Local LLM)                          │
│              llama3 / mistral                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI + Chat UI                           │
│              POST /api/chat                              │
│              GET  /                                      │
└─────────────────────────────────────────────────────────┘
```

---

## Design Rationale

### Why Hybrid Retrieval (Vector + Graph)?

**Vector search alone** (ChromaDB) is great at semantic similarity — finding chunks that *sound like* what you're asking. But it struggles with structured relational queries like:
> "What was the average tax owed by Corporations in Texas in 2022?"

**Graph search alone** (Neo4j) excels at traversing entity relationships but can't handle fuzzy/semantic queries over unstructured text (PDFs, PPTs).

**Hybrid approach** gives us the best of both:
- ChromaDB handles: "What does the IRS say about charitable contribution deductions?"
- Neo4j handles: "Show me all Partnerships with income over $500k in CA"
- Both contribute context to the LLM for complex blended queries

### Why FastAPI?
- Typed, async, auto-generates OpenAPI docs at `/docs`
- Familiar structure to Express.js developers
- Clean separation of concerns

### Why Ollama?
- Fully local — no API keys, no costs, no data leaving the machine
- Evaluators can verify it works offline
- Supports llama3 (best open-source reasoning for finance)

### Chunking Strategy
- **CSV:** Each row becomes a document + aggregated summaries are pre-computed and stored as additional chunks (totals by taxpayer type, state, year)
- **PDFs:** Split by page, then by paragraph (≥100 chars). Overlapping chunks of 512 tokens with 50-token overlap preserve context across page boundaries
- **PPT (PDF):** Slide-by-slide chunking; figure captions extracted separately

### Graph Schema (Neo4j)
```
(TaxRecord) -[:FILED_BY]-> (TaxpayerType)
(TaxRecord) -[:IN_STATE]-> (State)
(TaxRecord) -[:IN_YEAR]-> (TaxYear)
(TaxRecord) -[:HAS_SOURCE]-> (IncomeSource)
(TaxRecord) -[:HAS_DEDUCTION]-> (DeductionType)
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://python.org) |
| Node.js | ≥ 18 | [nodejs.org](https://nodejs.org) |
| Ollama | latest | [ollama.ai](https://ollama.ai) |
| Neo4j | ≥ 5.x | [neo4j.com/download](https://neo4j.com/download) |

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone <your-private-repo-url>
cd taxgpt-chatbot
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start Neo4j
```bash
# Option A: Docker (recommended)
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5

# Option B: Neo4j Desktop
# Download from neo4j.com/download, create a local DB with password "password123"
```

### 4. Install and start Ollama

**Download and install Ollama for your OS:**
- **Windows/Mac:** https://ollama.com/download — download and run the installer
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

After installation, Ollama **auto-starts in the background** — you do NOT need to run `ollama serve` manually.

> **Windows note:** If you see `Error: listen tcp 127.0.0.1:11434: bind: Only one usage...` when trying to start it, Ollama is **already running** — proceed to the next step.

**Pull the model:**
```bash
ollama pull llama3
# Lighter alternative if you have less than 8GB RAM:
# ollama pull mistral
# Then set OLLAMA_MODEL=mistral in your .env file
```

### 5. Configure environment
```bash
cp .env.example .env
# Edit .env if your Neo4j password or Ollama model differs
```

### 6. Place your data files
```bash
# Copy the provided datasets into the data/ folder:
cp /path/to/Financial_Information.csv data/
cp /path/to/financial_report.pdf data/
cp /path/to/financial_presentation.pdf data/
# (PPT should be converted to PDF, or placed as .pptx — both supported)
```

### 7. Run ingestion pipeline
```bash
python scripts/ingest.py
# This will:
# - Parse all files in data/
# - Create ChromaDB collection in ./chroma_db/
# - Populate Neo4j with entities and relationships
# Expected time: ~2-5 minutes for 5000 CSV rows + PDFs
```

### 8. Start the API server
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 9. Open the chat UI
```
http://localhost:8000
```

Or use the API directly:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What was the average tax rate for Corporations in 2022?"}'
```

---

## Example Queries

| Query | Expected source |
|-------|----------------|
| "What is the average taxable income for Individuals in California?" | CSV → Neo4j |
| "Explain the impact of an excise tax with inelastic demand" | PPT → ChromaDB |
| "What does the IRS 1040 say about deductions for mortgage interest?" | PDF → ChromaDB |
| "Which taxpayer type paid the most taxes in 2023?" | CSV → Neo4j + ChromaDB |
| "Compare tax rates between Partnerships and Corporations" | CSV → Neo4j |

---

## Running Tests
```bash
pytest tests/ -v
# Generates test report in tests/results/
```

---

## Project Structure
```
taxgpt-chatbot/
├── README.md
├── requirements.txt
├── .env.example
├── data/                          # Put your datasets here
├── chroma_db/                     # Auto-created by ingest.py
├── scripts/
│   └── ingest.py                  # One-shot ingestion pipeline
├── src/
│   ├── ingestion/
│   │   ├── csv_ingestor.py        # CSV parser + aggregator
│   │   ├── pdf_ingestor.py        # PDF chunker
│   │   └── ppt_ingestor.py        # PPT/PDF slide parser
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB interface
│   │   ├── graph_store.py         # Neo4j queries
│   │   └── hybrid_retriever.py    # Merges both retrievals
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   ├── models.py              # Pydantic schemas
│   │   └── llm.py                 # Ollama client
│   └── graph/
│       └── schema.py              # Neo4j Cypher schema setup
├── frontend/
│   └── index.html                 # Chat UI
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_api.py
    └── eval_dataset.json          # Ground-truth Q&A pairs
```
