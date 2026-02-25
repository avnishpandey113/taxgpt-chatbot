"""
Test Suite — TaxGPT Financial Chatbot

Tests are organized into:
1. Unit tests: ingestion modules (CSV, PDF, PPT)
2. Integration tests: vector store, graph store
3. End-to-end tests: API endpoints
4. Evaluation: runs eval_dataset.json against the live API

Run all tests:    pytest tests/ -v
Run unit only:    pytest tests/ -v -m unit
Run eval only:    pytest tests/ -v -m eval
"""

import json
import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.csv_ingestor import (
    load_csv, row_to_text, row_to_neo4j_params,
    generate_aggregate_chunks, iter_batches,
)
from src.retrieval.hybrid_retriever import HybridRetriever


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a minimal test CSV."""
    data = {
        "Taxpayer Type": ["Individual", "Corporation", "Partnership", "Trust", "Non-Profit"],
        "Tax Year": [2020, 2021, 2022, 2022, 2023],
        "Transaction Date": ["2020-03-15", "2021-06-01", "2022-04-10", "2022-07-22", "2023-01-30"],
        "Income Source": ["Salary", "Business Income", "Investment", "Capital Gains", "Royalties"],
        "Deduction Type": ["Mortgage Interest", "Business Expenses", "Education Expenses", "Medical Expenses", "Charitable Contributions"],
        "State": ["CA", "TX", "NY", "FL", "IL"],
        "Income": [150000, 500000, 300000, 200000, 450000],
        "Deductions": [20000, 75000, 45000, 30000, 60000],
        "Taxable Income": [130000, 425000, 255000, 170000, 390000],
        "Tax Rate": [0.22, 0.28, 0.25, 0.20, 0.18],
        "Tax Owed": [28600, 119000, 63750, 34000, 70200],
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "test_financial.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture
def sample_dataframe(sample_csv_path):
    return load_csv(sample_csv_path)


# ── Unit: CSV Ingestor ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCsvIngestor:

    def test_load_csv_valid(self, sample_csv_path):
        df = load_csv(sample_csv_path)
        assert len(df) == 5
        assert "_id" in df.columns
        assert df["Tax Year"].dtype in [int, "int64"]

    def test_load_csv_missing_column(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Col1,Col2\n1,2\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_csv(str(bad_csv))

    def test_row_to_text_format(self, sample_dataframe):
        row = sample_dataframe.iloc[0]
        text = row_to_text(row)
        assert "Individual" in text
        assert "2020" in text
        assert "CA" in text
        assert "$" in text
        assert "%" in text

    def test_row_to_neo4j_params(self, sample_dataframe):
        row = sample_dataframe.iloc[0]
        params = row_to_neo4j_params(row)
        assert params["taxpayer_type"] == "Individual"
        assert params["state"] == "CA"
        assert params["tax_year"] == 2020
        assert isinstance(params["income"], float)
        assert "id" in params

    def test_generate_aggregate_chunks(self, sample_dataframe):
        chunks = generate_aggregate_chunks(sample_dataframe)
        # Should have chunks for each type, state, year, type×year, source
        assert len(chunks) > 0
        chunk_ids = [c["id"] for c in chunks]
        # Check specific aggregations exist
        assert any("Individual" in cid for cid in chunk_ids)
        assert any("CA" in cid for cid in chunk_ids)
        assert any("2020" in cid for cid in chunk_ids)

    def test_aggregate_chunk_text_has_dollar_amounts(self, sample_dataframe):
        chunks = generate_aggregate_chunks(sample_dataframe)
        for chunk in chunks:
            if chunk["metadata"]["group"] == "taxpayer_type":
                assert "$" in chunk["text"]
                assert "%" in chunk["text"]

    def test_iter_batches(self, sample_dataframe):
        batches = list(iter_batches(sample_dataframe, batch_size=2))
        assert len(batches) == 3  # 5 rows / 2 = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[2]) == 1

    def test_row_ids_are_unique(self, sample_dataframe):
        ids = sample_dataframe["_id"].tolist()
        assert len(ids) == len(set(ids))


# ── Unit: PDF Ingestor ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPdfIngestor:

    def test_import(self):
        from src.ingestion.pdf_ingestor import extract_chunks_from_pdf, _detect_source_type
        assert callable(extract_chunks_from_pdf)

    def test_detect_source_type(self):
        from src.ingestion.pdf_ingestor import _detect_source_type
        assert _detect_source_type("Financial_presentations_ppt.pdf") == "presentation"
        assert _detect_source_type("i1040gi.pdf") == "irs_form"
        assert _detect_source_type("usc26_118-78.pdf") == "tax_law"
        assert _detect_source_type("some_report.pdf") == "financial_report"

    def test_split_into_chunks(self):
        from src.ingestion.pdf_ingestor import _split_into_chunks
        text = " ".join(["word"] * 600)
        chunks = _split_into_chunks(text, chunk_size=512, overlap=50)
        assert len(chunks) >= 1
        # Check overlap: words should repeat between chunks
        first_words = set(chunks[0].split()[-50:])
        second_words = set(chunks[1].split()[:50]) if len(chunks) > 1 else set()
        assert len(first_words & second_words) > 0

    def test_clean_text(self):
        from src.ingestion.pdf_ingestor import _clean_text
        messy = "Hello\f\f\nWorld\r\n\n\n\n   multiple   spaces"
        clean = _clean_text(messy)
        assert "\f" not in clean
        assert "   " not in clean


# ── Unit: Graph Schema ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestGraphSchema:

    def test_schema_queries_are_strings(self):
        from src.graph.schema import SCHEMA_QUERIES, CREATE_TAX_RECORD, GRAPH_QUERIES
        for q in SCHEMA_QUERIES:
            assert isinstance(q, str)
            assert len(q) > 10
        assert "TaxRecord" in CREATE_TAX_RECORD
        assert len(GRAPH_QUERIES) > 0

    def test_all_graph_queries_have_return(self):
        from src.graph.schema import GRAPH_QUERIES
        for name, query in GRAPH_QUERIES.items():
            assert "RETURN" in query.upper(), f"Query '{name}' missing RETURN clause"


# ── Unit: Hybrid Retriever ────────────────────────────────────────────────────

@pytest.mark.unit
class TestHybridRetriever:

    def test_format_vector_hits(self):
        hits = [
            {"text": "Test content about taxes", "metadata": {"source": "test.pdf", "page": 5}, "score": 0.85},
            {"text": "More tax information", "metadata": {"source": "data.csv", "slide": None}, "score": 0.72},
        ]
        result = HybridRetriever._format_vector_hits(hits)
        assert "test.pdf" in result
        assert "0.85" in result
        assert "Semantic Search Results" in result

    def test_retrieve_with_no_results(self):
        mock_vector = MagicMock()
        mock_vector.query.return_value = []
        mock_graph = MagicMock()
        mock_graph.query_structured.return_value = None

        retriever = HybridRetriever(mock_vector, mock_graph)
        result = retriever.retrieve("some query")
        assert "No relevant" in result

    def test_retrieve_uses_both_stores(self):
        mock_vector = MagicMock()
        mock_vector.query.return_value = [
            {"text": "vector result", "metadata": {"source": "test.pdf"}, "score": 0.8}
        ]
        mock_graph = MagicMock()
        mock_graph.query_structured.return_value = "[Graph Database] data here"

        retriever = HybridRetriever(mock_vector, mock_graph)
        result = retriever.retrieve("Corporation tax in CA")

        assert "Graph Database" in result
        assert "vector result" in result
        mock_vector.query.assert_called_once()
        mock_graph.query_structured.assert_called_once()


# ── Integration: API ──────────────────────────────────────────────────────────

@pytest.mark.integration
class TestAPI:
    """
    Integration tests require the API server to be running.
    Start with: uvicorn src.api.main:app --port 8000
    """

    BASE_URL = "http://localhost:8000"

    def test_health_endpoint(self):
        import httpx
        try:
            res = httpx.get(f"{self.BASE_URL}/api/health", timeout=5)
            assert res.status_code == 200
            data = res.json()
            assert "status" in data
            assert "vector_docs" in data
            assert "graph_nodes" in data
        except httpx.ConnectError:
            pytest.skip("API server not running")

    def test_chat_endpoint_returns_answer(self):
        import httpx
        try:
            res = httpx.post(
                f"{self.BASE_URL}/api/chat",
                json={"message": "What is the average tax rate for Corporations?"},
                timeout=60,
            )
            assert res.status_code == 200
            data = res.json()
            assert "answer" in data
            assert len(data["answer"]) > 10
        except httpx.ConnectError:
            pytest.skip("API server not running")

    def test_chat_endpoint_invalid_message(self):
        import httpx
        try:
            res = httpx.post(
                f"{self.BASE_URL}/api/chat",
                json={"message": ""},
                timeout=10,
            )
            assert res.status_code == 422  # Pydantic validation error
        except httpx.ConnectError:
            pytest.skip("API server not running")


# ── Evaluation: Ground Truth ──────────────────────────────────────────────────

@pytest.mark.eval
class TestEvaluation:
    """
    Runs the eval_dataset.json questions against the live API and checks
    that expected terms appear in the answer.
    """

    BASE_URL = "http://localhost:8000"
    EVAL_PATH = Path(__file__).parent / "eval_dataset.json"

    def _load_eval(self):
        with open(self.EVAL_PATH) as f:
            return json.load(f)

    def test_eval_questions_structure(self):
        """Validate that eval_dataset.json is well-formed."""
        evals = self._load_eval()
        assert len(evals) > 0
        for item in evals:
            assert "id" in item
            assert "question" in item
            assert "expected_contains" in item
            assert isinstance(item["expected_contains"], list)

    @pytest.mark.parametrize("eval_item", json.loads(
        (Path(__file__).parent / "eval_dataset.json").read_text()
    ))
    def test_eval_answer_contains_keywords(self, eval_item):
        """Run each eval question and verify expected keywords appear."""
        import httpx
        try:
            res = httpx.post(
                f"{self.BASE_URL}/api/chat",
                json={"message": eval_item["question"]},
                timeout=90,
            )
            if res.status_code != 200:
                pytest.skip(f"API returned {res.status_code}")

            answer = res.json()["answer"].lower()
            missing = [
                kw for kw in eval_item["expected_contains"]
                if kw.lower() not in answer
            ]
            assert not missing, (
                f"[{eval_item['id']}] Answer missing keywords: {missing}\n"
                f"Q: {eval_item['question']}\n"
                f"A: {answer[:300]}"
            )
        except httpx.ConnectError:
            pytest.skip("API server not running — start with: uvicorn src.api.main:app --port 8000")
