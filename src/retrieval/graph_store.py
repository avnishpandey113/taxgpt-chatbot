"""
Graph Store — Neo4j interface for structured financial queries.

Provides:
1. Schema initialization
2. Batch node/relationship creation
3. Query routing: detects what entities are in a question and runs
   the appropriate Cypher query automatically
"""

import re
import logging
from typing import Optional

from neo4j import GraphDatabase
from src.graph.schema import SCHEMA_QUERIES, CREATE_TAX_RECORD, GRAPH_QUERIES

logger = logging.getLogger(__name__)

# Known entity values for extraction from query text
TAXPAYER_TYPES = {"Individual", "Corporation", "Partnership", "Trust", "Non-Profit"}
STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}
INCOME_SOURCES = {"Salary", "Investment", "Business Income", "Capital Gains", "Royalties", "Rental"}
DEDUCTION_TYPES = {
    "Mortgage Interest", "Charitable Contributions", "Business Expenses",
    "Education Expenses", "Medical Expenses",
}


class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Neo4j connected: {uri}")

    def close(self):
        self.driver.close()

    def init_schema(self):
        """Create constraints and indexes."""
        with self.driver.session() as session:
            for query in SCHEMA_QUERIES:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Schema query warning (may already exist): {e}")
        logger.info("Neo4j schema initialized.")

    def ingest_csv_batch(self, rows: list[dict]) -> None:
        """Batch-insert TaxRecord nodes and relationships."""
        with self.driver.session() as session:
            session.execute_write(self._create_records_batch, rows)

    @staticmethod
    def _create_records_batch(tx, rows: list[dict]):
        for row in rows:
            tx.run(CREATE_TAX_RECORD, **row)

    def query_structured(self, question: str) -> Optional[str]:
        """
        Parse the question for financial entities and run the best Cypher query.
        Returns a formatted text summary, or None if no structured query applies.
        """
        q = question.lower()

        # Extract entities from the question
        found_type = next((t for t in TAXPAYER_TYPES if t.lower() in q), None)
        found_state = next((s for s in STATES if re.search(rf'\b{s}\b', question)), None)
        found_year = re.search(r'\b(201[0-9]|202[0-9])\b', question)
        found_source = next((s for s in INCOME_SOURCES if s.lower() in q), None)
        found_deduction = next((d for d in DEDUCTION_TYPES if d.lower() in q), None)
        year = int(found_year.group()) if found_year else None

        with self.driver.session() as session:

            # ── Type + State combo ──────────────────────────────────────────
            if found_type and found_state:
                result = session.run(
                    GRAPH_QUERIES["type_state_summary"],
                    taxpayer_type=found_type, state=found_state
                ).data()
                if result:
                    return self._format_result(result, f"{found_type} in {found_state}")

            # ── By taxpayer type ────────────────────────────────────────────
            if found_type and not found_state and not year:
                result = session.run(
                    GRAPH_QUERIES["avg_tax_by_type"],
                    taxpayer_type=found_type
                ).data()
                if result:
                    return self._format_result(result, f"{found_type} taxpayers")

            # ── By state ────────────────────────────────────────────────────
            if found_state and not found_type:
                result = session.run(
                    GRAPH_QUERIES["avg_tax_by_state"],
                    state=found_state
                ).data()
                if result:
                    return self._format_result(result, f"State: {found_state}")

            # ── By year ─────────────────────────────────────────────────────
            if year and not found_type and not found_state:
                result = session.run(
                    GRAPH_QUERIES["records_by_year"],
                    year=year
                ).data()
                if result:
                    return self._format_result(result, f"Tax Year {year}")

            # ── Income source ────────────────────────────────────────────────
            if found_source or any(kw in q for kw in ["income source", "income type"]):
                result = session.run(GRAPH_QUERIES["top_income_sources"]).data()
                if result:
                    return self._format_result(result, "Income sources")

            # ── Deduction analysis ───────────────────────────────────────────
            if found_deduction or any(kw in q for kw in ["deduction", "deduct"]):
                result = session.run(GRAPH_QUERIES["deduction_impact"]).data()
                if result:
                    return self._format_result(result, "Deduction types")

            # ── General summary fallback ─────────────────────────────────────
            if any(kw in q for kw in [
                "average", "total", "summary", "overall", "most", "highest",
                "lowest", "compare", "breakdown"
            ]):
                result = session.run(GRAPH_QUERIES["general_summary"]).data()
                if result:
                    return self._format_result(result, "Overall financial summary")

        return None

    @staticmethod
    def _format_result(records: list[dict], label: str) -> str:
        """Convert Neo4j records into a human-readable text block."""
        if not records:
            return ""

        lines = [f"[Graph Database — {label}]"]
        for rec in records[:15]:  # cap at 15 rows to keep context manageable
            parts = []
            for k, v in rec.items():
                if v is None:
                    continue
                if isinstance(v, float):
                    if "rate" in k.lower():
                        parts.append(f"{k}: {v*100:.2f}%")
                    else:
                        parts.append(f"{k}: ${v:,.2f}")
                else:
                    parts.append(f"{k}: {v}")
            lines.append("  • " + ", ".join(parts))

        return "\n".join(lines)

    def node_count(self) -> int:
        with self.driver.session() as session:
            result = session.run("MATCH (n:TaxRecord) RETURN count(n) AS cnt")
            return result.single()["cnt"]
