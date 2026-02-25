"""
Hybrid Retriever — Merges ChromaDB semantic search + Neo4j graph queries.

Retrieval strategy:
1. Always run semantic vector search (handles unstructured PDF/PPT + CSV text)
2. Detect if question has structured financial entities → also run graph query
3. Merge results: graph data first (precise), then semantic chunks (context)
4. Deduplicate and rank by relevance score
5. Return a formatted context string for the LLM

This approach ensures:
- "What does the IRS say about..." → answered by PDF chunks (vector only)
- "Average tax for Corporations in TX?" → answered by graph + CSV chunks
- "Explain excise tax from the slides" → answered by PPT chunks (vector only)
"""

import logging
from typing import Optional

from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_store import GraphStore

logger = logging.getLogger(__name__)

# Minimum similarity score to include a vector result
SCORE_THRESHOLD = 0.25


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vector = vector_store
        self.graph = graph_store

    def retrieve(
        self,
        query: str,
        n_vector: int = 8,
        include_graph: bool = True,
    ) -> str:
        """
        Main retrieval entry point.
        Returns a formatted context string ready to be injected into the LLM prompt.
        """
        context_parts = []

        # ── 1. Graph retrieval (structured, precise) ────────────────────────
        if include_graph:
            try:
                graph_result = self.graph.query_structured(query)
                if graph_result:
                    context_parts.append(graph_result)
                    logger.info("Graph retrieval: hit")
                else:
                    logger.info("Graph retrieval: no structured match")
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")

        # ── 2. Vector retrieval (semantic, covers all sources) ───────────────
        try:
            vector_hits = self.vector.query(query, n_results=n_vector)
            filtered = [h for h in vector_hits if h["score"] >= SCORE_THRESHOLD]

            if filtered:
                vector_section = self._format_vector_hits(filtered)
                context_parts.append(vector_section)
                logger.info(f"Vector retrieval: {len(filtered)} hits above threshold")
            else:
                logger.info("Vector retrieval: no hits above threshold")
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")

        if not context_parts:
            return "No relevant financial information found in the knowledge base."

        return "\n\n".join(context_parts)

    @staticmethod
    def _format_vector_hits(hits: list[dict]) -> str:
        """Format vector search results into a labelled context block."""
        lines = ["[Semantic Search Results]"]
        for i, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            source = meta.get("source", "unknown")
            page = meta.get("page", "")
            slide = meta.get("slide", "")
            score = hit["score"]

            location = ""
            if slide:
                location = f"slide {slide}"
            elif page:
                location = f"page {page}"

            header = f"[{i}] {source}"
            if location:
                header += f" ({location})"
            header += f" — relevance: {score:.2f}"

            lines.append(header)
            lines.append(hit["text"])
            lines.append("")

        return "\n".join(lines)
