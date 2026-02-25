"""
Vector Store — ChromaDB with sentence-transformers embeddings.

Uses the all-MiniLM-L6-v2 model (22M params, 80MB, runs fully local).
All embeddings are computed locally — no API calls needed.
"""

import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# Local embedding model — fast, small, good for financial text
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
        )
        logger.info(
            f"ChromaDB initialized: collection='{collection_name}', "
            f"docs={self.collection.count()}"
        )

    def add_chunks(self, chunks: list[dict], batch_size: int = 500) -> None:
        """
        Add text chunks to ChromaDB.
        Each chunk: {id: str, text: str, metadata: dict}
        Skips chunks whose IDs already exist (idempotent ingestion).
        """
        if not chunks:
            return

        # Filter already-existing IDs to support re-runs
        existing_ids = set(self.collection.get(
            ids=[c["id"] for c in chunks]
        )["ids"])
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already exist in ChromaDB — skipping.")
            return

        # Batch upserts to avoid memory spikes
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c.get("metadata", {}) for c in batch],
            )
            logger.info(f"Added batch {i // batch_size + 1}: {len(batch)} chunks")

        logger.info(f"Total docs in collection: {self.collection.count()}")

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        source_filter: Optional[str] = None,
        source_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search: returns top-n most relevant chunks.
        Optional filters narrow results to a specific file or source type.

        Returns list of: {id, text, metadata, distance}
        """
        where = {}
        if source_filter:
            where["source"] = {"$eq": source_filter}
        if source_type_filter:
            where["source_type"] = {"$eq": source_type_filter}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),  # convert distance to similarity score
            })

        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (for re-ingestion)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("ChromaDB collection reset.")
