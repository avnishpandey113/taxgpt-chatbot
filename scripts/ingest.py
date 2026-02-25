#!/usr/bin/env python3
"""
Ingestion Pipeline — orchestrates CSV, PDF, and PPT ingestion.

Usage:
    python scripts/ingest.py                      # ingest ./data/
    python scripts/ingest.py --data-dir /path     # custom data dir
    python scripts/ingest.py --reset              # wipe & re-ingest
    python scripts/ingest.py --skip-graph         # ChromaDB only
    python scripts/ingest.py --skip-vector        # Neo4j only
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Make sure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.csv_ingestor import (
    load_csv, row_to_text, row_to_neo4j_params,
    generate_aggregate_chunks, iter_batches,
)
from src.ingestion.pdf_ingestor import extract_all_pdfs
from src.ingestion.ppt_ingestor import extract_all_pptx
from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_store import GraphStore
from src.graph.schema import SCHEMA_QUERIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TaxGPT ingestion pipeline")
    parser.add_argument("--data-dir", default="./data", help="Directory containing data files")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB before ingesting")
    parser.add_argument("--skip-graph", action="store_true", help="Skip Neo4j ingestion")
    parser.add_argument("--skip-vector", action="store_true", help="Skip ChromaDB ingestion")
    return parser.parse_args()


def ingest_csv(
    filepath: str,
    vector_store: VectorStore,
    graph_store: GraphStore,
    skip_vector: bool,
    skip_graph: bool,
):
    logger.info(f"Loading CSV: {filepath}")
    df = load_csv(filepath)

    # ── ChromaDB: row-level chunks ──────────────────────────────────────────
    if not skip_vector:
        logger.info("Generating row-level text chunks for ChromaDB...")
        row_chunks = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV rows → ChromaDB"):
            row_chunks.append({
                "id": f"csv_row_{row['_id']}",
                "text": row_to_text(row),
                "metadata": {
                    "source": "Financial_Information.csv",
                    "source_type": "csv",
                    "taxpayer_type": row["Taxpayer Type"],
                    "tax_year": str(row["Tax Year"]),
                    "state": row["State"],
                },
            })

        logger.info("Adding row chunks to ChromaDB...")
        vector_store.add_chunks(row_chunks)

        logger.info("Generating aggregate summary chunks...")
        agg_chunks = generate_aggregate_chunks(df)
        vector_store.add_chunks(agg_chunks)

    # ── Neo4j: graph nodes + relationships ─────────────────────────────────
    if not skip_graph:
        logger.info("Ingesting CSV rows into Neo4j...")
        for batch_df in tqdm(list(iter_batches(df, 100)), desc="CSV batches → Neo4j"):
            params = [row_to_neo4j_params(row) for _, row in batch_df.iterrows()]
            graph_store.ingest_csv_batch(params)

    logger.info(f"CSV ingestion complete: {len(df)} rows processed.")


def ingest_pdfs(data_dir: str, vector_store: VectorStore):
    logger.info("Extracting PDF chunks...")
    chunks = extract_all_pdfs(
        data_dir,
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
    )
    if chunks:
        logger.info(f"Adding {len(chunks)} PDF chunks to ChromaDB...")
        vector_store.add_chunks(chunks)
    else:
        logger.warning("No PDF chunks extracted — check your data/ directory")


def ingest_pptx(data_dir: str, vector_store: VectorStore):
    logger.info("Extracting PPTX chunks...")
    chunks = extract_all_pptx(data_dir)
    if chunks:
        logger.info(f"Adding {len(chunks)} PPTX chunks to ChromaDB...")
        vector_store.add_chunks(chunks)
    else:
        logger.info("No .pptx files found (PDFs from PPT are handled by pdf_ingestor)")


def main():
    args = parse_args()
    data_dir = args.data_dir

    if not Path(data_dir).exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # ── Initialize stores ───────────────────────────────────────────────────
    vector_store = None
    graph_store = None

    if not args.skip_vector:
        vector_store = VectorStore(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION", "financial_data"),
        )
        if args.reset:
            logger.warning("Resetting ChromaDB collection...")
            vector_store.reset()

    if not args.skip_graph:
        graph_store = GraphStore(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password123"),
        )
        graph_store.init_schema()

    # ── CSV ─────────────────────────────────────────────────────────────────
    csv_files = list(Path(data_dir).glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
    for csv_path in csv_files:
        ingest_csv(
            str(csv_path),
            vector_store,
            graph_store,
            skip_vector=args.skip_vector,
            skip_graph=args.skip_graph,
        )

    # ── PDFs ─────────────────────────────────────────────────────────────────
    if not args.skip_vector:
        ingest_pdfs(data_dir, vector_store)
        ingest_pptx(data_dir, vector_store)

    # ── Final stats ──────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("INGESTION COMPLETE")
    if vector_store:
        logger.info(f"  ChromaDB docs: {vector_store.count()}")
    if graph_store:
        logger.info(f"  Neo4j TaxRecord nodes: {graph_store.node_count()}")
        graph_store.close()
    logger.info("=" * 50)
    logger.info("Run: uvicorn src.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
