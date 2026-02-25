"""
CSV Ingestor — Financial_Information.csv

Responsibilities:
1. Load and validate the CSV
2. Generate pre-computed aggregate summaries (stored as extra ChromaDB docs)
3. Batch-insert TaxRecord nodes and relationships into Neo4j
4. Add individual rows + aggregates to ChromaDB as text chunks
"""

import os
import uuid
import logging
from typing import Generator

import pandas as pd

logger = logging.getLogger(__name__)

# ── Column aliases ──────────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "Taxpayer Type", "Tax Year", "Transaction Date", "Income Source",
    "Deduction Type", "State", "Income", "Deductions",
    "Taxable Income", "Tax Rate", "Tax Owed",
]


def load_csv(filepath: str) -> pd.DataFrame:
    """Load and basic-validate the financial CSV."""
    df = pd.read_csv(filepath)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.dropna(subset=["Taxpayer Type", "Tax Year", "Income", "Tax Owed"])
    df["Tax Year"] = df["Tax Year"].astype(int)
    df["_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def row_to_text(row: pd.Series) -> str:
    """Convert a CSV row to a human-readable text chunk for ChromaDB."""
    return (
        f"{row['Taxpayer Type']} taxpayer filed in {row['Tax Year']} "
        f"(transaction: {row['Transaction Date']}) in {row['State']}. "
        f"Income source: {row['Income Source']}. "
        f"Gross income: ${row['Income']:,.2f}. "
        f"Deduction type: {row['Deduction Type']}, amount: ${row['Deductions']:,.2f}. "
        f"Taxable income: ${row['Taxable Income']:,.2f}. "
        f"Tax rate: {row['Tax Rate']*100:.2f}%. "
        f"Tax owed: ${row['Tax Owed']:,.2f}."
    )


def row_to_neo4j_params(row: pd.Series) -> dict:
    """Convert a CSV row to Neo4j CREATE parameters."""
    return {
        "id": row["_id"],
        "taxpayer_type": row["Taxpayer Type"],
        "tax_year": int(row["Tax Year"]),
        "transaction_date": str(row["Transaction Date"]),
        "income_source": row["Income Source"],
        "deduction_type": row["Deduction Type"],
        "state": row["State"],
        "income": float(row["Income"]),
        "deductions": float(row["Deductions"]),
        "taxable_income": float(row["Taxable Income"]),
        "tax_rate": float(row["Tax Rate"]),
        "tax_owed": float(row["Tax Owed"]),
    }


def generate_aggregate_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Pre-compute aggregate summaries and return them as text chunks.
    These are stored in ChromaDB alongside row-level chunks so the
    LLM can answer aggregate questions without needing to scan all rows.
    """
    chunks = []

    # ── By taxpayer type ────────────────────────────────────────────────────
    for tp_type, grp in df.groupby("Taxpayer Type"):
        summary = (
            f"Summary for {tp_type} taxpayers across all years and states: "
            f"Total records: {len(grp)}. "
            f"Average income: ${grp['Income'].mean():,.2f}. "
            f"Average deductions: ${grp['Deductions'].mean():,.2f}. "
            f"Average taxable income: ${grp['Taxable Income'].mean():,.2f}. "
            f"Average tax rate: {grp['Tax Rate'].mean()*100:.2f}%. "
            f"Average tax owed: ${grp['Tax Owed'].mean():,.2f}. "
            f"Total tax owed: ${grp['Tax Owed'].sum():,.2f}."
        )
        chunks.append({
            "id": f"agg_type_{tp_type.replace(' ', '_')}",
            "text": summary,
            "metadata": {"source": "csv_aggregate", "group": "taxpayer_type", "value": tp_type},
        })

    # ── By state ────────────────────────────────────────────────────────────
    for state, grp in df.groupby("State"):
        summary = (
            f"Summary for the state of {state}: "
            f"Total records: {len(grp)}. "
            f"Average income: ${grp['Income'].mean():,.2f}. "
            f"Average tax owed: ${grp['Tax Owed'].mean():,.2f}. "
            f"Top income source: {grp['Income Source'].mode().iloc[0]}. "
            f"Most common taxpayer type: {grp['Taxpayer Type'].mode().iloc[0]}."
        )
        chunks.append({
            "id": f"agg_state_{state}",
            "text": summary,
            "metadata": {"source": "csv_aggregate", "group": "state", "value": state},
        })

    # ── By year ─────────────────────────────────────────────────────────────
    for year, grp in df.groupby("Tax Year"):
        summary = (
            f"Summary for tax year {year}: "
            f"Total records: {len(grp)}. "
            f"Average income: ${grp['Income'].mean():,.2f}. "
            f"Average tax rate: {grp['Tax Rate'].mean()*100:.2f}%. "
            f"Average tax owed: ${grp['Tax Owed'].mean():,.2f}. "
            f"Total tax collected: ${grp['Tax Owed'].sum():,.2f}."
        )
        chunks.append({
            "id": f"agg_year_{year}",
            "text": summary,
            "metadata": {"source": "csv_aggregate", "group": "tax_year", "value": str(year)},
        })

    # ── By taxpayer type × year ─────────────────────────────────────────────
    for (tp_type, year), grp in df.groupby(["Taxpayer Type", "Tax Year"]):
        summary = (
            f"{tp_type} taxpayer summary for {year}: "
            f"Records: {len(grp)}. "
            f"Avg income: ${grp['Income'].mean():,.2f}. "
            f"Avg tax rate: {grp['Tax Rate'].mean()*100:.2f}%. "
            f"Avg tax owed: ${grp['Tax Owed'].mean():,.2f}."
        )
        chunks.append({
            "id": f"agg_type_year_{tp_type.replace(' ', '_')}_{year}",
            "text": summary,
            "metadata": {
                "source": "csv_aggregate",
                "group": "taxpayer_type_year",
                "taxpayer_type": tp_type,
                "year": str(year),
            },
        })

    # ── By income source ────────────────────────────────────────────────────
    for source, grp in df.groupby("Income Source"):
        summary = (
            f"Income source '{source}': "
            f"Total records: {len(grp)}. "
            f"Average income: ${grp['Income'].mean():,.2f}. "
            f"Average tax owed: ${grp['Tax Owed'].mean():,.2f}. "
            f"Average tax rate: {grp['Tax Rate'].mean()*100:.2f}%."
        )
        chunks.append({
            "id": f"agg_source_{source.replace(' ', '_')}",
            "text": summary,
            "metadata": {"source": "csv_aggregate", "group": "income_source", "value": source},
        })

    logger.info(f"Generated {len(chunks)} aggregate chunks from CSV")
    return chunks


def iter_batches(df: pd.DataFrame, batch_size: int = 100) -> Generator:
    """Yield dataframe rows in batches."""
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]
