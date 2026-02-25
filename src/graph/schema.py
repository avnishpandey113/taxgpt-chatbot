"""
Graph schema definitions for Neo4j.
Sets up constraints, indexes, and node labels for the financial knowledge graph.
"""

SCHEMA_QUERIES = [
    # ── Uniqueness constraints ──────────────────────────────────────────────
    "CREATE CONSTRAINT taxpayer_type_name IF NOT EXISTS FOR (t:TaxpayerType) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT state_code IF NOT EXISTS FOR (s:State) REQUIRE s.code IS UNIQUE",
    "CREATE CONSTRAINT tax_year_value IF NOT EXISTS FOR (y:TaxYear) REQUIRE y.value IS UNIQUE",
    "CREATE CONSTRAINT income_source_name IF NOT EXISTS FOR (i:IncomeSource) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT deduction_type_name IF NOT EXISTS FOR (d:DeductionType) REQUIRE d.name IS UNIQUE",

    # ── Indexes for fast lookup ─────────────────────────────────────────────
    "CREATE INDEX tax_record_id IF NOT EXISTS FOR (r:TaxRecord) ON (r.id)",
    "CREATE INDEX tax_record_year IF NOT EXISTS FOR (r:TaxRecord) ON (r.tax_year)",
    "CREATE INDEX tax_record_income IF NOT EXISTS FOR (r:TaxRecord) ON (r.income)",
    "CREATE INDEX tax_record_tax_owed IF NOT EXISTS FOR (r:TaxRecord) ON (r.tax_owed)",
]


CREATE_TAX_RECORD = """
MERGE (tp:TaxpayerType {name: $taxpayer_type})
MERGE (st:State {code: $state})
MERGE (yr:TaxYear {value: $tax_year})
MERGE (is:IncomeSource {name: $income_source})
MERGE (dt:DeductionType {name: $deduction_type})

CREATE (r:TaxRecord {
    id: $id,
    tax_year: $tax_year,
    transaction_date: $transaction_date,
    income: $income,
    deductions: $deductions,
    taxable_income: $taxable_income,
    tax_rate: $tax_rate,
    tax_owed: $tax_owed
})

CREATE (r)-[:FILED_BY]->(tp)
CREATE (r)-[:IN_STATE]->(st)
CREATE (r)-[:IN_YEAR]->(yr)
CREATE (r)-[:HAS_SOURCE]->(is)
CREATE (r)-[:HAS_DEDUCTION]->(dt)
"""

# Aggregation queries exposed via the retriever
GRAPH_QUERIES = {
    "avg_tax_by_type": """
        MATCH (r:TaxRecord)-[:FILED_BY]->(tp:TaxpayerType {name: $taxpayer_type})
        RETURN tp.name AS taxpayer_type,
               avg(r.tax_owed) AS avg_tax_owed,
               avg(r.tax_rate) AS avg_tax_rate,
               count(r) AS record_count
    """,

    "avg_tax_by_state": """
        MATCH (r:TaxRecord)-[:IN_STATE]->(s:State {code: $state})
        OPTIONAL MATCH (r)-[:FILED_BY]->(tp:TaxpayerType)
        RETURN s.code AS state,
               tp.name AS taxpayer_type,
               avg(r.tax_owed) AS avg_tax_owed,
               count(r) AS record_count
        ORDER BY avg_tax_owed DESC
    """,

    "records_by_year": """
        MATCH (r:TaxRecord)-[:IN_YEAR]->(y:TaxYear {value: $year})
        OPTIONAL MATCH (r)-[:FILED_BY]->(tp:TaxpayerType)
        OPTIONAL MATCH (r)-[:IN_STATE]->(s:State)
        RETURN y.value AS year,
               tp.name AS taxpayer_type,
               s.code AS state,
               avg(r.income) AS avg_income,
               avg(r.tax_owed) AS avg_tax_owed,
               count(r) AS record_count
        ORDER BY avg_tax_owed DESC
    """,

    "top_income_sources": """
        MATCH (r:TaxRecord)-[:HAS_SOURCE]->(i:IncomeSource)
        OPTIONAL MATCH (r)-[:FILED_BY]->(tp:TaxpayerType)
        RETURN i.name AS income_source,
               tp.name AS taxpayer_type,
               sum(r.income) AS total_income,
               avg(r.tax_rate) AS avg_tax_rate,
               count(r) AS record_count
        ORDER BY total_income DESC
    """,

    "deduction_impact": """
        MATCH (r:TaxRecord)-[:HAS_DEDUCTION]->(d:DeductionType)
        RETURN d.name AS deduction_type,
               avg(r.deductions) AS avg_deduction_amount,
               avg(r.taxable_income / r.income) AS avg_effective_reduction_ratio,
               count(r) AS record_count
        ORDER BY avg_deduction_amount DESC
    """,

    "type_state_summary": """
        MATCH (r:TaxRecord)-[:FILED_BY]->(tp:TaxpayerType {name: $taxpayer_type})
        MATCH (r)-[:IN_STATE]->(s:State {code: $state})
        RETURN tp.name AS taxpayer_type,
               s.code AS state,
               avg(r.income) AS avg_income,
               avg(r.taxable_income) AS avg_taxable_income,
               avg(r.tax_owed) AS avg_tax_owed,
               avg(r.tax_rate) AS avg_tax_rate,
               count(r) AS record_count
    """,

    "general_summary": """
        MATCH (r:TaxRecord)
        OPTIONAL MATCH (r)-[:FILED_BY]->(tp:TaxpayerType)
        OPTIONAL MATCH (r)-[:IN_STATE]->(s:State)
        OPTIONAL MATCH (r)-[:IN_YEAR]->(y:TaxYear)
        RETURN tp.name AS taxpayer_type,
               s.code AS state,
               y.value AS year,
               count(r) AS count,
               avg(r.income) AS avg_income,
               avg(r.tax_owed) AS avg_tax_owed,
               avg(r.tax_rate) AS avg_tax_rate
        ORDER BY count DESC
        LIMIT 20
    """,
}
