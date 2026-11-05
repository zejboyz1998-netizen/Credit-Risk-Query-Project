"""
Northbridge Bank – Credit Risk Query Engine
Streamlit application converted from the Jupyter Notebook.

Required files in the same directory:
  - credit_risk_portfolio.db

Environment variables (or Streamlit secrets):
  - OPENAI_API_KEY
  - OPENAI_API_BASE   (optional, only needed for a custom endpoint)
"""

import os
import re
import json
import sqlite3
import time

import numpy as np
import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Northbridge Bank – Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Credentials – env vars or Streamlit secrets
# ---------------------------------------------------------------------------
def _resolve_credentials():
    """Read OPENAI_API_KEY / OPENAI_API_BASE from env or st.secrets."""
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    try:
        if not api_key:
            api_key  = st.secrets.get("OPENAI_API_KEY", "")
        if not api_base:
            api_base = st.secrets.get("OPENAI_API_BASE", "")
    except Exception:
        pass
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
    return bool(api_key)

credentials_ok = _resolve_credentials()

# ---------------------------------------------------------------------------
# LLM – cached so it is instantiated only once per server run
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_llms():
    llm           = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm, evaluator_llm

# ---------------------------------------------------------------------------
# Database connection – read-only, cached
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_connection():
    db_path = "credit_risk_portfolio.db"
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
    )
    return conn

# ---------------------------------------------------------------------------
# Database schema (preserved exactly from notebook)
# ---------------------------------------------------------------------------
DATABASE_SCHEMA = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# ---------------------------------------------------------------------------
# Verified SQL queries (preserved exactly from notebook)
# ---------------------------------------------------------------------------
sql_1 = """
SELECT s.sector_name,
ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_exposure_mn
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_mn DESC
"""

sql_2 = """
SELECT l.loan_category,
       ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
       COUNT(*) AS loan_count
FROM loan_master l
GROUP BY l.loan_category
ORDER BY total_outstanding_mn DESC
"""

sql_3 = """
SELECT p.ifrs9_stage,
       COUNT(*) AS loan_count,
       ROUND(SUM(p.ead_amount) / 1000000.0, 2) AS total_ead_mn,
       ROUND(SUM(p.ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning p
WHERE p.reporting_date = '2025-09-30'
GROUP BY p.ifrs9_stage
ORDER BY p.ifrs9_stage ASC
"""

sql_4 = """
SELECT s.sector_name,
       ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master l
    ON p.loan_account_number = l.loan_account_number
JOIN sector_master s
    ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""

sql_5 = """
SELECT l.borrower_name,
       s.sector_name,
       ROUND(l.total_outstanding / 1000000.0, 2) AS outstanding_amount_mn,
       l.asset_classification
FROM loan_master l
JOIN sector_master s
    ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10
"""

sql_6 = """
SELECT l.group_name,
       COUNT(*) AS loan_count,
       ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master l
WHERE l.group_name IS NOT NULL
GROUP BY l.group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5
"""

sql_7 = """
SELECT l.loan_account_number,
       l.borrower_name,
       s.sector_name,
       ROUND(l.total_outstanding / 1000000.0, 2) AS outstanding_amount_mn,
       l.days_past_due,
       l.asset_classification
FROM loan_master l
JOIN sector_master s
    ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC
"""

sql_8 = """
SELECT CASE
           WHEN l.days_past_due = 0 THEN '0 (Current)'
           WHEN l.days_past_due BETWEEN 1 AND 30 THEN '1-30'
           WHEN l.days_past_due BETWEEN 31 AND 60 THEN '31-60'
           WHEN l.days_past_due BETWEEN 61 AND 90 THEN '61-90'
           ELSE '90+'
       END AS dpd_bucket,
       COUNT(*) AS loan_count,
       ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master l
GROUP BY dpd_bucket
ORDER BY MIN(l.days_past_due) ASC
"""

sql_9 = """
SELECT r.borrower_id,
       r.previous_rating,
       r.internal_rating,
       r.pd_estimate
FROM borrower_rating r
WHERE r.rating_date = '2025-09-30'
  AND r.rating_direction = 'Downgraded'
ORDER BY r.pd_estimate DESC
"""

sql_10 = """
SELECT p.reporting_date,
       ROUND(SUM(p.ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning p
GROUP BY p.reporting_date
ORDER BY p.reporting_date ASC
"""

VERIFIED_QUERY_LIBRARY = {
    "VQ1":  {"description": "Sector-wise total outstanding and NPA amount breakdown across all sectors",                                    "sql": sql_1},
    "VQ2":  {"description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)",                     "sql": sql_2},
    "VQ3":  {"description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter", "sql": sql_3},
    "VQ4":  {"description": "Average provision coverage ratio by sector for the latest reporting quarter",                                  "sql": sql_4},
    "VQ5":  {"description": "Top 10 largest loan exposures by outstanding amount at the borrower level",                                   "sql": sql_5},
    "VQ6":  {"description": "Top 5 largest exposures aggregated at the business group level",                                              "sql": sql_6},
    "VQ7":  {"description": "All overdue loan accounts with their days past due and asset classification",                                  "sql": sql_7},
    "VQ8":  {"description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio",                   "sql": sql_8},
    "VQ9":  {"description": "Borrowers whose internal rating was downgraded in the latest rating cycle",                                   "sql": sql_9},
    "VQ10": {"description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time",             "sql": sql_10},
}

# ---------------------------------------------------------------------------
# Ground truth (from notebook, TC03 corrected to 'generated' / NaN)
# ---------------------------------------------------------------------------
GROUND_TRUTH = pd.DataFrame([
    {
        "Test Case":        "TC01",
        "User Query":       "Show the total outstanding and NPA amount breakdown across all sectors.",
        "Expected Route":   "verified",
        "Expected Query ID": "VQ1",
        "Expected Answer":  "Sector-wise total outstanding and NPA breakdown",
    },
    {
        "Test Case":        "TC02",
        "User Query":       "What is the distribution of loans and outstanding amounts across DPD aging buckets?",
        "Expected Route":   "verified",
        "Expected Query ID": "VQ8",
        "Expected Answer":  "Distribution across DPD aging buckets",
    },
    {
        "Test Case":        "TC03",
        "User Query":       "Provide the total outstanding and loan count for restructured versus non-restructured loans.",
        "Expected Route":   "generated",
        "Expected Query ID": np.nan,
        "Expected Answer":  "Restructured vs non-restructured summary",
    },
    {
        "Test Case":        "TC04",
        "User Query":       "What is the average interest rate charged on loans across different industry sectors?",
        "Expected Route":   "generated",
        "Expected Query ID": np.nan,
        "Expected Answer":  "Average interest rate by sector",
    },
    {
        "Test Case":        "TC05",
        "User Query":       "Show the trend of Stage 3 total exposure and ECL amounts across quarterly reporting dates.",
        "Expected Route":   "generated",
        "Expected Query ID": np.nan,
        "Expected Answer":  "Stage 3 exposure and ECL across reporting dates",
    },
])

# ---------------------------------------------------------------------------
# Pipeline tools (preserved exactly from notebook)
# ---------------------------------------------------------------------------

def classify_intent(user_question, query_library, llm):
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )
    classification_prompt = f"""
You are an intent classification system for Northbridge Bank's commercial lending credit risk portfolio.
Decide whether the user's question can be answered by one of the pre-approved verified query templates, or needs a new query.

Rules:
1. Route to "verified" if the question matches the core reporting intent of one template below, even if phrased differently.
2. If verified, set "query_id" to that template ID (VQ1 to VQ10).
3. Route to "generated" with "query_id" null only if no template matches.

Choose the "verified" route ONLY if one library query answers the WHOLE question: every metric, filter and breakdown the user asks for must already be in that query's description.
If the question needs a metric, filter, grouping or time dimension that no single query provides, even if a query is partly related, choose "generated" with query_id null.
Do not pick a query just because it shares a topic or keyword with the question. For example, a question about Stage 3 exposure AND ECL does not match a query that only reports ECL.
When unsure, choose "generated".

User Question:

{user_question}

{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""
    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context, llm):
    generation_prompt = f"""
You are an expert SQLite analyst for Northbridge Bank's commercial lending credit risk portfolio.
Write ONE read-only SQLite query that answers the user's question, using only the tables and columns in the schema below.

Rules:
1. Output ONLY the SQL query. No explanations, comments, headings or markdown.
2. The query must start with SELECT or WITH.
3. Use only SQLite-compatible syntax.
4. Use only tables and columns that appear in the schema. Do not invent columns.
5. Never use INSERT, UPDATE, DELETE, DROP, ALTER or CREATE.
6. Never select borrower names, borrower IDs or other customer-identifiable information unless the question explicitly requires it.
7. NPA means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
8. Show monetary amounts in millions: divide by 1000000.0 and ROUND to 2 decimals.
9. Join sector_master to loan_master on sector_code when sector names are needed.
10. Use clear column aliases and a sensible ORDER BY.

User Question:

{user_question}

{schema_context}

"""
    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, query_id=None):
    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None,
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH"]
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {kw}"
            return result
    if ";" in candidate_sql.rstrip(";").rstrip():
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple statements are not allowed"
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed to parse or plan: {str(e)}"
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )
    relevance_prompt = f"""
You are a SQL reviewer for Northbridge Bank's commercial lending credit risk portfolio.
Decide whether the SQL query below correctly answers the user's question.

Check that:
1. The metric the query calculates matches what the user asked for.
2. The tables, joins and aggregation make sense for the question.
3. The filters, grouping and ordering fit the question.

Answer "yes" only if the SQL would give a correct answer to the question. Answer "no" if the query answers a different question, uses the wrong metric, or is missing something essential.
Give a confidence from 0.0 to 1.0 for your verdict.

Track context:

{track_context}

{user_question}

{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result["relevance_confidence"] = relevance_json.get("confidence", 0.0)
        if relevance_json.get("verdict") == "no" or relevance_json.get("confidence", 0.0) < 0.6:
            result["failed_check"] = "llm_relevance"
            result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]["sql"]
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols   = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = f"Expected {len(expected_cols)} columns, got {len(actual_cols)}"
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity check failed: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    retry_prompt = f"""
You are an expert SQLite analyst for Northbridge Bank's commercial lending credit risk portfolio.
A SQL query you wrote failed validation. Fix it so it answers the user's question and passes validation.

Rules:
1. Output ONLY the corrected SQL query. No explanations, comments, headings or markdown.
2. The query must start with SELECT or WITH and contain a single statement.
3. Use only SQLite-compatible syntax.
4. Use only tables and columns that appear in the schema. Do not invent columns.
5. Never use INSERT, UPDATE, DELETE, DROP, ALTER or CREATE.
6. Never select borrower names, borrower IDs or other customer-identifiable information unless the question explicitly requires it.
7. NPA means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
8. Show monetary amounts in millions: divide by 1000000.0 and ROUND to 2 decimals.
9. Fix the specific problem named in the validation error below.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""
    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r"^```sql\s*|\s*```$", "", revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r"^```\s*|\s*```$", "", revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    result = {"dataframe": None, "reasonable": True, "warnings": []}
    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df
    if df.empty:
        result["warnings"].append("Query returned an empty result")
    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")
    if len(result["warnings"]) > 2:
        result["reasonable"] = False
    return result


def generate_response(user_question, dataframe, llm, route, query_id=None):
    response_prompt = f"""
You are a credit risk analytics assistant for Northbridge Bank's commercial lending portfolio.
Answer the user's question using ONLY the query result below.

Rules:
1. Use only numbers that appear in the query result. Never calculate new totals, averages or percentages, and never estimate or invent figures.
2. Copy numbers exactly as shown. Do not round them differently.
3. Check the column names for units. A column ending in "_mn" or "millions" is in millions of dollars. Report it as millions, for example "$10,795.43 million". Never convert it to thousands or billions, and never shorten it.
4. Focus on what the user asked. If the question names a specific sector, category or bucket, highlight those rows. Otherwise show the full result.
5. Be concise. Start with a direct answer, then give a short table or a few bullet points.
6. Do not mention SQL, tables, columns or the database.
7. Do not include customer-identifiable information.
8. If the result is empty, say that no matching data was found. Do not guess.

User Question:

{user_question}

{dataframe.to_string()}

"""
    return llm.invoke(response_prompt).content.strip()


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm):
    log = {
        "user_question":  user_question,
        "route":          None,
        "query_id":       None,
        "match_reason":   None,
        "candidate_sql":  None,
        "gate_result":    None,
        "retry_used":     False,
        "escalated":      False,
        "executed_sql":   None,
        "row_count":      None,
        "confidence":     None,
        "narrative":      None,
    }

    # Step 1: Intent classification
    classification     = classify_intent(user_question, query_library, llm)
    log["route"]       = classification["route"]
    log["query_id"]    = classification.get("query_id")
    log["match_reason"] = classification.get("match_reason")

    # Step 2: Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"]
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    log["candidate_sql"] = candidate_sql

    # Step 3: Validation gate
    gate             = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, log["query_id"])
    log["gate_result"] = gate

    # Step 4: Retry once on generated track if validation fails
    if not gate["passed"] and log["route"] == "generated":
        candidate_sql      = retry_generation(user_question, candidate_sql, gate["details"], schema_context, llm)
        log["candidate_sql"] = candidate_sql
        log["retry_used"]  = True
        gate               = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, None)
        log["gate_result"] = gate

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        log["escalated"] = True
        log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log["confidence"] = "ESCALATED"
        return {"log": log, "dataframe": None, **log}

    # Step 6: Execute
    log["executed_sql"] = candidate_sql
    exec_result         = execute_query(candidate_sql, db_connection)
    df                  = exec_result["dataframe"]
    log["row_count"]    = len(df)

    # Step 7: Response generation
    narrative       = generate_response(user_question, df, llm, log["route"], log["query_id"])
    log["narrative"]  = narrative
    log["confidence"] = gate.get("relevance_confidence")

    return {"log": log, "dataframe": df, **log}


# ===========================================================================
# Streamlit UI
# ===========================================================================

# Sidebar – branding
with st.sidebar:
    st.title("🏦 Northbridge Bank")
    st.caption("Credit Risk Query Engine")
    st.divider()
    st.markdown("**Pipeline stages**")
    st.markdown(
        "1. Intent classification\n"
        "2. Query construction\n"
        "3. Validation gate (5 checks)\n"
        "4. Retry (generated track)\n"
        "5. Query execution\n"
        "6. Response generation"
    )
    st.divider()
    st.caption("Model: gpt-4o-mini · Database: credit_risk_portfolio.db")

# Check prerequisites before rendering any other UI
if not credentials_ok:
    st.error(
        "**OpenAI API key not found.**\n\n"
        "Set `OPENAI_API_KEY` as an environment variable or add it to your "
        "`.streamlit/secrets.toml` file:\n\n"
        "```toml\nOPENAI_API_KEY = \"sk-...\"\nOPENAI_API_BASE = \"https://...\"\n```"
    )
    st.stop()

conn = _get_connection()
if conn is None:
    st.error(
        "**Database file not found.**\n\n"
        "Place `credit_risk_portfolio.db` in the same directory as `app.py` and restart."
    )
    st.stop()

llm, evaluator_llm = _get_llms()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_query, tab_library, tab_eval = st.tabs(
    ["🔍 Query Engine", "📚 Verified Query Library", "📊 Test Cases & Evaluation"]
)

# ===========================================================================
# Tab 1 – Query Engine
# ===========================================================================
with tab_query:
    st.header("Credit Risk Query Engine")
    st.markdown(
        "Ask any question about Northbridge Bank's commercial lending portfolio. "
        "The engine will route it to a pre-verified template or generate a fresh SQL query."
    )

    example_questions = [
        "Show the total outstanding and NPA amount breakdown across all sectors.",
        "What is the distribution of loans and outstanding amounts across DPD aging buckets?",
        "What is the average interest rate charged on loans across different industry sectors?",
        "Show the trend of Stage 3 total exposure and ECL amounts across quarterly reporting dates.",
        "Which borrowers had their internal credit rating downgraded in the latest rating cycle?",
        "What is the IFRS 9 stage breakdown of the portfolio for the latest quarter?",
        "Show the top 5 business groups by total outstanding exposure.",
    ]

    with st.expander("💡 Example questions"):
        for q in example_questions:
            if st.button(q, key=f"ex_{q[:30]}"):
                st.session_state["prefill_question"] = q

    prefill = st.session_state.pop("prefill_question", "")
    user_question = st.text_area(
        "Your question",
        value=prefill,
        height=80,
        placeholder="e.g. What is the total NPA exposure by sector?",
    )

    run_btn = st.button("▶ Run Query", type="primary", use_container_width=True)

    if run_btn and user_question.strip():
        with st.spinner("Running pipeline…"):
            t0     = time.time()
            result = run_pipeline(
                user_question.strip(),
                conn,
                VERIFIED_QUERY_LIBRARY,
                DATABASE_SCHEMA,
                llm,
                evaluator_llm,
            )
            elapsed = time.time() - t0

        # ── Pipeline status bar ──────────────────────────────────────────
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        route_label = result["route"] or "—"
        qid_label   = result["query_id"] or "generated"
        conf_val    = result["confidence"]
        conf_label  = f"{conf_val:.2f}" if isinstance(conf_val, float) else str(conf_val)
        rows_label  = str(result["row_count"]) if result["row_count"] is not None else "—"

        c1.metric("Route",        route_label.capitalize())
        c2.metric("Query ID",     qid_label)
        c3.metric("Confidence",   conf_label)
        c4.metric("Rows returned", rows_label)

        st.caption(f"Pipeline completed in {elapsed:.1f} s · retry used: {result['retry_used']}")

        # ── Escalation ───────────────────────────────────────────────────
        if result["escalated"]:
            st.error(f"⚠️ Escalated to human analyst — {result['narrative']}")

        else:
            # ── Match reason ─────────────────────────────────────────────
            if result.get("match_reason"):
                st.info(f"**Classification reason:** {result['match_reason']}")

            # ── Narrative ────────────────────────────────────────────────
            st.subheader("📝 Answer")
            st.markdown(result["narrative"])

            # ── Result table ─────────────────────────────────────────────
            if result["dataframe"] is not None and not result["dataframe"].empty:
                st.subheader("📋 Result Data")
                st.dataframe(result["dataframe"], use_container_width=True)

            # ── SQL ──────────────────────────────────────────────────────
            with st.expander("🗃️ Executed SQL"):
                st.code(result["executed_sql"], language="sql")

    elif run_btn:
        st.warning("Please enter a question before running the pipeline.")

# ===========================================================================
# Tab 2 – Verified Query Library
# ===========================================================================
with tab_library:
    st.header("Verified Query Library")
    st.markdown(
        f"The engine maintains **{len(VERIFIED_QUERY_LIBRARY)} pre-approved query templates** "
        "that have been validated for correctness and safety. "
        "When a user question matches one of these templates, the pipeline routes directly to it."
    )

    selected_qid = st.selectbox(
        "Select a template to preview",
        list(VERIFIED_QUERY_LIBRARY.keys()),
        format_func=lambda k: f"{k} – {VERIFIED_QUERY_LIBRARY[k]['description'][:70]}…",
    )

    entry = VERIFIED_QUERY_LIBRARY[selected_qid]
    st.markdown(f"**Description:** {entry['description']}")
    st.code(entry["sql"].strip(), language="sql")

    if st.button("▶ Run this template against the database", key="run_template"):
        with st.spinner("Executing…"):
            exec_res = execute_query(entry["sql"], conn)
        df = exec_res["dataframe"]
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(df)} rows returned")
        else:
            st.info("Query returned no rows.")

    st.divider()
    st.subheader("All templates at a glance")
    lib_df = pd.DataFrame(
        [{"ID": k, "Description": v["description"]} for k, v in VERIFIED_QUERY_LIBRARY.items()]
    )
    st.dataframe(lib_df, use_container_width=True, hide_index=True)

# ===========================================================================
# Tab 3 – Test Cases & Evaluation
# ===========================================================================
with tab_eval:
    st.header("Test Cases & Pipeline Evaluation")
    st.markdown(
        "Run all five test cases from the notebook and compare the pipeline output "
        "against ground truth to calculate path accuracy, query accuracy, and average confidence."
    )

    st.subheader("Ground Truth")
    st.dataframe(GROUND_TRUTH, use_container_width=True, hide_index=True)

    if st.button("▶ Run All 5 Test Cases", type="primary"):
        test_results = []
        progress = st.progress(0, text="Running TC01…")
        result_placeholders = []

        for i, (_, gt) in enumerate(GROUND_TRUTH.iterrows()):
            progress.progress((i) / 5, text=f"Running {gt['Test Case']}…")
            with st.spinner(f"Running {gt['Test Case']}: {gt['User Query'][:60]}…"):
                r = run_pipeline(
                    gt["User Query"],
                    conn,
                    VERIFIED_QUERY_LIBRARY,
                    DATABASE_SCHEMA,
                    llm,
                    evaluator_llm,
                )
            test_results.append(r)

        progress.progress(1.0, text="All test cases complete.")

        # ── Build evaluation DataFrame ────────────────────────────────────
        eval_rows = []
        for i, (_, gt) in enumerate(GROUND_TRUTH.iterrows()):
            tr = test_results[i]
            route_match = tr["route"] == gt["Expected Route"]
            qid_match   = (
                (pd.isna(gt["Expected Query ID"]) and (tr["query_id"] is None or pd.isna(tr["query_id"])))
                or tr["query_id"] == gt["Expected Query ID"]
            )
            conf = tr["confidence"]
            eval_rows.append({
                "Test Case":        gt["Test Case"],
                "Expected Route":   gt["Expected Route"],
                "Actual Route":     tr["route"],
                "Route Match":      "✅" if route_match else "❌",
                "Expected Query ID": gt["Expected Query ID"],
                "Actual Query ID":  tr["query_id"] if tr["query_id"] else "—",
                "Query ID Match":   "✅" if qid_match else "❌",
                "Confidence":       f"{conf:.2f}" if isinstance(conf, float) else str(conf),
                "Rows Returned":    tr["row_count"] if tr["row_count"] is not None else "—",
                "_route_match_bool": route_match,
                "_qid_match_bool":  qid_match,
                "_conf_float":      conf if isinstance(conf, float) else None,
                "_expected_route":  gt["Expected Route"],
            })

        eval_df = pd.DataFrame(eval_rows)

        path_accuracy     = eval_df["_route_match_bool"].mean() * 100
        verified_mask     = eval_df["_expected_route"].str.lower() == "verified"
        query_accuracy    = eval_df.loc[verified_mask, "_qid_match_bool"].mean() * 100
        avg_confidence    = eval_df["_conf_float"].dropna().mean()

        st.divider()
        st.subheader("Evaluation Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Path Accuracy",    f"{path_accuracy:.1f}%")
        m2.metric("Query Accuracy",   f"{query_accuracy:.1f}%")
        m3.metric("Avg Confidence",   f"{avg_confidence:.2f}")

        display_cols = [
            "Test Case", "Expected Route", "Actual Route", "Route Match",
            "Expected Query ID", "Actual Query ID", "Query ID Match",
            "Confidence", "Rows Returned",
        ]
        st.dataframe(eval_df[display_cols], use_container_width=True, hide_index=True)

        # ── Individual TC narratives ──────────────────────────────────────
        st.divider()
        st.subheader("Individual Test Case Outputs")
        for i, (_, gt) in enumerate(GROUND_TRUTH.iterrows()):
            tr = test_results[i]
            with st.expander(f"{gt['Test Case']} – {gt['User Query'][:70]}…"):
                st.markdown(f"**Route:** {tr['route']}  |  **Query ID:** {tr['query_id'] or 'N/A'}  |  **Confidence:** {tr['confidence']}")
                if tr["narrative"]:
                    st.markdown(tr["narrative"])
                if tr["dataframe"] is not None and not tr["dataframe"].empty:
                    st.dataframe(tr["dataframe"], use_container_width=True)
                if tr["executed_sql"]:
                    st.code(tr["executed_sql"], language="sql")
