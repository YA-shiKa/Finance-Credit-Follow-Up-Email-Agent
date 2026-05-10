"""
agents/credit_graph.py
──────────────────────
LangGraph agent orchestrator for the Finance Credit Follow-Up Email Agent.

Architecture (StateGraph — ReAct-style sequential batch):
─────────────────────────────────────────────────────────
  START
    │
    ▼
  ingest_records        ← load CSV / list, sanitise every record
    │
    ▼
  classify_invoices     ← assign stage 1-5 per record (get_stage)
    │
    ▼
  route_invoice         ← conditional edge: stage 1-4 → generate_email
                                             stage 5   → escalate_record
    │                         │
    ▼                         ▼
  generate_email        escalate_record   ← log to escalation_flags table
    │
    ▼
  send_or_dryrun        ← SMTP live or dry-run skip
    │
    ▼
  audit_log             ← persist to email_audit table (PII-masked body)
    │
    ▼
  END

The graph iterates over records sequentially inside a single run.
Each node reads from / writes to AgentState (TypedDict).

Security mitigations carried forward from email_agent.py:
  - sanitise_record()  (Mitigation 1 — Prompt Injection)
  - mask_pii()         (Mitigation 2 — Data Privacy)
  - Keys from env only (Mitigation 3 — API Key Exposure)
  - EmailOutput Pydantic validation (Mitigation 4 — Hallucination)
  - dry_run=True default (Mitigation 5 — Unauthorised Send)
  - SMTPConfig locks sender domain (Mitigation 6 — Spoofing)
"""

import logging
import os
from typing import Any, Optional

import pandas as pd
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.email_agent import (
    # DB_PATH_DEFAULT,
    TONE_MATRIX,
    SMTPConfig,
    generate_and_send,
    get_escalations,
    get_stage,
    init_db,
    log_email,
    log_escalation,
    sanitise_record,
)

logger = logging.getLogger(__name__)

DB_PATH_DEFAULT = "logs/audit.db"

# Shared Agent State
class AgentState(TypedDict):
    """
    The shared state dict threaded through every LangGraph node.

    Fields
    ──────
    records        : raw invoice records loaded from CSV/list
    staged         : records after stage classification
    current_idx    : index of the record currently being processed
    results        : list of successfully generated email result dicts
    escalations    : list of records flagged for legal (stage 5)
    errors         : list of (invoice_no, error_message) tuples
    dry_run        : True = no real emails sent
    api_key        : Groq API key (read from env if None)
    smtp_config    : SMTPConfig instance (None in dry_run mode)
    db_path        : SQLite audit DB path
    batch_limit    : max records to process in one run (0 = all)
    delay_sec      : sleep between API calls (for rate-limit safety)
    done           : True when all records have been processed
"""
    records: list[dict]
    staged: list[dict]
    current_idx: int
    results: list[dict]
    escalations: list[dict]
    errors: list[tuple[str, str]]
    dry_run: bool
    api_key: Optional[str]
    smtp_config: Optional[Any]
    db_path: str
    batch_limit: int
    delay_sec: float
    done: bool


# Node helpers
def _safe_invoice_no(record: dict) -> str:
    return str(record.get("invoice_no", "UNKNOWN"))


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — ingest_records
# ─────────────────────────────────────────────────────────────────────────────
def ingest_records(state: AgentState) -> AgentState:
    """
    Sanitise every record in state["records"] using sanitise_record().
    Drops records that fail validation and logs a warning.

    Security: this is the first line of defence against prompt injection —
    all string fields are stripped, HTML-escaped, and length-capped before
    any record touches the LLM prompt builder.
    """
    clean: list[dict] = []
    errors = list(state["errors"])

    for record in state["records"]:
        try:
            clean.append(sanitise_record(record))
        except ValueError as exc:
            inv = _safe_invoice_no(record)
            logger.warning("ingest_records: dropping %s — %s", inv, exc)
            errors.append((inv, f"Sanitisation failed: {exc}"))

    limit = state["batch_limit"]
    if limit > 0:
        clean = clean[:limit]

    logger.info("ingest_records: %d valid records (batch_limit=%d)", len(clean), limit)
    return {**state, "records": clean, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — classify_invoices
# ─────────────────────────────────────────────────────────────────────────────
def classify_invoices(state: AgentState) -> AgentState:
    """
    Add 'stage' key to every record using get_stage().
    Stage 1-4 → email path.  Stage 5 → legal escalation path.
    """
    staged = []
    for record in state["records"]:
        stage = get_stage(
            days_overdue=int(record["days_overdue"]),
            follow_up_count=int(record.get("follow_up_count", 0)),
        )
        staged.append({**record, "stage": stage})

    stage_counts = {}
    for r in staged:
        stage_counts[r["stage"]] = stage_counts.get(r["stage"], 0) + 1
    logger.info("classify_invoices: %s", stage_counts)

    return {**state, "staged": staged, "current_idx": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — route_invoice  (conditional edge source)
# ─────────────────────────────────────────────────────────────────────────────
def route_invoice(state: AgentState) -> AgentState:
    """
    Passes state through unchanged.
    The actual routing decision is made in the conditional edge function
    _pick_path() below.  This node exists so the graph has a named node
    to attach the conditional edge to.
    """
    return state


def _pick_path(state: AgentState) -> str:
    """
    Conditional edge function.
    Returns the name of the next node based on the current record's stage.
    """
    idx = state["current_idx"]
    staged = state["staged"]

    if idx >= len(staged):
        return "done_node"

    stage = staged[idx]["stage"]
    return "generate_email" if stage < 5 else "escalate_record"


# ─────────────────────────────────────────────────────────────────────────────
# Node 4a — generate_email
# ─────────────────────────────────────────────────────────────────────────────
def generate_email_node(state: AgentState) -> AgentState:
    """
    Call generate_and_send() for the current record.
    Handles both dry-run and live SMTP modes.

    Security mitigations applied inside generate_and_send():
      - Pydantic EmailOutput validation       (Hallucination)
      - Input already sanitised by Node 1    (Prompt Injection)
      - api_key from env                     (API Key Exposure)
      - smtp_config locks sender domain      (Email Spoofing)
    """
    import time

    idx = state["current_idx"]
    record = state["staged"][idx]
    results = list(state["results"])
    errors = list(state["errors"])

    inv = _safe_invoice_no(record)
    logger.info(
        "generate_email [%d/%d]: %s (Stage %d — %s)",
        idx + 1, len(state["staged"]), inv,
        record["stage"],
        TONE_MATRIX.get(record["stage"], {}).get("label", "?"),
    )

    try:
        result = generate_and_send(
            record=record,
            stage=record["stage"],
            dry_run=state["dry_run"],
            api_key=state["api_key"] or os.getenv("GROQ_API_KEY"),
            smtp_config=state["smtp_config"],
        )
        results.append(result)
    except Exception as exc:
        logger.error("generate_email: %s failed — %s", inv, exc)
        errors.append((inv, str(exc)))

    # Rate-limit courtesy delay
    if state["delay_sec"] > 0 and (idx + 1) < len(state["staged"]):
        time.sleep(state["delay_sec"])

    return {**state, "results": results, "errors": errors, "current_idx": idx + 1}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4b — escalate_record
# ─────────────────────────────────────────────────────────────────────────────
def escalate_record_node(state: AgentState) -> AgentState:
    """
    Flag a 30+ day overdue record for legal/finance review.
    No email is generated.  Record is written to escalation_flags table.
    """
    idx = state["current_idx"]
    record = state["staged"][idx]
    escalations = list(state["escalations"])

    inv = _safe_invoice_no(record)
    logger.warning(
        "escalate_record [%d/%d]: %s — %d days overdue → legal flag",
        idx + 1, len(state["staged"]), inv, record["days_overdue"],
    )

    try:
        log_escalation(record, state["db_path"])
        escalations.append(record)
    except Exception as exc:
        logger.error("escalate_record: DB write failed for %s — %s", inv, exc)

    return {**state, "escalations": escalations, "current_idx": idx + 1}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — send_or_dryrun  (no-op: handled inside generate_and_send)
# ─────────────────────────────────────────────────────────────────────────────
def send_or_dryrun_node(state: AgentState) -> AgentState:
    """
    send_and_generate() already handled SMTP / dry-run.
    This node is a named checkpoint for observability / future hooks
    (e.g. LangSmith tracing, Langfuse spans, webhook callbacks).
    """
    last = state["results"][-1] if state["results"] else {}
    if last:
        logger.info(
            "send_or_dryrun: %s → status=%s",
            last.get("invoice_no", "?"), last.get("send_status", "?"),
        )
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — audit_log
# ─────────────────────────────────────────────────────────────────────────────
def audit_log_node(state: AgentState) -> AgentState:
    """
    Persist the most recently generated email result to the SQLite audit table.

    Security: mask_pii() is called inside log_email() before writing the
    body column — email addresses and phone numbers are redacted in the
    stored log text to reduce PII exposure in log files.
    """
    results = state["results"]
    if not results:
        return state

    last = results[-1]
    try:
        log_email(last, state["db_path"])
        logger.info("audit_log: logged %s (status=%s)", last["invoice_no"], last["send_status"])
    except Exception as exc:
        logger.error("audit_log: DB write failed for %s — %s", last["invoice_no"], exc)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — loop_or_end  (conditional edge: keep looping until all done)
# ─────────────────────────────────────────────────────────────────────────────
def loop_or_end_node(state: AgentState) -> AgentState:
    """Marker node; routing decided by _continue_or_end()."""
    return state


def _continue_or_end(state: AgentState) -> str:
    """Return 'route_invoice' to process next record, or 'end' when done."""
    if state["current_idx"] < len(state["staged"]):
        return "route_invoice"
    return "end"


# ─────────────────────────────────────────────────────────────────────────────
# Done sentinel node (reached when staged list is empty or all processed)
# ─────────────────────────────────────────────────────────────────────────────
def done_node(state: AgentState) -> AgentState:
    return {**state, "done": True}


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Graph topology
    ──────────────
    START → ingest_records → classify_invoices → route_invoice
                                                    │
                        ┌───────────────────────────┤
                        │ stage 1-4                 │ stage 5
                        ▼                           ▼
                  generate_email_node       escalate_record_node
                        │                           │
                        ▼                           │
                  send_or_dryrun_node               │
                        │                           │
                        ▼                           │
                  audit_log_node ←──────────────────┘
                        │
                        ▼
                  loop_or_end_node
                    │           │
               (more)           (done)
                 ▼                ▼
            route_invoice        END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("ingest_records",    ingest_records)
    graph.add_node("classify_invoices", classify_invoices)
    graph.add_node("route_invoice",     route_invoice)
    graph.add_node("generate_email",    generate_email_node)
    graph.add_node("escalate_record",   escalate_record_node)
    graph.add_node("send_or_dryrun",    send_or_dryrun_node)
    graph.add_node("audit_log",         audit_log_node)
    graph.add_node("loop_or_end",       loop_or_end_node)
    graph.add_node("done_node",         done_node)

    # Linear edges
    graph.add_edge(START, "ingest_records")
    graph.add_edge("ingest_records",    "classify_invoices")
    graph.add_edge("classify_invoices", "route_invoice")

    # Conditional: route_invoice → generate_email | escalate_record | done_node
    graph.add_conditional_edges(
        "route_invoice",
        _pick_path,
        {
            "generate_email":  "generate_email",
            "escalate_record": "escalate_record",
            "done_node":       "done_node",
        },
    )

    # Email path: generate → send → audit → loop
    graph.add_edge("generate_email",  "send_or_dryrun")
    graph.add_edge("send_or_dryrun",  "audit_log")
    graph.add_edge("audit_log",       "loop_or_end")

    # Escalation path: escalate → loop
    graph.add_edge("escalate_record", "loop_or_end")

    # Loop back or finish
    graph.add_conditional_edges(
        "loop_or_end",
        _continue_or_end,
        {
            "route_invoice": "route_invoice",
            "end":           "done_node",
        },
    )

    graph.add_edge("done_node", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────
def run_agent(
    records: list[dict],
    dry_run: bool = True,
    api_key: Optional[str] = None,
    smtp_config: Optional[SMTPConfig] = None,
    db_path: str = DB_PATH_DEFAULT,
    batch_limit: int = 0,
    delay_sec: float = 0.3,
) -> AgentState:
    """
    Execute the LangGraph credit follow-up agent on a list of invoice records.

    Parameters
    ──────────
    records     : list of invoice dicts (from CSV or mock data)
    dry_run     : True (default) = no real emails sent
    api_key     : Groq API key; falls back to GROQ_API_KEY env var
    smtp_config : SMTPConfig instance; None is safe in dry_run mode
    db_path     : SQLite audit DB path
    batch_limit : process only first N records; 0 = all
    delay_sec   : pause between Groq API calls (rate-limit courtesy)

    Returns
    ───────
    Final AgentState containing results, escalations, and errors.
    """
    init_db(db_path)

    initial_state: AgentState = {
        "records":     records,
        "staged":      [],
        "current_idx": 0,
        "results":     [],
        "escalations": [],
        "errors":      [],
        "dry_run":     dry_run,
        "api_key":     api_key or os.getenv("GROQ_API_KEY"),
        "smtp_config": smtp_config,
        "db_path":     db_path,
        "batch_limit": batch_limit,
        "delay_sec":   delay_sec,
        "done":        False,
    }

    graph = build_graph()
    final_state: AgentState = graph.invoke(initial_state)

    logger.info(
        "Agent run complete — %d emails, %d escalations, %d errors",
        len(final_state["results"]),
        len(final_state["escalations"]),
        len(final_state["errors"]),
    )
    return final_state


def run_agent_from_csv(
    csv_path: str = "data/invoices.csv",
    **kwargs,
) -> AgentState:
    """Convenience wrapper: load CSV then run the agent."""
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    return run_agent(records, **kwargs)
