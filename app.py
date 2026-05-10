"""
Finance Credit Follow-Up Email Agent — Streamlit Dashboard
Run: streamlit run app.py

v3.1 Changes
────────────
• LangSmith tracing integrated at startup (reads from .env)
• "Download All as PDF" button in Tab 2 (Generate Emails)
• "Download All as PDF" button in Tab 5 (Agent Run — LangGraph)
"""

import os
import json
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# LANGSMITH TRACING — set up before any LangGraph/LangChain imports
# ─────────────────────────────────────────────────────────────────────────────
_langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
_tracing_on    = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
_project       = os.getenv("LANGCHAIN_PROJECT", "credit-agent")

if _langsmith_key and _tracing_on:
    os.environ["LANGCHAIN_API_KEY"]      = _langsmith_key
    os.environ["LANGCHAIN_TRACING_V2"]   = "true"
    os.environ["LANGCHAIN_PROJECT"]      = _project
    _tracing_status = f"LangSmith tracing ON — project: **{_project}**"
    _tracing_ok     = True
else:
    # Make sure tracing is explicitly OFF so LangGraph doesn't warn
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    _tracing_status = "LangSmith tracing OFF (set LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true to enable)"
    _tracing_ok     = False

from agents.email_agent import (
    TONE_MATRIX,
    get_stage,
    generate_and_send,
    SMTPConfig,
    init_db,
    log_email,
    log_escalation,
    get_audit_log,
    get_escalations,
)
from agents.credit_graph import run_agent
from pdf_utils import generate_emails_pdf   

st.set_page_config(
    page_title="Credit Follow-Up Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {
        font-family: 'DM Sans', sans-serif !important;
        background-color: #f4f6f9 !important;
        color: #1a1a2e !important;
    }
    .block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #dde1ea !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] *:not(button):not(input) { color: #1a1a2e !important; }
    [data-testid="stSidebar"] .stTextInput input { background: #f4f6f9 !important; color: #1a1a2e !important; border: 1px solid #c9d0dc !important; border-radius: 8px !important; font-size: 13px !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #ffffff; border-radius: 10px; padding: 4px 6px; border: 1px solid #dde1ea; gap: 2px; margin-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 7px; color: #555e7a !important; font-weight: 500; font-size: 14px; padding: 8px 20px; border: none !important; }
    .stTabs [aria-selected="true"] { background-color: #1a1a2e !important; color: #ffffff !important; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dde1ea; border-radius: 12px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    [data-testid="stMetricLabel"] { color: #555e7a !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { color: #1a1a2e !important; font-size: 26px !important; font-weight: 700 !important; }
    .stButton > button { background: #1a1a2e !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 13.5px !important; padding: 9px 18px !important; }
    .stButton > button:hover { opacity: 0.82 !important; }
    .badge { display:inline-block; padding:3px 11px; border-radius:20px; font-size:12px; font-weight:600; }
    .b1 { background:#d1fae5; color:#065f46; }
    .b2 { background:#fef3c7; color:#78350f; }
    .b3 { background:#fee2e2; color:#7f1d1d; }
    .b4 { background:#f3e8ff; color:#581c87; }
    .b5 { background:#f3f4f6; color:#374151; }
    .inv-table { width:100%; border-collapse:collapse; font-size:13px; }
    .inv-table thead th { background:#f4f6f9; color:#374151; font-weight:600; padding:10px 13px; text-align:left; border-bottom:2px solid #dde1ea; white-space:nowrap; }
    .inv-table tbody td { padding:8px 13px; border-bottom:1px solid #f0f2f5; color:#1a1a2e; vertical-align:middle; }
    .inv-table tbody tr:hover td { background:#f9fafc; }
    .inv-table tbody tr:last-child td { border-bottom:none; }
    [data-testid="stExpander"] { background: #ffffff !important; border: 1px solid #dde1ea !important; border-radius: 10px !important; margin-bottom: 8px !important; overflow: hidden; }
    details > summary { color: #1a1a2e !important; font-weight: 500 !important; font-size: 13.5px !important; padding: 13px 16px !important; background: #ffffff !important; list-style: none; }
    details > summary:hover { background: #f9fafc !important; }
    details > summary p, details > summary span { color: #1a1a2e !important; }
    .streamlit-expanderHeader { color: #1a1a2e !important; }
    .streamlit-expanderHeader p { color: #1a1a2e !important; }
    .email-meta    { font-size:12px; color:#555e7a; font-weight:500; margin-bottom:10px; }
    .email-subject { font-size:14px; font-weight:700; color:#1a1a2e; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid #dde1ea; }
    .email-body    { font-size:13.5px; color:#374151; white-space:pre-wrap; line-height:1.75; font-family:'DM Sans',sans-serif; }
    .box { border-radius:8px; padding:12px 16px; font-size:13.5px; margin-bottom:14px; font-weight:500; }
    .bi  { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a; }
    .bw  { background:#fffbeb; border:1px solid #fde68a; color:#78350f; }
    .bs  { background:#f0fdf4; border:1px solid #bbf7d0; color:#14532d; }
    .be  { background:#fef2f2; border:1px solid #fecaca; color:#7f1d1d; }
    .bg  { background:#f0f4ff; border:1px solid #c7d2fe; color:#312e81; }
    .bv  { background:#f5f3ff; border:1px solid #ddd6fe; color:#4c1d95; }
    .pg-title { font-size:22px; font-weight:700; color:#1a1a2e; margin-bottom:4px; }
    .pg-sub   { font-size:13.5px; color:#555e7a; margin-bottom:22px; }
    .sec-lbl  { font-size:11px; font-weight:700; color:#555e7a; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:9px; }
    .s-row { display:flex; align-items:center; justify-content:space-between; padding:7px 11px; border-radius:7px; margin-bottom:4px; font-size:12.5px; }
    .stDownloadButton > button { background:#f4f6f9 !important; color:#1a1a2e !important; border:1px solid #dde1ea !important; border-radius:8px !important; font-weight:500 !important; font-size:13px !important; }
    .pdf-btn > button { background:#312e81 !important; color:#ffffff !important; border:none !important; border-radius:8px !important; font-weight:600 !important; font-size:13px !important; }
    .node-step { display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:8px; background:#f4f6f9; margin-bottom:6px; font-size:13px; }
    .node-pill { display:inline-block; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; }
    hr { border:none; border-top:1px solid #dde1ea; margin:18px 0; }
    #MainMenu, footer, header { visibility:hidden; }
    .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stTextInput label { color:#374151 !important; font-weight:500 !important; font-size:13px !important; }
</style>
""", unsafe_allow_html=True)

DB_PATH = "logs/audit.db"
init_db(DB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set. Add it to your .env file and restart.")
        st.stop()

    # ── LangSmith status indicator ────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">Observability</div>', unsafe_allow_html=True)
    if _tracing_ok:
        st.markdown(
            f'<div class="box bs" style="padding:7px 12px;font-size:12px;">'
            f'LangSmith tracing ON<br>'
            f'<span style="opacity:0.75;">Project: {_project}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="box bi" style="padding:7px 12px;font-size:12px;">'
            'LangSmith tracing OFF<br>'
            '<span style="opacity:0.75;">Set LANGCHAIN_API_KEY +<br>LANGCHAIN_TRACING_V2=true</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Send mode ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">Send Mode</div>', unsafe_allow_html=True)
    dry_run = st.checkbox("Dry-run mode (no real emails sent)", value=True)

    if dry_run:
        st.markdown(
            '<div class="box bi" style="padding:7px 12px;font-size:12px;">'
            'Emails generated and logged only — nothing dispatched.</div>',
            unsafe_allow_html=True,
        )
    else:
        smtp_cfg = SMTPConfig()
        if smtp_cfg.is_configured():
            st.markdown(
                f'<div class="box bs" style="padding:7px 12px;font-size:12px;">'
                f'Live mode via SMTP<br>'
                f'<span style="opacity:0.75;">Host: {smtp_cfg.host}:{smtp_cfg.port}<br>'
                f'From: {smtp_cfg.sender_email}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="box be" style="padding:7px 12px;font-size:12px;">'
                'Live mode selected but SMTP not configured.<br>'
                'Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SENDER_EMAIL in .env</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="sec-lbl">Escalation Stages</div>', unsafe_allow_html=True)
    stages_info = [
        ("Stage 1", "Warm and Friendly",  "1–7 days",   "#d1fae5", "#065f46"),
        ("Stage 2", "Polite but Firm",    "8–14 days",  "#fef3c7", "#78350f"),
        ("Stage 3", "Formal and Serious", "15–21 days", "#fee2e2", "#7f1d1d"),
        ("Stage 4", "Stern and Urgent",   "22–30 days", "#f3e8ff", "#581c87"),
        ("Stage 5", "Legal Flag",         "30+ days",   "#f3f4f6", "#374151"),
    ]
    for lbl, tone, rng, bg, fg in stages_info:
        st.markdown(
            f'<div class="s-row" style="background:{bg};">'
            f'<span style="font-weight:700;color:{fg};">{lbl}</span>'
            f'<span style="color:{fg};opacity:0.85;">{rng}</span>'
            f'</div>'
            f'<div style="color:#555e7a;font-size:11px;padding:1px 11px 6px;">{tone}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Finance Credit Follow-Up Agent v3.1 — LangGraph + LangSmith Edition")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — PDF download block (reused in Tab 2 and Tab 5)
# ─────────────────────────────────────────────────────────────────────────────
def _render_pdf_download(results: list[dict], key_suffix: str) -> None:
    """
    Renders a styled 'Download All Emails as PDF' button.
    key_suffix ensures unique widget keys between tabs.
    """
    if not results:
        return

    st.markdown("---")
    st.markdown('<div class="sec-lbl">Export</div>', unsafe_allow_html=True)

    dl_col, info_col = st.columns([2, 3])
    with dl_col:
        try:
            pdf_bytes = generate_emails_pdf(results)
            filename  = f"credit_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label=f"Download All {len(results)} Email(s) as PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key=f"pdf_dl_{key_suffix}",
                use_container_width=True,
            )
        except Exception as exc:
            st.markdown(
                f'<div class="box be">PDF generation failed: {exc}</div>',
                unsafe_allow_html=True,
            )
    with info_col:
        st.markdown(
            f'<div class="box bv" style="padding:8px 14px;font-size:12.5px;">'
            f'PDF includes a summary table + full email body for each of the '
            f'<b>{len(results)}</b> generated email(s). '
            f'Each email is formatted with its stage badge, subject, and body.'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard", "Generate Emails", "Audit Log", "Legal Escalations", "Agent Run (LangGraph)"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="pg-title">Finance Credit Follow-Up Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pg-sub">Automated follow-up emails with progressive tone escalation based on '
        'days overdue. Orchestrated by LangGraph — traced by LangSmith.</div>',
        unsafe_allow_html=True,
    )

    col_up, col_btn = st.columns([4, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"], label_visibility="collapsed",
            help="Columns: invoice_no, client_name, client_email, amount, due_date, follow_up_count, days_overdue",
        )
    with col_btn:
        use_sample = st.button("Load sample data", use_container_width=True)

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.session_state.pop("results", None)
    elif use_sample or "df" in st.session_state:
        if "df" not in st.session_state or use_sample:
            try:
                st.session_state["df"] = pd.read_csv("data/invoices.csv")
            except FileNotFoundError:
                st.error("Sample file not found at data/invoices.csv. Please upload a CSV.")
                st.stop()
        df = st.session_state["df"]
    else:
        st.markdown(
            '<div class="box bi">Upload a CSV file or click <b>Load sample data</b> to get started.<br>'
            '<span style="font-weight:400;opacity:0.8;">Required columns: invoice_no, client_name, '
            'client_email, amount, due_date, follow_up_count, days_overdue</span></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    df = st.session_state["df"].copy()
    df["stage"] = df.apply(lambda r: get_stage(r["days_overdue"], r["follow_up_count"]), axis=1)
    df["tone"]  = df["stage"].apply(lambda s: TONE_MATRIX.get(s, {}).get("label", "Legal Escalation"))
    st.session_state["df_staged"] = df

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Invoices",    len(df))
    c2.metric("Actionable (1–4)",  int((df["stage"] < 5).sum()))
    c3.metric("Stage 4 Urgent",    int((df["stage"] == 4).sum()))
    c4.metric("Legal Flags",       int((df["stage"] == 5).sum()))
    c5.metric("Total Outstanding", f"Rs {df['amount'].sum():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        st.markdown('<div class="sec-lbl">Stage Distribution</div>', unsafe_allow_html=True)
        sc = df.groupby("stage").size().reset_index(name="count")
        sc["Label"] = sc["stage"].apply(
            lambda s: TONE_MATRIX.get(s, {}).get("label", "Legal Flag") if s < 5 else "Legal Flag"
        )
        st.bar_chart(sc.set_index("Label")["count"], height=240)

    with col_table:
        st.markdown('<div class="sec-lbl">Invoice Queue</div>', unsafe_allow_html=True)
        badge_cls = {1:"b1", 2:"b2", 3:"b3", 4:"b4", 5:"b5"}
        stage_lbl = {1:"Stage 1", 2:"Stage 2", 3:"Stage 3", 4:"Stage 4", 5:"Legal"}
        rows_html = ""
        for _, row in df.head(100).iterrows():
            bc = badge_cls.get(row["stage"], "b5")
            sl = stage_lbl.get(row["stage"], "?")
            rows_html += (
                f"<tr><td>{row['invoice_no']}</td>"
                f"<td><b>{row['client_name']}</b></td>"
                f"<td style='color:#555e7a'>{row['client_email']}</td>"
                f"<td><b>Rs {int(row['amount']):,}</b></td>"
                f"<td>{row['days_overdue']}d</td>"
                f"<td><span class='badge {bc}'>{sl}</span></td></tr>"
            )
        note = (
            f'<p style="font-size:12px;color:#555e7a;margin-top:8px;">'
            f'Showing 100 of {len(df)}. All processed in Generate tab.</p>'
            if len(df) > 100 else ""
        )
        st.write(
            f'<div style="max-height:340px;overflow-y:auto;border:1px solid #dde1ea;border-radius:10px;background:#fff;">'
            f'<table class="inv-table"><thead><tr>'
            f'<th>Invoice</th><th>Client</th><th>Email</th><th>Amount</th><th>Overdue</th><th>Stage</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>{note}',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Generate Emails
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="pg-title">Generate Emails</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pg-sub">Generate tone-matched follow-up emails via Groq / Llama 3.3 70B. '
        f'Mode: <b>{"Dry Run (log only)" if dry_run else "Live Send via SMTP"}</b>. '
        f'For full batch processing use the <b>Agent Run</b> tab.</div>',
        unsafe_allow_html=True,
    )

    if "df_staged" not in st.session_state:
        st.markdown('<div class="box bw">Load invoice data from the Dashboard tab first.</div>', unsafe_allow_html=True)
        st.stop()

    smtp_config = None
    if not dry_run:
        smtp_config = SMTPConfig()
        if not smtp_config.is_configured():
            st.markdown(
                '<div class="box be">⛔ <b>Live send mode is on but SMTP is not configured.</b><br>'
                'Add SMTP credentials to your .env file, or switch to Dry-run mode in the sidebar.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

    df       = st.session_state["df_staged"].copy()
    email_df = df[df["stage"] < 5].copy()
    legal_df = df[df["stage"] == 5].copy()
    options  = email_df["invoice_no"].tolist()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="sec-lbl">Select Invoices</div>', unsafe_allow_html=True)
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("Select all", use_container_width=True):
                st.session_state["sel"] = options
        with qc2:
            if st.button("Urgent only (3 and 4)", use_container_width=True):
                st.session_state["sel"] = email_df[email_df["stage"] >= 3]["invoice_no"].tolist()
        with qc3:
            if st.button("Clear selection", use_container_width=True):
                st.session_state["sel"] = []

        default_sel = [x for x in st.session_state.get("sel", options[:5]) if x in options]
        selected = st.multiselect("Invoices", options=options, default=default_sel, label_visibility="collapsed")
        st.session_state["sel"] = selected

        if legal_df.shape[0] > 0:
            st.markdown(
                f'<div class="box bw"><b>{legal_df.shape[0]} invoice(s)</b> are 30+ days overdue and '
                f'flagged for legal review. No emails will be sent — they will be logged as escalations.</div>',
                unsafe_allow_html=True,
            )
        if selected:
            sel_total = email_df[email_df["invoice_no"].isin(selected)]["amount"].sum()
            st.markdown(
                f'<div class="box bi"><b>{len(selected)}</b> invoice(s) selected '
                f'&nbsp;&middot;&nbsp; Outstanding: <b>Rs {sel_total:,.0f}</b></div>',
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown('<div class="sec-lbl">Stage Breakdown</div>', unsafe_allow_html=True)
        sc2 = email_df.groupby("stage").size().reset_index(name="count")
        sc2["Tone"] = sc2["stage"].apply(lambda s: TONE_MATRIX.get(s, {}).get("label", "?"))
        st.bar_chart(sc2.set_index("Tone")["count"], height=190)

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])
    with col_right:
        batch_size = st.number_input("Max invoices per batch", min_value=1, max_value=200, value=20)
        delay_sec  = st.number_input("Delay between calls (seconds)", min_value=0.0, max_value=5.0, value=0.3, step=0.1)
    with col_left:
        mode_label = "Dry Run" if dry_run else "Send Live"
        run_label  = f"{mode_label} — {len(selected)} email(s)" if selected else "Select invoices above to continue"
        run_btn    = st.button(run_label, type="primary", disabled=not selected, use_container_width=True)

    if run_btn and selected:
        if "results" not in st.session_state:
            st.session_state["results"] = []

        subset  = email_df[email_df["invoice_no"].isin(selected)].head(int(batch_size))
        n       = len(subset)
        prog    = st.progress(0, text=f"Starting — 0 of {n}...")
        res_run = []
        err_run = []

        for i, (_, row) in enumerate(subset.iterrows(), 1):
            prog.progress(
                i / n,
                text=f"{'Sending' if not dry_run else 'Generating'} {i} of {n}: {row['invoice_no']}...",
            )
            try:
                res = generate_and_send(
                    record=row.to_dict(), stage=row["stage"],
                    dry_run=dry_run, api_key=api_key, smtp_config=smtp_config,
                )
                log_email(res, DB_PATH)
                res_run.append(res)
                st.session_state["results"].append(res)
            except Exception as e:
                err_run.append(f"{row['invoice_no']}: {e}")
            if delay_sec > 0 and i < n:
                time.sleep(delay_sec)

        for _, row in legal_df.iterrows():
            try:
                log_escalation(row.to_dict(), DB_PATH)
            except Exception:
                pass

        prog.progress(1.0, text="Done.")
        sent_count   = sum(1 for r in res_run if r["send_status"] == "SENT")
        failed_count = sum(1 for r in res_run if str(r["send_status"]).startswith("FAILED"))
        dry_count    = sum(1 for r in res_run if r["send_status"] == "DRY_RUN")
        skipped      = len(selected) - len(subset)

        summary = (
            f"<b>{dry_count} email(s) generated and logged</b> (dry-run)." if dry_run else
            f"<b>{sent_count} email(s) sent.</b>" +
            (f" {failed_count} failed." if failed_count else "")
        )
        if skipped:
            summary += f" {skipped} skipped — increase batch size."

        box_class = "bs" if failed_count == 0 else "bw"
        st.markdown(f'<div class="box {box_class}">{summary}</div>', unsafe_allow_html=True)
        for err in err_run:
            st.markdown(f'<div class="box be">❌ {err}</div>', unsafe_allow_html=True)

        # ── PDF download (immediately after generation) ────────────────────────
        if res_run:
            _render_pdf_download(res_run, key_suffix="tab2")

        # ── Individual email previews ─────────────────────────────────────────
        st.markdown('<div class="sec-lbl" style="margin-top:18px;">Generated Emails</div>', unsafe_allow_html=True)
        for res in res_run:
            bc  = f"b{res['stage']}"
            status_icon = {"SENT": "✅", "DRY_RUN": "📋"}.get(res["send_status"], "❌")
            lbl = (
                f"{status_icon}  {res['invoice_no']}  |  {res['client_name']}  |  "
                f"Stage {res['stage']}: {res['tone']}  |  Rs {int(res['amount']):,}"
            )
            with st.expander(lbl, expanded=(len(res_run) <= 4)):
                st.markdown(f'<span class="badge {bc}">Stage {res["stage"]}: {res["tone"]}</span>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="email-meta" style="margin-top:10px;">'
                    f'To: {res["client_email"]} &nbsp;&middot;&nbsp; {res["timestamp"][:19]} UTC</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="email-subject">Subject: {res["subject"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="email-body">{res["body"]}</div>', unsafe_allow_html=True)
                st.download_button(
                    "Download this email as .txt",
                    data=res["full_email"],
                    file_name=f"{res['invoice_no']}_email.txt",
                    mime="text/plain",
                    key=f"dl_{res['invoice_no']}_{res['timestamp']}",
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Audit Log
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="pg-title">Audit Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Every generated email is permanently recorded here for compliance and review.</div>', unsafe_allow_html=True)

    logs = get_audit_log(DB_PATH)

    if not logs:
        st.markdown('<div class="box bi">No emails logged yet. Generate some emails first.</div>', unsafe_allow_html=True)
    else:
        sc_log = {}
        for lg in logs:
            sc_log[lg["stage"]] = sc_log.get(lg["stage"], 0) + 1
        top_s = max(sc_log, key=sc_log.get) if sc_log else "-"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Logged", len(logs))
        c2.metric("Dry Runs",     sum(1 for lg in logs if lg["dry_run"]))
        c3.metric("Live Sent",    sum(1 for lg in logs if not lg["dry_run"] and lg["send_status"] == "SENT"))
        c4.metric("Top Stage",    f"Stage {top_s}" if top_s != "-" else "-")

        st.markdown("<br>", unsafe_allow_html=True)
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1: f_stage  = st.selectbox("Stage",  ["All"] + [str(i) for i in range(1, 5)])
        with fcol2: f_status = st.selectbox("Status", ["All", "DRY_RUN", "SENT", "FAILED"])
        with fcol3: f_search = st.text_input("Search", placeholder="Invoice no. or client name")

        fl = logs
        if f_stage  != "All": fl = [lg for lg in fl if lg["stage"] == int(f_stage)]
        if f_status != "All":
            if f_status == "FAILED":
                fl = [lg for lg in fl if str(lg["send_status"]).startswith("FAILED")]
            else:
                fl = [lg for lg in fl if lg["send_status"] == f_status]
        if f_search:
            s  = f_search.lower()
            fl = [lg for lg in fl if s in lg["invoice_no"].lower() or s in lg["client_name"].lower()]

        st.caption(f"Showing {len(fl)} of {len(logs)} record(s)")
        for lg in fl:
            bc  = f"b{lg['stage']}"
            lbl = (
                f"{lg['invoice_no']}  |  {lg['client_name']}  |  "
                f"Stage {lg['stage']}  |  {lg['timestamp'][:19]}  |  {lg['send_status']}"
            )
            with st.expander(lbl):
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Amount",       f"Rs {int(lg['amount']):,}")
                r2.metric("Days Overdue", lg["days_overdue"])
                r3.metric("Stage",        lg["stage"])
                r4.metric("Status",       lg["send_status"])
                st.markdown(f'<span class="badge {bc}">{lg["tone"]}</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="email-subject" style="margin-top:10px;">Subject: {lg["subject"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="email-body">{lg["body"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "Download full audit log as JSON",
            data=json.dumps(logs, indent=2),
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Legal Escalations
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="pg-title">Legal Escalations</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Invoices 30+ days overdue. No automated emails sent — manual review required.</div>', unsafe_allow_html=True)

    flags = get_escalations(DB_PATH)

    if not flags:
        st.markdown('<div class="box bi">No escalations on record.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        c1.metric("Total Escalated", len(flags))
        c2.metric("Amount at Risk",  f"Rs {sum(f['amount'] for f in flags):,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        esc_df = pd.DataFrame(flags)
        esc_df["Amount"] = esc_df["amount"].apply(lambda x: f"Rs {int(x):,}")
        esc_df = esc_df[["timestamp", "invoice_no", "client_name", "Amount", "days_overdue", "reason"]].copy()
        esc_df.columns = ["Timestamp", "Invoice", "Client", "Amount", "Days Overdue", "Reason"]
        st.dataframe(esc_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "Download escalation report as JSON",
            data=json.dumps(flags, indent=2),
            file_name=f"escalations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Agent Run (LangGraph)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="pg-title">Agent Run — LangGraph</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pg-sub">Execute the full LangGraph pipeline: ingest → classify → route → '
        'generate/escalate → send/dry-run → audit. All records processed automatically in one run.</div>',
        unsafe_allow_html=True,
    )

    # LangSmith notice
    if _tracing_ok:
        st.markdown(
            f'<div class="box bs" style="padding:8px 14px;font-size:12.5px;">'
            f'LangSmith tracing is active — view traces at '
            f'<a href="https://smith.langchain.com" target="_blank">smith.langchain.com</a> '
            f'under project <b>{_project}</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="box bi" style="padding:8px 14px;font-size:12.5px;">'
            'LangSmith tracing is off. Add <code>LANGCHAIN_API_KEY</code> and '
            '<code>LANGCHAIN_TRACING_V2=true</code> to your .env to enable full agent tracing.</div>',
            unsafe_allow_html=True,
        )

    # ── Graph nodes reference ─────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">Graph Nodes</div>', unsafe_allow_html=True)
    node_info = [
        ("ingest_records",    "Sanitise all CSV records (prompt injection defence)", "#d1fae5", "#065f46"),
        ("classify_invoices", "Assign stage 1–5 per record based on days overdue",  "#eff6ff", "#1e3a8a"),
        ("route_invoice",     "Conditional edge: stage 1–4 → email, stage 5 → legal","#fef3c7", "#78350f"),
        ("generate_email",    "LLM (Groq / Llama 3.3 70B) + Pydantic validation",   "#eff6ff", "#1e3a8a"),
        ("escalate_record",   "Flag 30+ day records for legal review (no email)",    "#fef2f2", "#7f1d1d"),
        ("send_or_dryrun",    "SMTP live send or dry-run checkpoint + LangSmith span","#eff6ff", "#1e3a8a"),
        ("audit_log",         "Persist result to SQLite (PII-masked body column)",   "#f0fdf4", "#14532d"),
    ]
    cols = st.columns(2)
    for i, (node, desc, bg, fg) in enumerate(node_info):
        with cols[i % 2]:
            st.markdown(
                f'<div class="node-step" style="background:{bg};">'
                f'<span class="node-pill" style="background:{fg};color:#fff;">{node}</span>'
                f'<span style="color:{fg};font-size:12px;">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    if "df_staged" not in st.session_state:
        st.markdown('<div class="box bw">Load invoice data from the Dashboard tab first.</div>', unsafe_allow_html=True)
        st.stop()

    df_all = st.session_state["df_staged"].copy()

    # ── Run settings ──────────────────────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        agent_batch = st.number_input(
            "Batch limit (0 = all records)", min_value=0, max_value=500, value=0,
            help="0 processes all invoices. Set a limit for testing.",
        )
    with col_cfg2:
        agent_delay = st.number_input(
            "Delay between API calls (s)", min_value=0.0, max_value=5.0, value=0.3, step=0.1,
        )
    with col_cfg3:
        st.markdown("<br>", unsafe_allow_html=True)
        records_to_process = len(df_all) if agent_batch == 0 else min(int(agent_batch), len(df_all))
        st.markdown(
            f'<div class="box bi" style="padding:8px 12px;">'
            f'Records to process: <b>{records_to_process}</b><br>'
            f'Mode: <b>{"Dry Run" if dry_run else "Live Send"}</b></div>',
            unsafe_allow_html=True,
        )

    # ── SMTP check for live mode ──────────────────────────────────────────────
    agent_smtp = None
    if not dry_run:
        agent_smtp = SMTPConfig()
        if not agent_smtp.is_configured():
            st.markdown(
                '<div class="box be">⛔ SMTP not configured — switch to Dry-run or add credentials to .env.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

    run_agent_btn = st.button(
        f"▶  Run LangGraph Agent — {records_to_process} records",
        type="primary",
        use_container_width=True,
    )

    if run_agent_btn:
        records = df_all.to_dict(orient="records")

        with st.spinner("LangGraph agent running…"):
            t0 = time.time()
            final_state = run_agent(
                records=records,
                dry_run=dry_run,
                api_key=api_key,
                smtp_config=agent_smtp,
                db_path=DB_PATH,
                batch_limit=int(agent_batch),
                delay_sec=float(agent_delay),
            )
            elapsed = time.time() - t0

        # ── Summary metrics ───────────────────────────────────────────────────
        n_emails    = len(final_state["results"])
        n_escalated = len(final_state["escalations"])
        n_errors    = len(final_state["errors"])
        sent_cnt    = sum(1 for r in final_state["results"] if r["send_status"] == "SENT")
        dry_cnt     = sum(1 for r in final_state["results"] if r["send_status"] == "DRY_RUN")
        failed_cnt  = sum(1 for r in final_state["results"] if str(r["send_status"]).startswith("FAILED"))

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Emails generated", n_emails)
        m2.metric("Dry-run logged",   dry_cnt)
        m3.metric("Sent (live)",      sent_cnt)
        m4.metric("Escalated",        n_escalated)
        m5.metric("Elapsed",          f"{elapsed:.1f}s")

        if n_errors:
            st.markdown(f'<div class="box bw">⚠️ {n_errors} error(s) — see details below.</div>', unsafe_allow_html=True)
            for inv, err in final_state["errors"]:
                st.markdown(f'<div class="box be">❌ {inv}: {err}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="box bs">✅ Agent run complete — '
                f'{n_emails} email(s) generated, {n_escalated} legal flag(s). '
                f'All results saved to audit DB.</div>',
                unsafe_allow_html=True,
            )

        # ── PDF + JSON downloads ──────────────────────────────────────────────
        if final_state["results"]:
            _render_pdf_download(final_state["results"], key_suffix="tab5")

            dl_json_col, _ = st.columns([2, 3])
            with dl_json_col:
                st.download_button(
                    "Download agent run results as JSON",
                    data=json.dumps(final_state["results"], indent=2),
                    file_name=f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="json_dl_tab5",
                )

        # ── Email previews ────────────────────────────────────────────────────
        if final_state["results"]:
            st.markdown('<div class="sec-lbl" style="margin-top:20px;">Generated Emails</div>', unsafe_allow_html=True)
            for res in final_state["results"]:
                bc  = f"b{res['stage']}"
                status_icon = {"SENT": "✅", "DRY_RUN": "📋"}.get(res["send_status"], "❌")
                lbl = (
                    f"{status_icon}  {res['invoice_no']}  |  {res['client_name']}  |  "
                    f"Stage {res['stage']}: {res['tone']}  |  Rs {int(res['amount']):,}  |  {res['send_status']}"
                )
                with st.expander(lbl, expanded=False):
                    st.markdown(f'<span class="badge {bc}">Stage {res["stage"]}: {res["tone"]}</span>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="email-meta" style="margin-top:10px;">'
                        f'To: {res["client_email"]} &nbsp;&middot;&nbsp; {res["timestamp"][:19]} UTC</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<div class="email-subject">Subject: {res["subject"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="email-body">{res["body"]}</div>', unsafe_allow_html=True)
                    st.download_button(
                        "Download this email as .txt",
                        data=res["full_email"],
                        file_name=f"{res['invoice_no']}_email.txt",
                        mime="text/plain",
                        key=f"agent_dl_{res['invoice_no']}_{res['timestamp']}",
                    )

        # ── Legal escalations ─────────────────────────────────────────────────
        if final_state["escalations"]:
            st.markdown('<div class="sec-lbl" style="margin-top:20px;">Legal Escalations (this run)</div>', unsafe_allow_html=True)
            esc_data = [
                {
                    "Invoice":      r.get("invoice_no"),
                    "Client":       r.get("client_name"),
                    "Amount":       f"Rs {int(r.get('amount', 0)):,}",
                    "Days Overdue": r.get("days_overdue"),
                }
                for r in final_state["escalations"]
            ]
            st.dataframe(pd.DataFrame(esc_data), use_container_width=True, hide_index=True)
