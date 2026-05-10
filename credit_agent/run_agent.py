"""
run_agent.py — CLI runner for the LangGraph Credit Follow-Up Email Agent.

Usage
─────
    # Dry run (default — no real emails sent):
    python run_agent.py --csv data/invoices.csv

    # Live send (SMTP must be configured in .env):
    python run_agent.py --csv data/invoices.csv --send

    # Limit to first 10 records, 0.5s delay:
    python run_agent.py --csv data/invoices.csv --batch 10 --delay 0.5

    # Output results to a custom path:
    python run_agent.py --csv data/invoices.csv --out output/my_run.json
"""

import argparse
import json
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from agents.credit_graph import run_agent_from_csv
from agents.email_agent import TONE_MATRIX

STAGE_EMOJI = {1: "😊", 2: "📋", 3: "⚠️", 4: "🔴", 5: "🚨"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance Credit Follow-Up Email Agent (LangGraph)"
    )
    parser.add_argument("--csv",     default="data/invoices.csv", help="Path to invoice CSV")
    parser.add_argument("--dry-run", action="store_true",  default=True,
                        help="Simulate sending — no real emails dispatched (default: True)")
    parser.add_argument("--send",    action="store_true",
                        help="Live SMTP send (overrides --dry-run)")
    parser.add_argument("--out",     default="output/results.json", help="Output JSON path")
    parser.add_argument("--batch",   type=int, default=0,
                        help="Process only the first N records (0 = all)")
    parser.add_argument("--delay",   type=float, default=0.5,
                        help="Seconds between Groq API calls (default: 0.5)")
    args = parser.parse_args()

    dry_run = not args.send

    print(f"\n{'='*65}")
    print(f"  Finance Credit Follow-Up Agent — LangGraph Edition")
    print(f"  Mode     : {'DRY RUN (no emails sent)' if dry_run else '⚡ LIVE SEND via SMTP'}")
    print(f"  CSV      : {args.csv}")
    print(f"  Batch    : {args.batch if args.batch else 'all records'}")
    print(f"  Delay    : {args.delay}s between API calls")
    print(f"{'='*65}\n")

    # ── Run the LangGraph agent ───────────────────────────────────────────────
    final_state = run_agent_from_csv(
        csv_path=args.csv,
        dry_run=dry_run,
        batch_limit=args.batch,
        delay_sec=args.delay,
    )

    # ── Print results summary ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  RESULTS")
    print(f"{'─'*65}")

    for res in final_state["results"]:
        emoji = STAGE_EMOJI.get(res["stage"], "📧")
        tone  = TONE_MATRIX.get(res["stage"], {}).get("label", "?")
        print(
            f"  {emoji}  Stage {res['stage']} | {res['invoice_no']} | "
            f"{res['client_name']} | ₹{int(res['amount']):,} | "
            f"{res['days_overdue']}d | {res['send_status']}"
        )
        print(f"       Subject: {res['subject'][:70]}{'…' if len(res['subject']) > 70 else ''}")

    for rec in final_state["escalations"]:
        print(
            f"  🚨 LEGAL FLAG | {rec['invoice_no']} | "
            f"{rec['client_name']} | {rec['days_overdue']}d overdue"
        )

    for inv, err in final_state["errors"]:
        print(f"  ❌ ERROR | {inv}: {err}")

    print(f"\n{'='*65}")
    print(f"  ✅ Agent run complete.")
    print(f"  📧 Emails generated : {len(final_state['results'])}")
    print(f"  🚨 Legal escalations: {len(final_state['escalations'])}")
    print(f"  ❌ Errors           : {len(final_state['errors'])}")

    # ── Save output JSON ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(final_state["results"], f, indent=2)
    print(f"  📄 Results saved to : {args.out}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
