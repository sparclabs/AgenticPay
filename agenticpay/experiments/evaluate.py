#!/usr/bin/env python3
"""Aggregate negotiation evaluation results.

Computes GlobalScore, BuyerScore, SellerScore, Deal Rate, and Overflow Rate
from the summary.json files produced by task runs.

Usage:
    python agenticpay/experiments/evaluate.py
    python agenticpay/experiments/evaluate.py --model Qwen3.5
    python agenticpay/experiments/evaluate.py --results-dir /path/to/results --csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


# ---------------------------------------------------------------------------
# Result file discovery
# ---------------------------------------------------------------------------

def find_result_files(results_dir: Path) -> list:
    return sorted(results_dir.rglob("summary.json"))


# ---------------------------------------------------------------------------
# Overflow detection
# ---------------------------------------------------------------------------

def _check_overflow(data: dict) -> bool:
    """Return True if any final agreed price violates a private constraint.

    Price overflow = buyer paid strictly above their max acceptable price, OR
                     seller accepted strictly below their min acceptable price.
    Only checked when the negotiation reached agreement (success=True).
    """
    if not data.get("success"):
        return False

    selected_buyer = data.get("selected_buyer")
    selected_seller = data.get("selected_seller")

    if selected_buyer and selected_seller:
        # Multi-buyer × multi-seller: b{n}s{m}_buyer_price / b{n}s{m}_seller_price
        b, s = selected_buyer, selected_seller
        bp = data.get(f"b{b}s{s}_buyer_price")
        sp = data.get(f"b{b}s{s}_seller_price")
        bmax = data.get(f"buyer{b}_max_price")
        smin = data.get(f"seller{s}_min_price")

    elif selected_buyer:
        # Multi-buyer, single seller: buyer{n}_price / seller_price_buyer{n}
        b = selected_buyer
        bp = data.get(f"buyer{b}_price")
        sp = data.get(f"seller_price_buyer{b}")
        bmax = data.get(f"buyer{b}_max_price")
        smin = data.get("seller_min_price")

    elif selected_seller:
        # Single buyer, multi-seller: buyer_price_seller{n} / seller{n}_price
        s = selected_seller
        bp = data.get(f"buyer_price_seller{s}")
        sp = data.get(f"seller{s}_price")
        bmax = data.get("buyer_max_price")
        smin = data.get(f"seller{s}_min_price")

    else:
        # Standard single-buyer single-seller
        bp = data.get("buyer_price") or data.get("agreed_price")
        sp = data.get("seller_price") or data.get("agreed_price")
        bmax = data.get("buyer_max_price")
        smin = data.get("seller_min_price")

    EPS = 1e-6
    if bp is not None and bmax is not None and bp > bmax + EPS:
        return True
    if sp is not None and smin is not None and sp < smin - EPS:
        return True
    return False


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(path: Path, model_filter) -> dict:
    """Parse one summary.json and return a normalised metrics dict, or None to skip."""
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  Warning: could not read {path}: {exc}", file=sys.stderr)
        return None

    model = data.get("model", "")
    if model_filter and model_filter.lower() not in model.lower():
        return None

    # Category is 4th component from the end:
    #   results/<category>/<model_safe>/batch_evaluation_<ts>/summary.json
    parts = path.parts
    category = parts[-4] if len(parts) >= 4 else "unknown"

    return {
        "model": model,
        "category": category,
        "success": bool(data.get("success", False)),
        "global_score": data.get("global_score"),
        "buyer_score": data.get("buyer_score"),
        "seller_score": data.get("seller_score"),
        "overflow": _check_overflow(data),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list) -> dict:
    n = len(records)
    if n == 0:
        return dict(n=0, deal_rate=None, overflow_rate=None,
                    global_score=None, buyer_score=None, seller_score=None)

    deal_rate     = sum(r["success"]  for r in records) / n * 100
    overflow_rate = sum(r["overflow"] for r in records) / n * 100

    gs = [r["global_score"] for r in records if r["global_score"] is not None]
    bs = [r["buyer_score"]  for r in records if r["buyer_score"]  is not None]
    ss = [r["seller_score"] for r in records if r["seller_score"] is not None]

    return dict(
        n=n,
        deal_rate=deal_rate,
        overflow_rate=overflow_rate,
        global_score=mean(gs) if gs else None,
        buyer_score=mean(bs)  if bs else None,
        seller_score=mean(ss) if ss else None,
    )


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_COLS   = ["Category/Label",       "N",  "Deal%", "Overflow%", "GlobalScore", "BuyerScore", "SellerScore"]
_WIDTHS = [                    36,     5,       9,          10,           12,           11,           12]


def _fmt(val, decimals=1):
    return f"{val:.{decimals}f}" if val is not None else "—"


def _sep(widths):
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def print_table(title: str, rows: list) -> None:
    sep = _sep(_WIDTHS)
    header_cells = [h.center(w) for h, w in zip(_COLS, _WIDTHS)]
    header_line = "| " + " | ".join(header_cells) + " |"

    bar = "=" * len(sep)
    print(f"\n{bar}")
    print(title.center(len(sep)))
    print(bar)
    print(sep)
    print(header_line)
    print(sep)

    for label, m in rows:
        cells = [
            label.ljust(_WIDTHS[0]),
            str(m["n"]).rjust(_WIDTHS[1]),
            _fmt(m["deal_rate"]).rjust(_WIDTHS[2]),
            _fmt(m["overflow_rate"]).rjust(_WIDTHS[3]),
            _fmt(m["global_score"],  2).rjust(_WIDTHS[4]),
            _fmt(m["buyer_score"],   2).rjust(_WIDTHS[5]),
            _fmt(m["seller_score"],  2).rjust(_WIDTHS[6]),
        ]
        print("| " + " | ".join(cells) + " |")

    print(sep)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "n", "deal_rate_%", "overflow_rate_%",
                    "global_score", "buyer_score", "seller_score"])
        for label, m in rows:
            w.writerow([
                label,
                m["n"],
                round(m["deal_rate"],     4) if m["deal_rate"]     is not None else "",
                round(m["overflow_rate"], 4) if m["overflow_rate"] is not None else "",
                round(m["global_score"],  4) if m["global_score"]  is not None else "",
                round(m["buyer_score"],   4) if m["buyer_score"]   is not None else "",
                round(m["seller_score"],  4) if m["seller_score"]  is not None else "",
            ])
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Aggregate negotiation evaluation results (GlobalScore, BuyerScore, "
                    "SellerScore, Deal Rate, Overflow Rate)."
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=repo_root / "agenticpay" / "results",
        help="Root results directory  (default: agenticpay/results/)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Filter by model name substring, e.g. 'Qwen3.5'",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Also write overall.csv and by_category.csv into --results-dir",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    files = find_result_files(args.results_dir)
    if not files:
        print(f"No summary.json files found under {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    records = [m for f in files for m in [extract_metrics(f, args.model)] if m]

    if not records:
        suffix = f" matching model '{args.model}'" if args.model else ""
        print(f"No results found{suffix}.", file=sys.stderr)
        sys.exit(1)

    model_tag = f"  (model filter: '{args.model}')" if args.model else ""
    print(f"\nLoaded {len(records)} result(s) from {args.results_dir}{model_tag}")

    # Overall
    overall_rows = [("Overall", aggregate(records))]
    print_table("Overall Results", overall_rows)

    # Per-category
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    cat_rows = [(cat, aggregate(recs)) for cat, recs in sorted(by_cat.items())]
    print_table("Results by Category", cat_rows)

    # Per-model (only printed when results span multiple models)
    models_seen = sorted({r["model"] for r in records})
    if len(models_seen) > 1:
        model_rows = [(m, aggregate([r for r in records if r["model"] == m]))
                      for m in models_seen]
        print_table("Results by Model", model_rows)

    # CSV
    if args.csv:
        print("\nWriting CSV files...")
        save_csv(args.results_dir / "overall.csv",     overall_rows)
        save_csv(args.results_dir / "by_category.csv", cat_rows)
        if len(models_seen) > 1:
            save_csv(args.results_dir / "by_model.csv", model_rows)


if __name__ == "__main__":
    main()
