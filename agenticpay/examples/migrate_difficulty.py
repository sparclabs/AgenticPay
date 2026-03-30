#!/usr/bin/env python3
"""One-shot migration: add --difficulty / adjust_zopa support to all task files.

Run from the repo root:
    python agenticpay/examples/migrate_difficulty.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# A single price-variable definition line inside main():
#   "    buyer_max_price = 150.0  # ..."
#   "    buyer1_max_price = 150.0  # ..."
#   "    seller_min_price = 80.0   # ..."
#   "    seller3_min_price = 90.0  # ..."
_PRICE_LINE = re.compile(
    r'^( {4})(buyer\d*_max_price|seller\d*_min_price)(\s*=\s*[\d.]+.*)',
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_price_block(lines: list) -> tuple:
    """Return (first_idx, last_idx) of the price-definition block, or (-1,-1)."""
    first = last = -1
    for i, line in enumerate(lines):
        if _PRICE_LINE.match(line):
            if first == -1:
                first = i
            last = i
    return first, last


def _collect_price_vars(lines: list) -> tuple:
    """Return sorted lists of buyer_max and seller_min variable names found."""
    buyers, sellers = [], []
    for line in lines:
        m = _PRICE_LINE.match(line)
        if m:
            var = m.group(2)
            if "buyer" in var:
                buyers.append(var)
            else:
                sellers.append(var)
    return buyers, sellers


def _build_adjustments(buyers: list, sellers: list, indent: str = "    ") -> list:
    """Generate the adjust_zopa call lines for the detected price variables."""
    lines = []

    if not buyers or not sellers:
        return lines  # nothing to adjust

    if len(buyers) == 1 and len(sellers) == 1:
        b, s = buyers[0], sellers[0]
        lines.append(f"{indent}{b}, {s} = adjust_zopa({b}, {s}, difficulty)")

    elif len(buyers) > 1 and len(sellers) == 1:
        s = sellers[0]
        for b in buyers:
            lines.append(f"{indent}{b}, _ = adjust_zopa({b}, {s}, difficulty)")

    elif len(buyers) == 1 and len(sellers) > 1:
        b = buyers[0]
        seller_list = ", ".join(sellers)
        lines.append(f"{indent}_ref_seller_min = min({seller_list})")
        lines.append(f"{indent}{b}, _ = adjust_zopa({b}, _ref_seller_min, difficulty)")

    else:  # multiple buyers, multiple sellers
        seller_list = ", ".join(sellers)
        lines.append(f"{indent}_ref_seller_min = min({seller_list})")
        for b in buyers:
            lines.append(f"{indent}{b}, _ = adjust_zopa({b}, _ref_seller_min, difficulty)")

    return lines


# ---------------------------------------------------------------------------
# Per-file transformation
# ---------------------------------------------------------------------------

_DIFFICULTY_ARG_BLOCK = (
    '    parser.add_argument(\n'
    '        "--difficulty",\n'
    '        choices=["normal", "hard", "no_deal"],\n'
    '        default=os.environ.get("DIFFICULTY", "normal"),\n'
    '        help="ZOPA difficulty: normal (default), hard (tight ZOPA ~5%% spread), "\n'
    '             "no_deal (buyer max < seller min — no rational agreement possible).",\n'
    '    )\n'
)


def transform(content: str) -> tuple:
    """Apply all difficulty-related changes to a task file's source text.

    Returns (new_content, list_of_changes_made).
    """
    changes = []

    # ---- 1. Add adjust_zopa to config import --------------------------------
    def _add_adjust_zopa(m):
        existing = m.group(0)
        if "adjust_zopa" in existing:
            return existing
        changes.append("added adjust_zopa to config import")
        return existing.rstrip() + ", adjust_zopa"

    content = re.sub(
        r'(from (?:agenticpay\.examples\.)?config import [^\n]+get_model_name[^\n]*)',
        _add_adjust_zopa,
        content,
    )

    # ---- 2. Update def main() signature -------------------------------------
    if "def main(model_name=None):" in content:
        content = content.replace(
            "def main(model_name=None):",
            "def main(model_name=None, difficulty=\"normal\"):",
        )
        changes.append("updated main() signature")

    # ---- 3. Insert --difficulty argparse arg --------------------------------
    if "--difficulty" not in content:
        # Insert just before args = parser.parse_args()
        content = content.replace(
            "    args = parser.parse_args()",
            _DIFFICULTY_ARG_BLOCK + "    args = parser.parse_args()",
        )
        changes.append("added --difficulty argparse argument")

    # ---- 4. Pass difficulty to main() call ----------------------------------
    content = re.sub(
        r'main\(model_name=args\.model\)',
        'main(model_name=args.model, difficulty=args.difficulty)',
        content,
    )
    if "difficulty=args.difficulty" in content:
        changes.append("passed difficulty to main() call")

    # ---- 5. Insert adjust_zopa calls after price-variable block -------------
    if "adjust_zopa(" not in content:
        lines = content.splitlines(keepends=True)
        first, last = _find_price_block(lines)

        if first != -1:
            buyers, sellers = _collect_price_vars(lines)
            adj_lines = _build_adjustments(buyers, sellers)

            if adj_lines:
                insert_at = last + 1
                adj_text = "".join(l + "\n" for l in adj_lines)
                lines.insert(insert_at, adj_text)
                content = "".join(lines)
                changes.append(
                    f"inserted adjust_zopa calls for "
                    f"{buyers} × {sellers}"
                )

    # ---- 6. Add difficulty to results dict ----------------------------------
    if '"difficulty"' not in content and "results = {" in content:
        content = content.replace(
            '"task":',
            '"difficulty": difficulty,\n        "task":',
        )
        changes.append('added "difficulty" to results dict')

    return content, changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SKIP_FILES = {
    "config.py",
    "config_example.py",
    "__init__.py",
    "migrate_difficulty.py",
}

EXAMPLES_ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate task files to support --difficulty")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing files",
    )
    args = parser.parse_args()

    task_files = [
        p for p in EXAMPLES_ROOT.rglob("*.py")
        if p.name not in SKIP_FILES
        and ".ipynb_checkpoints" not in str(p)
    ]

    total = changed = skipped = 0

    for path in sorted(task_files):
        total += 1
        original = path.read_text(encoding="utf-8")

        # Only process files that already import from config with get_model_name
        if "get_model_name" not in original:
            skipped += 1
            continue

        new_content, applied = transform(original)

        if not applied:
            skipped += 1
            continue

        changed += 1
        rel = path.relative_to(EXAMPLES_ROOT)
        print(f"{'[DRY RUN] ' if args.dry_run else ''}{'✓'} {rel}")
        for c in applied:
            print(f"    • {c}")

        if not args.dry_run:
            path.write_text(new_content, encoding="utf-8")

    print(f"\nTotal: {total}  Modified: {changed}  Skipped: {skipped}")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
