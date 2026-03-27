"""
label.py — Manual ground truth labeling tool for LLMDataBench

Usage:
    python3 label.py /path/to/dataset.csv --rows 100 --output output/ground_truth.csv
    python3 label.py /path/to/dataset.csv --resume output/ground_truth.csv
"""

import os
import argparse
import pandas as pd


COLORS = {
    "header":  "\033[95m",
    "blue":    "\033[94m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "red":     "\033[91m",
    "bold":    "\033[1m",
    "end":     "\033[0m",
}

def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['end']}"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(current, total, output_path):
    print(c("="*60, "blue"))
    print(c("  LLMDataBench — Manual Labeling Tool", "bold"))
    print(c("="*60, "blue"))
    print(f"  Progress : {c(str(current), 'green')}/{c(str(total), 'green')} rows labeled")
    print(f"  Saving to: {c(output_path, 'yellow')}")
    print(c("-"*60, "blue"))


def print_row(row: dict, row_index: int):
    print(f"\n{c(f'Row #{row_index + 1}', 'bold')}\n")
    for col, val in row.items():
        val_str = str(val)
        # truncate long values for display
        if len(val_str) > 300:
            val_str = val_str[:300] + "..."
        print(f"  {c(col, 'yellow')}: {val_str}")
    print()


def get_score() -> int:
    while True:
        try:
            score = input(c("  Score (0-10): ", "bold")).strip()
            if score == "":
                print(c("  Score cannot be empty", "red"))
                continue
            score = int(score)
            if 0 <= score <= 10:
                return score
            print(c("  Please enter a number between 0 and 10", "red"))
        except ValueError:
            print(c("  Please enter a valid number", "red"))


def get_flags() -> str:
    print(c("  Flags (comma separated, or press Enter for none):", "bold"))
    print(c("  Examples: 'Age out of range', 'Missing billing amount', 'Invalid date'", "yellow"))
    flags = input("  > ").strip()
    return flags if flags else "No issues found"


def get_reasoning() -> str:
    print(c("  Reasoning (brief explanation, or press Enter to skip):", "bold"))
    reasoning = input("  > ").strip()
    return reasoning if reasoning else ""


def load_existing_labels(output_path: str) -> dict:
    """Load existing labels to support resume functionality."""
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        if "row_index" in df.columns:
            return {int(row["row_index"]): row for _, row in df.iterrows()}
    return {}


def run_labeler(csv_path: str, rows: int = None, output_path: str = "output/ground_truth.csv", resume: bool = False):
    # load dataset
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    if rows:
        df = df.sample(n=min(rows, total_rows), random_state=42).reset_index(drop=True)

    total = len(df)

    # load existing labels if resuming
    existing_labels = {}
    if resume and os.path.exists(output_path):
        existing_labels = load_existing_labels(output_path)
        print(c(f"\nResuming — {len(existing_labels)} rows already labeled", "green"))
        input("Press Enter to continue...")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    labeled_rows = []
    skipped      = 0

    for i, row in df.iterrows():
        # skip already labeled rows when resuming
        if i in existing_labels:
            labeled_rows.append(existing_labels[i].to_dict())
            continue

        clear()
        print_header(len(labeled_rows) - skipped, total, output_path)
        print_row(row.to_dict(), i)

        print(c("  Commands: [s] skip  [q] quit and save  [number] score", "blue"))
        print()

        # get score
        score_input = input(c("  Score (0-10) or [s]kip or [q]uit: ", "bold")).strip().lower()

        if score_input == "q":
            print(c("\n  Saving and quitting...", "yellow"))
            break

        if score_input == "s":
            skipped += 1
            continue

        try:
            score = int(score_input)
            if not 0 <= score <= 10:
                raise ValueError
        except ValueError:
            print(c("  Invalid input — skipping row", "red"))
            skipped += 1
            continue

        # get flags
        print()
        flags    = get_flags()
        print()
        reasoning = get_reasoning()

        labeled_rows.append({
            "row_index": i,
            **row.to_dict(),
            "human_score":     score,
            "human_flags":     flags,
            "human_reasoning": reasoning,
        })

        # save after every label — never lose progress
        pd.DataFrame(labeled_rows).to_csv(output_path, index=False)

        print(c(f"\n  ✓ Saved row {i+1} — score: {score}", "green"))

    # final save
    if labeled_rows:
        pd.DataFrame(labeled_rows).to_csv(output_path, index=False)

    clear()
    print(c("="*60, "green"))
    print(c("  Labeling Complete!", "bold"))
    print(c("="*60, "green"))
    print(f"\n  Rows labeled : {c(str(len(labeled_rows) - len(existing_labels)), 'green')}")
    print(f"  Rows skipped : {c(str(skipped), 'yellow')}")
    print(f"  Total saved  : {c(str(len(labeled_rows)), 'green')}")
    print(f"  Output file  : {c(output_path, 'yellow')}")
    print()

    # show score distribution
    if labeled_rows:
        scores = [r["human_score"] for r in labeled_rows]
        dist   = {"0-3": 0, "4-5": 0, "6-7": 0, "8-10": 0}
        for s in scores:
            if s <= 3:   dist["0-3"] += 1
            elif s <= 5: dist["4-5"] += 1
            elif s <= 7: dist["6-7"] += 1
            else:        dist["8-10"] += 1

        print(c("  Score Distribution:", "bold"))
        for range_label, count in dist.items():
            bar = "█" * count
            print(f"    {range_label:>5} | {c(bar, 'green')} {count}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual ground truth labeling tool for LLMDataBench")
    parser.add_argument("csv",           help="Path to CSV file")
    parser.add_argument("--rows",        default=None, type=int, help="Number of rows to label")
    parser.add_argument("--output",      default="output/ground_truth.csv", help="Output path for labels")
    parser.add_argument("--resume",      action="store_true", help="Resume from existing labels file")
    args = parser.parse_args()

    run_labeler(
        csv_path    = args.csv,
        rows        = args.rows,
        output_path = args.output,
        resume      = args.resume,
    )