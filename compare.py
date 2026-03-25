import os
import json
import glob
import pandas as pd


def load_model_results(output_dir: str = "output") -> dict:
    """Load all model score CSVs and metadata into a dict keyed by model name."""
    results = {}

    for meta_file in glob.glob(os.path.join(output_dir, "meta_*.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        model = meta["model"]

        scores_file = os.path.join(output_dir, f"scores_{model}.csv")
        if not os.path.exists(scores_file):
            continue

        df = pd.read_csv(scores_file)
        results[model] = {
            "meta":   meta,
            "scores": df["score"].tolist(),
            "flags":  df["flags"].tolist(),
        }

    return results


def score_consistency(scores: list) -> dict:
    """How consistent is this model? Lower std = more consistent."""
    import statistics
    return {
        "mean":   round(statistics.mean(scores), 2),
        "stdev":  round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
        "min":    min(scores),
        "max":    max(scores),
    }


def flag_accuracy(flags: list) -> dict:
    # convert all to string, handle NaN
    flags = [str(f) if f == f else "" for f in flags]  # f == f is False for NaN

    total        = len(flags)
    empty_flags  = sum(1 for f in flags if not f or f.strip() == "")
    parse_errors = sum(1 for f in flags if "parse error" in f.lower())
    valid_flags  = total - empty_flags - parse_errors

    return {
        "total_rows":   total,
        "rows_flagged": sum(1 for f in flags if f and f.strip()),
        "parse_errors": parse_errors,
        "valid_flags":  valid_flags,
        "flag_rate":    round(sum(1 for f in flags if f and f.strip()) / total, 2),
    }


def detect_sycophancy(scores: list) -> dict:
    """A model is sycophantic if it gives the same high score to everything."""
    unique_scores  = len(set(scores))
    avg            = sum(scores) / len(scores)
    is_sycophantic = unique_scores <= 2 and avg >= 8.0

    return {
        "unique_score_values": unique_scores,
        "is_sycophantic":      is_sycophantic,
        "verdict":             "SYCOPHANTIC" if is_sycophantic else "RELIABLE"
    }


def compare_models(output_dir: str = "output"):
    results = load_model_results(output_dir)

    if not results:
        print("No model results found in output/. Run main.py with different --model flags first.")
        return

    print("\n" + "="*60)
    print("MODEL COMPARISON REPORT")
    print("="*60)

    rows = []
    for model, data in results.items():
        consistency  = score_consistency(data["scores"])
        accuracy     = flag_accuracy(data["flags"])
        meta         = data["meta"]
        sycophancy   = detect_sycophancy(data["scores"])

        rows.append({
            "model":          model,
            "avg_score":      consistency["mean"],
            "score_stdev":    consistency["stdev"],
            "score_range":    f"{consistency['min']}-{consistency['max']}",
            "flag_rate":      accuracy["flag_rate"],
            "parse_errors":   accuracy["parse_errors"],
            "valid_flags":    accuracy["valid_flags"],
            "time_per_row_s": meta.get("time_per_row", "N/A"),
            "total_time_s":   meta.get("total_time_s", "N/A"),
            "sycophancy":     sycophancy["verdict"],
            "unique_scores":  sycophancy["unique_score_values"],
        })

    df = pd.DataFrame(rows).sort_values("score_stdev")

    print("\n--- Score Consistency (lower stdev = more consistent) ---")
    print(df[["model", "avg_score", "score_stdev", "score_range"]].to_string(index=False))

    print("\n--- Flag Accuracy (fewer parse errors = better) ---")
    print(df[["model", "flag_rate", "parse_errors", "valid_flags"]].to_string(index=False))

    print("\n--- Speed ---")
    print(df[["model", "time_per_row_s", "total_time_s"]].to_string(index=False))

    # ← add this block
    print("\n--- Sycophancy Detection ---")
    for _, row in df.iterrows():
        print(f"{row['model']:>20} | unique scores: {row['unique_scores']:>3} | {row['sycophancy']}")

    # save comparison
    out_path = os.path.join(output_dir, "comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull comparison saved to {out_path}")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "output"
    compare_models(folder)