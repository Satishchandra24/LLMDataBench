"""
accuracy.py — Compare model scores against ground truth labels

Usage:
    python3 accuracy.py \
        --ground-truth output/ground_truth_medquad.csv \
        --scores output/medquad/scores_llama3.2.csv \
        --model llama3.2

    # Compare all models at once
    python3 accuracy.py \
        --ground-truth output/ground_truth_medquad.csv \
        --scores-dir output/medquad
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from scipy import stats


def load_ground_truth(gt_path: str) -> pd.DataFrame:
    df = pd.read_csv(gt_path)
    required = ["human_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Ground truth missing column: {col}")
    return df


def load_model_scores(scores_path: str) -> pd.DataFrame:
    df = pd.read_csv(scores_path)
    if "score" not in df.columns:
        raise ValueError(f"Scores file missing 'score' column: {scores_path}")
    return df


def cohen_kappa(y1, y2, bins=None):
    """Compute Cohen's Kappa for ordinal agreement."""
    if bins:
        y1 = pd.cut(y1, bins=bins, labels=False)
        y2 = pd.cut(y2, bins=bins, labels=False)

    # build confusion matrix
    labels = sorted(set(list(y1) + list(y2)))
    n      = len(labels)
    matrix = np.zeros((n, n))

    for a, b in zip(y1, y2):
        if pd.isna(a) or pd.isna(b):
            continue
        i = labels.index(a)
        j = labels.index(b)
        matrix[i][j] += 1

    total    = matrix.sum()
    p_o      = np.diag(matrix).sum() / total
    p_e      = (matrix.sum(axis=1) * matrix.sum(axis=0)).sum() / (total ** 2)
    kappa    = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0

    return round(kappa, 3)


def compute_accuracy_metrics(gt: pd.DataFrame, scores: pd.DataFrame, model: str) -> dict:
    # align by row index if available, otherwise by position
    if "row_index" in scores.columns and "row_index" in gt.columns:
        merged = gt.merge(scores[["row_index", "score", "flags"]], on="row_index", how="inner")
    else:
        min_len = min(len(gt), len(scores))
        merged  = gt.head(min_len).copy()
        merged["score"] = scores["score"].head(min_len).values

    if len(merged) == 0:
        print(f"Warning: no matching rows between ground truth and {model} scores")
        return {}

    human  = merged["human_score"].astype(float)
    model_scores = merged["score"].astype(float)

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(human, model_scores)

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(human, model_scores)

    # Mean Absolute Error
    mae = round(float(np.abs(human - model_scores).mean()), 3)

    # Root Mean Squared Error
    rmse = round(float(np.sqrt(((human - model_scores) ** 2).mean())), 3)

    # Cohen's Kappa — bucket scores into 4 categories
    bins   = [0, 3, 5, 7, 10]
    human_binned = pd.cut(human, bins=bins, labels=[0, 1, 2, 3]).astype(float)
    model_binned = pd.cut(model_scores, bins=bins, labels=[0, 1, 2, 3]).astype(float)
    valid = ~(human_binned.isna() | model_binned.isna())
    kappa = cohen_kappa(
        human_binned[valid].tolist(),
        model_binned[valid].tolist()
    )

    # Score bias — does model score higher or lower than humans?
    bias = round(float((model_scores - human).mean()), 3)

    # Agreement rate — within 2 points
    agreement_2 = round(float((abs(human - model_scores) <= 2).mean()), 3)

    # Agreement rate — exact match
    agreement_exact = round(float((human == model_scores).mean()), 3)

    return {
        "model":           model,
        "n_rows":          len(merged),
        "pearson_r":       round(pearson_r, 3),
        "pearson_p":       round(pearson_p, 4),
        "spearman_r":      round(spearman_r, 3),
        "mae":             mae,
        "rmse":            rmse,
        "cohen_kappa":     kappa,
        "score_bias":      bias,
        "agreement_2pts":  agreement_2,
        "agreement_exact": agreement_exact,
        "human_avg":       round(float(human.mean()), 2),
        "model_avg":       round(float(model_scores.mean()), 2),
    }


def interpret_kappa(kappa: float) -> str:
    if kappa < 0:      return "Poor (worse than chance)"
    elif kappa < 0.2:  return "Slight"
    elif kappa < 0.4:  return "Fair"
    elif kappa < 0.6:  return "Moderate"
    elif kappa < 0.8:  return "Substantial"
    else:              return "Almost perfect"


def interpret_correlation(r: float) -> str:
    r = abs(r)
    if r < 0.2:   return "Negligible"
    elif r < 0.4: return "Weak"
    elif r < 0.6: return "Moderate"
    elif r < 0.8: return "Strong"
    else:         return "Very strong"


def run_accuracy_comparison(gt_path: str, scores_dir: str = None, scores_path: str = None, model: str = None, output_dir: str = None):
    print("\n" + "="*60)
    print("ACCURACY COMPARISON vs GROUND TRUTH")
    print("="*60)

    gt = load_ground_truth(gt_path)
    print(f"\nGround truth: {len(gt)} rows")
    print(f"Human score distribution:")
    print(gt["human_score"].value_counts().sort_index().to_string())

    # collect all model scores
    results = []

    if scores_dir:
        score_files = glob.glob(os.path.join(scores_dir, "scores_*.csv"))
        for sf in score_files:
            model_name = os.path.basename(sf).replace("scores_", "").replace(".csv", "")
            scores     = load_model_scores(sf)
            metrics    = compute_accuracy_metrics(gt, scores, model_name)
            if metrics:
                results.append(metrics)

    elif scores_path and model:
        scores  = load_model_scores(scores_path)
        metrics = compute_accuracy_metrics(gt, scores, model)
        if metrics:
            results.append(metrics)

    if not results:
        print("No results to compare.")
        return

    df = pd.DataFrame(results).sort_values("pearson_r", ascending=False)

    print("\n--- Correlation with Human Scores (higher = better) ---")
    for _, row in df.iterrows():
        print(f"\n  {row['model']}")
        print(f"    Pearson r:     {row['pearson_r']:>6} ({interpret_correlation(row['pearson_r'])})")
        print(f"    Spearman r:    {row['spearman_r']:>6}")
        print(f"    Cohen's Kappa: {row['cohen_kappa']:>6} ({interpret_kappa(row['cohen_kappa'])})")

    print("\n--- Error Metrics (lower = better) ---")
    for _, row in df.iterrows():
        bias_dir = "overestimates" if row['score_bias'] > 0 else "underestimates"
        print(f"\n  {row['model']}")
        print(f"    MAE:       {row['mae']:>6} (avg error per row)")
        print(f"    RMSE:      {row['rmse']:>6}")
        print(f"    Bias:      {row['score_bias']:>+6} ({bias_dir} by {abs(row['score_bias']):.2f} pts)")

    print("\n--- Agreement with Human Labels ---")
    for _, row in df.iterrows():
        print(f"\n  {row['model']}")
        print(f"    Within 2 pts:  {row['agreement_2pts']*100:.1f}% of rows")
        print(f"    Exact match:   {row['agreement_exact']*100:.1f}% of rows")
        print(f"    Human avg:     {row['human_avg']}/10")
        print(f"    Model avg:     {row['model_avg']}/10")

    # save results
    out_dir = output_dir or (scores_dir or os.path.dirname(scores_path or "."))
    out_path = os.path.join(out_dir, "accuracy_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nAccuracy comparison saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model scores against ground truth")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV")
    parser.add_argument("--scores-dir",   default=None,  help="Directory with scores_*.csv files")
    parser.add_argument("--scores",       default=None,  help="Path to single scores CSV")
    parser.add_argument("--model",        default=None,  help="Model name (when using --scores)")
    parser.add_argument("--output-dir",   default=None,  help="Where to save accuracy_comparison.csv")
    args = parser.parse_args()

    run_accuracy_comparison(
        gt_path     = args.ground_truth,
        scores_dir  = args.scores_dir,
        scores_path = args.scores,
        model       = args.model,
        output_dir  = args.output_dir,
    )