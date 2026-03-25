import os
import time
import pandas as pd
from utils.csv_loader import load_csv
from agents.schema_agent import run_schema_agent
from agents.meaning_agent import run_meaning_agent
from agents.evaluator_agent import run_evaluator_agent
from agents.report_agent import run_report_agent
from agents.orchestrator_agent import run_orchestrator_agent
import json

def flatten_evaluation(e: dict, index: int) -> dict:
    flags = e.get("flags", [])
    flat_flags = []
    for f in flags:
        if isinstance(f, dict):
            flat_flags.append("; ".join(f"{k}: {v}" for k, v in f.items()))
        else:
            flat_flags.append(str(f))
    return {
        "row_index": int(e.get("row_index", index)),
        "score":     int(e.get("score", 5)),
        "flags":     flat_flags,
        "reasoning": str(e.get("reasoning", ""))
    }


def save_scores_to_csv(df, evaluations: list, model: str, output_dir: str = "output"):
    scores_df = df.copy()
    eval_map = {e["row_index"]: e for e in evaluations}

    scores_df["score"]     = scores_df.index.map(lambda i: eval_map.get(i, {}).get("score", ""))
    scores_df["flags"]     = scores_df.index.map(lambda i: ", ".join(eval_map.get(i, {}).get("flags", [])))
    scores_df["reasoning"] = scores_df.index.map(lambda i: eval_map.get(i, {}).get("reasoning", ""))
    scores_df["model"]     = model

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"scores_{model}.csv")
    scores_df.to_csv(path, index=False)
    print(f"Scores saved to {path}")


def run_pipeline(csv_path: str, model: str = "llama3.2", concurrency: int = 5, rows: int = None, output_dir: str = "output", ctx: int = 8192):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"{'='*50}")

    print("Loading dataset...")
    df = load_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    if rows:
        df = df.sample(n=rows, random_state=42).reset_index(drop=True)
        print(f"Sampled {rows} rows for evaluation\n")

    # NEW — Agent 0
    print("Agent 0: Orchestrating pipeline...")
    agent_prompts = run_orchestrator_agent(df, model, ctx)


    print("Agent 1: Analyzing schema...")
    schema_summary = run_schema_agent(df, model, ctx,
        custom_prompt=agent_prompts.get("schema_prompt"))

    print("Agent 2: Understanding column meanings...")
    column_meanings = run_meaning_agent(df, schema_summary, model, ctx,
        custom_prompt=agent_prompts.get("meaning_prompt"))
    print(f"Defined scoring rules for {len(column_meanings)} columns\n")

    print("Agent 3: Evaluating rows...")
    t_start = time.time()
    raw_evaluations = run_evaluator_agent(df, column_meanings, model=model,
        concurrency=concurrency, ctx=ctx,
        evaluator_prompt=agent_prompts.get("evaluator_prompt"))
    latency = round(time.time() - t_start, 2)
    print(f"Evaluation took {latency}s ({round(latency/len(df), 2)}s per row)")

    evaluations = [flatten_evaluation(e, i) for i, e in enumerate(raw_evaluations)]

    print("Agent 4: Writing report...")
    report = run_report_agent(schema_summary, column_meanings, evaluations, model, ctx,
        custom_prompt=agent_prompts.get("report_prompt"))

    # save outputs
    report_path = os.path.join(output_dir, f"report_{model}.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # save orchestrator config for reproducibility
    config_path = os.path.join(output_dir, f"orchestrator_{model}.json")
    with open(config_path, "w") as f:
        json.dump(agent_prompts, f, indent=2)
    print(f"Orchestrator config saved to {config_path}")

    save_scores_to_csv(df, evaluations, model, output_dir)

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv",             help="Path to CSV file")
    parser.add_argument("--model",         default="llama3.2", help="Ollama model name")
    parser.add_argument("--concurrency",   default=5, type=int, help="Async concurrency limit")
    parser.add_argument("--rows", default=None, type=int, help="Number of rows to sample")
    parser.add_argument("--output-dir", default="output", help="Directory to save results")
    args = parser.parse_args()

    run_pipeline(args.csv, model=args.model, concurrency=args.concurrency, rows=args.rows, output_dir=args.output_dir)
