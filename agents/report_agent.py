from langchain_ollama import ChatOllama
import json


def run_report_agent(schema_summary, column_meanings, evaluations, model: str = "llama3.2", ctx: int = 8192, custom_prompt: str = None) -> str:
    llm = ChatOllama(model=model, temperature=0, timeout=120, num_ctx=ctx)

    if not evaluations:
        return "Error: no evaluations to report on."

    scores       = [e["score"] for e in evaluations]
    avg_score    = round(sum(scores) / len(scores), 2)
    low_scoring  = [e for e in evaluations if e["score"] < 5]
    high_scoring = [e for e in evaluations if e["score"] >= 8]

    flagged_rows = [
        {
            "row":       e["row_index"] + 1,
            "score":     e["score"],
            "flags":     e["flags"],
            "reasoning": e["reasoning"]
        }
        for e in evaluations
        if e["flags"] and e["flags"] != ["No issues found"]
    ]

    distribution = {"0-3": 0, "4-5": 0, "6-7": 0, "8-10": 0}
    for s in scores:
        if s <= 3:   distribution["0-3"] += 1
        elif s <= 5: distribution["4-5"] += 1
        elif s <= 7: distribution["6-7"] += 1
        else:        distribution["8-10"] += 1

    summary_data = json.dumps({
        "total_rows":    len(evaluations),
        "avg_score":     avg_score,
        "high_quality":  len(high_scoring),
        "low_quality":   len(low_scoring),
        "distribution":  distribution,
        "flagged_rows":  flagged_rows,
        "column_rules":  {k: v.get("scoring_rule") for k, v in column_meanings.items()}
    }, indent=2)

    if custom_prompt:
        prompt = custom_prompt.replace("[SUMMARY_DATA]", summary_data)
    else:
        prompt = f"""You are a data quality reporter. Write a specific, data-driven markdown report.

Dataset type:
{schema_summary[:500]}

Score summary:
- Total rows evaluated: {len(evaluations)}
- Average score: {avg_score}/10
- High quality rows (8-10): {len(high_scoring)}
- Low quality rows (0-4): {len(low_scoring)}
- Score distribution: {json.dumps(distribution)}

Flagged rows with details:
{json.dumps(flagged_rows, indent=2)}

Column scoring rules used:
{json.dumps({k: v.get("scoring_rule") for k, v in column_meanings.items()}, indent=2)}

Write a report with these sections:
1. Executive Summary (2-3 sentences with actual numbers)
2. Score Distribution (reference the actual distribution)
3. Flagged Rows (list each row with row number, score, and specific issues)
4. Column-level Insights (which columns caused the most flags)
5. Recommendations (specific to the actual issues found)

Be specific — reference actual row numbers, scores, and flag reasons."""

    response = llm.invoke(prompt)
    return response.content