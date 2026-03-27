from langchain_ollama import ChatOllama
import json
import re
import asyncio


def parse_json_response(content: str, expected_type="object"):
    content = content.strip()
    content = re.sub(r"```(?:json)?", "", content).strip("` \n")
    pattern = r'\[.*\]' if expected_type == "array" else r'\{.*\}'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return json.loads(match.group())
    return None


def build_evaluation_table(row: dict, column_meanings: dict) -> str:
    lines = ["column | type | invalid_if | actual_value"]
    lines.append("-------|------|------------|------------")
    for col, rules in column_meanings.items():
        actual = str(row.get(col, "NULL"))
        actual = actual[:200] if len(actual) > 200 else actual

        # handle case where rules is a string instead of dict
        if isinstance(rules, str):
            col_type = "unknown"
            invalid  = rules[:100]  # use the string as the rule description
        elif isinstance(rules, dict):
            col_type = rules.get("type", "unknown")
            invalid  = "; ".join(rules.get("invalid_indicators", [])) or "none"
        else:
            col_type = "unknown"
            invalid  = "none"

        lines.append(f"{col} | {col_type} | {invalid} | {actual}")
    return "\n".join(lines)

def run_evaluator_agent(df, column_meanings: dict, model: str = "llama3.2", concurrency: int = 5, ctx: int = 4096, evaluator_prompt: str = None) -> list:
    rows  = df.to_dict(orient="records")
    total = len(rows)
    print(f"Evaluating {total} rows with concurrency={concurrency}...")

    results = asyncio.run(evaluate_all_async(rows, column_meanings, concurrency, model, ctx, evaluator_prompt))

    print(f"\nDone — {total} rows evaluated.")
    return results


async def evaluate_all_async(rows: list, column_meanings: dict, concurrency: int, model: str, ctx: int, evaluator_prompt: str = None) -> list:
    semaphore = asyncio.Semaphore(concurrency)

    async def sem_evaluate(row, index):
        async with semaphore:
            return await evaluate_row_async(row, column_meanings, index, model, ctx, evaluator_prompt)

    tasks     = [sem_evaluate(row, i) for i, row in enumerate(rows)]
    results   = []
    total     = len(tasks)
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        print(f"  Evaluated {completed}/{total} rows...", end="\r")
        results.append(result)

    results.sort(key=lambda x: x["row_index"])
    return results


async def evaluate_row_async(row: dict, column_meanings: dict, row_index: int, model: str, ctx: int = 4096, evaluator_prompt: str = None) -> dict:
    llm   = ChatOllama(model=model, temperature=0, timeout=60, num_ctx=ctx)
    table = build_evaluation_table(row, column_meanings)

    if evaluator_prompt:
        # use orchestrator context as preamble, but keep our structured format
        prompt = f"""Dataset context:
{evaluator_prompt.replace("[ROW_DATA]", "")}

Now evaluate this specific row:
{table}

Only flag confirmed violations. Score starts at 10, subtract 2 per violation.
Return ONLY this JSON:
{{"score": 0-10, "flags": ["specific issue"], "reasoning": "brief"}}"""
    else:
        prompt = f"""You are a strict data validator. Check each column's actual value against its rule.

{table}

Only flag if actual value CLEARLY violates the invalid_if condition.
Do NOT invent violations.

Return ONLY this JSON:
{{"score": 0-10, "flags": ["column_name: specific violation"], "reasoning": "brief"}}

Score starts at 10. Subtract 2 for each confirmed violation."""

    try:
        response = await asyncio.to_thread(llm.invoke, prompt)
        result   = parse_json_response(response.content, expected_type="object")
        if result:
            result["row_index"] = row_index
            return result
    except Exception as e:
        print(f"\n  Error on row {row_index}: {e}")

    return {
        "row_index": row_index,
        "score":     5,
        "flags":     ["parse error"],
        "reasoning": "could not parse response"
    }