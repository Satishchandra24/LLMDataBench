from langchain_ollama import ChatOllama
import json
import re
import pandas as pd

def run_orchestrator_agent(df, model: str = "llama3.2", ctx: int = 4096) -> dict:
    """
    Analyzes the dataset and generates custom prompts for all downstream agents.
    Returns a dict of prompts keyed by agent name.
    """
    llm = ChatOllama(model=model, temperature=0, timeout=120, num_ctx=ctx)

    # compute basic dataset fingerprint
    avg_length     = df.astype(str).apply(lambda x: x.str.len().mean()).mean()
    num_cols       = len(df.columns)
    col_names      = df.columns.tolist()
    sample         = df.head(3).to_markdown(index=False)
    dtypes         = {col: str(df[col].dtype) for col in df.columns}
    null_counts    = {col: int(df[col].isnull().sum()) for col in df.columns}

    fingerprint = {
        "num_rows":        len(df),
        "num_columns":     num_cols,
        "column_names":    col_names,
        "dtypes":          dtypes,
        "null_counts":     null_counts,
        "avg_value_length": round(avg_length, 1),
        "sample_rows":     sample
    }

    prompt = f"""You are an expert data scientist and prompt engineer.
Analyze this dataset and generate detailed evaluation prompts for 4 downstream AI agents.

Dataset fingerprint:
{json.dumps(fingerprint, indent=2)}

Generate prompts that are specific, detailed, and include exact output format instructions.

Return ONLY a raw JSON object starting with {{ and ending with }}:
{{
  "dataset_type": "conversational | structured | mixed",
  "domain": "detected domain e.g. healthcare, finance, legal",
  "evaluation_focus": ["key criteria 1", "key criteria 2", "key criteria 3"],

  "schema_prompt": "A detailed prompt asking the agent to analyze column structure. Must ask for: dataset type, column descriptions, data quality concerns. End with: Return a structured markdown summary.",

  "meaning_prompt": "A detailed prompt for inferring column meanings and validation rules. Must include the exact JSON format to return. Return ONLY a raw JSON object starting with open-brace and ending with close-brace. Format: column_name with meaning, type (categorical|numeric|date|freetext|conversational), valid_values, invalid_indicators list, and weight 1-5.",

  "evaluator_prompt": "A detailed prompt for evaluating a single row of this dataset. The row data will be inserted at the placeholder [ROW_DATA]. Evaluate based on the dataset domain and type. Only flag confirmed violations. Score starts at 10, subtract 2 per violation. Return ONLY this JSON: score 0-10, flags list of specific issues, reasoning string.",

  "report_prompt": "A detailed prompt for writing a quality report. Summary data will be inserted at [SUMMARY_DATA]. Include: executive summary with numbers, score distribution, flagged rows table, column insights, specific recommendations."
}}

IMPORTANT: Do NOT use curly braces inside string values — use plain English descriptions of the format instead."""

    response = llm.invoke(prompt)
    content  = response.content.strip()
    content  = re.sub(r"```(?:json)?", "", content).strip("` \n")

    # brace counting extractor
    brace_count = 0
    start_idx   = None
    end_idx     = None

    for i, char in enumerate(content):
        if char == "{":
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                end_idx = i
                break

    if start_idx is not None and end_idx is not None:
        json_str = content[start_idx:end_idx + 1]
        try:
            result = json.loads(json_str)
            print(f"  Dataset type: {result.get('dataset_type')}")
            print(f"  Domain: {result.get('domain')}")
            print(f"  Evaluation focus: {result.get('evaluation_focus')}")
            return result
        except:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                return json.loads(json_str)
            except Exception as e:
                print(f"Warning: Could not parse orchestrator output: {e}")

    return {}