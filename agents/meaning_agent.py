from langchain_ollama import ChatOllama
from utils.csv_loader import get_sample
import json
import re
import pandas as pd
import warnings
#llm = ChatOllama(model="llama3.2", temperature=0)


def safe_int(val) -> int:
    try:
        if pd.isna(val):
            return 0
        return int(val)
    except:
        return 0

def compute_column_stats(df) -> dict:
    def safe_int(val) -> int:
        try:
            import math
            if math.isnan(float(val)):
                return 0
        except:
            pass
        try:
            return int(val)
        except:
            return 0

    stats = {}
    for col in df.columns:
        col_data = df[col].dropna()

        samples = [str(v)[:150] for v in col_data.unique().tolist()[:5]]

        col_stats = {
            "unique_values": samples,
            "total_unique":  int(df[col].nunique()),
            "null_count":    int(df[col].isnull().sum()),
            "total_count":   int(len(df[col])),
            "avg_length":    safe_int(col_data.astype(str).str.len().mean()),
            "max_length":    safe_int(col_data.astype(str).str.len().max()),
        }

        # try numeric
        try:
            numeric = pd.to_numeric(col_data, errors="raise")
            col_stats["min"]         = float(numeric.min())
            col_stats["max"]         = float(numeric.max())
            col_stats["mean"]        = round(float(numeric.mean()), 2)
            col_stats["is_numeric"]  = True
            col_stats["has_decimal"] = bool((numeric % 1 != 0).any())
        except:
            col_stats["is_numeric"] = False

        # try date
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(col_data, errors="raise")
            col_stats["min_date"] = str(parsed.min().date())
            col_stats["max_date"] = str(parsed.max().date())
            col_stats["is_date"]  = True
        except:
            col_stats["is_date"] = False

        cardinality_ratio = col_stats["total_unique"] / max(col_stats["total_count"], 1)
        col_stats["is_high_cardinality"] = cardinality_ratio > 0.5

        stats[col] = col_stats
    return stats



def run_meaning_agent(df, schema_summary: str, model: str = "llama3.2", ctx: int = 4096, custom_prompt: str = None) -> dict:
    llm = ChatOllama(model=model, temperature=0, timeout=120, num_ctx=ctx)

    avg_col_length = df.astype(str).apply(lambda x: x.str.len().mean()).mean()
    sample_size    = 5 if avg_col_length > 100 else 20
    sample         = get_sample(df, n=sample_size)
    col_stats      = compute_column_stats(df)

    print(f"  Average column length: {avg_col_length:.0f} chars — using {sample_size} sample rows")

    # use orchestrator context as a preamble but enforce our own JSON format
    orchestrator_context = ""
    if custom_prompt:
        orchestrator_context = f"""Dataset context from orchestrator:
{custom_prompt}

"""

    prompt = f"""{orchestrator_context}You are a data analyst. Infer validation rules for each column using the data below.

Column statistics (computed from full dataset):
{json.dumps(col_stats, indent=2)}

Sample rows:
{sample}

CRITICAL RULES:
- High cardinality columns (is_high_cardinality: true): type=freetext, only invalid if null/empty
- Numeric columns (is_numeric: true): use actual min/max, only flag impossible values
- Categorical columns (few unique values): list exact observed values
- Date columns (is_date: true): only flag unparseable dates
- Conversational columns (long text, avg_length > 200): type=conversational, evaluate on relevance/completeness/safety

Return ONLY a raw JSON object starting with {{ and ending with }}.
Format:
{{
  "column_name": {{
    "meaning": "what this column represents",
    "type": "categorical | numeric | date | freetext | conversational",
    "valid_values": "for categorical only — exact observed set. null for all others.",
    "invalid_indicators": ["only genuinely invalid values"],
    "weight": 1
  }}
}}"""

    response = llm.invoke(prompt)
    content  = response.content.strip()
    content  = re.sub(r"```(?:json)?", "", content).strip("` \n")

    # handle both array and object formats
    # if mistral returns an array, convert it to a dict
    if content.lstrip().startswith("["):
        try:
            arr = json.loads(content)
            result = {}
            for item in arr:
                if isinstance(item, dict) and "column_name" in item:
                    col_name = item.pop("column_name")
                    result[col_name] = item
            return {k: v for k, v in result.items() if isinstance(v, dict)}
        except:
            # try extracting array with bracket counting
            bracket_count = 0
            start_idx = None
            end_idx   = None
            for i, char in enumerate(content):
                if char == "[":
                    if start_idx is None:
                        start_idx = i
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0 and start_idx is not None:
                        end_idx = i
                        break
            if start_idx is not None and end_idx is not None:
                try:
                    arr = json.loads(content[start_idx:end_idx+1])
                    result = {}
                    for item in arr:
                        if isinstance(item, dict) and "column_name" in item:
                            col_name = item.pop("column_name")
                            result[col_name] = item
                    return {k: v for k, v in result.items() if isinstance(v, dict)}
                except Exception as e:
                    print(f"Warning: Could not parse array format: {e}")

    # existing brace-counting extractor for object format
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
            return {k: v for k, v in result.items() if isinstance(v, dict)}
        except:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                result = json.loads(json_str)
                return {k: v for k, v in result.items() if isinstance(v, dict)}
            except Exception as e:
                print(f"Warning: Could not parse meaning agent output: {e}")

    return {}