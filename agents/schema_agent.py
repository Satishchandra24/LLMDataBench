from langchain_ollama import ChatOllama
from utils.csv_loader import get_schema
import json


def run_schema_agent(df, model: str = "llama3.2", ctx: int = 8192, custom_prompt: str = None) -> str:
    llm    = ChatOllama(model=model, temperature=0, timeout=120, num_ctx=ctx)
    schema = get_schema(df)
    schema_str = json.dumps(schema, indent=2, default=str)

    prompt = custom_prompt or f"""You are a data analyst. Here is the schema of a dataset:
{schema_str}
Summarize what kind of dataset this is, what each column represents, and any data quality concerns."""

    # if custom prompt, inject schema into it
    if custom_prompt:
        prompt = custom_prompt + f"\n\nDataset schema:\n{schema_str}"

    response = llm.invoke(prompt)
    return response.content