import pandas as pd

def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def get_sample(df: pd.DataFrame, n: int = 30) -> str:
    sample = df.head(n)
    return sample.to_markdown(index=False)

def get_schema(df: pd.DataFrame) -> dict:
    schema = {}
    for col in df.columns:
        schema[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "sample_values": df[col].dropna().head(5).tolist()
        }
    return schema