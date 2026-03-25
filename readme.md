# LLMDataBench

**A multi-agent framework for benchmarking local LLMs as domain-specific data quality judges.**

LLMDataBench evaluates how well different local LLMs perform as data quality judges — measuring consistency, flag accuracy, sycophancy, and speed across any CSV dataset. Built for privacy-sensitive domains where sending data to cloud APIs is not an option.

---

## Why LLMDataBench?

Enterprises increasingly need automated data quality evaluation — but sending sensitive data (healthcare records, financial data, legal documents) to cloud APIs like GPT-4 raises serious privacy and compliance concerns.

Local LLMs offer a privacy-preserving alternative, but there is no standard benchmark for how well different local models perform as data quality judges.

**LLMDataBench fills that gap.**

---

## Key Features

- **Multi-agent pipeline** — 5 specialized agents handle schema analysis, rule inference, row evaluation, and report generation
- **Orchestrator agent** — automatically detects dataset type (structured, conversational, mixed) and adapts evaluation strategy
- **Async evaluation** — all rows evaluated concurrently using asyncio for maximum speed
- **Model-agnostic** — swap any Ollama model via `--model` flag
- **Sycophancy detection** — automatically flags models that give uniformly high scores regardless of data quality
- **Cross-dataset benchmarking** — compare model performance across different dataset types
- **Privacy-preserving** — runs entirely locally, no data leaves your machine

---

## Architecture

```
Dataset (CSV)
      │
      ▼
Agent 0 — Orchestrator
      Detects dataset type and domain
      Generates custom evaluation context for all agents
      │
      ▼
Agent 1 — Schema Reader
      Reads columns, data types, null counts, sample values
      │
      ▼
Agent 2 — Meaning Analyst
      Infers column meanings and validation rules from data
      │
      ▼
Agent 3 — Row Evaluator (async)
      Evaluates each row concurrently against inferred rules
      │
      ▼
Agent 4 — Report Writer
      Aggregates scores, flags issues, writes markdown report
      │
      ▼
Output: scores_{model}.csv + report_{model}.md + meta_{model}.json
```

---

## Benchmark Results

Tested on a 30-row healthcare patient records dataset across 5 models on Apple M4 Mac Mini 16GB.

### Score Consistency

| Model | Avg Score | Std Dev | Score Range |
|-------|-----------|---------|-------------|
| deepseek-r1:8b | 10.00 | 0.00 | 10-10 |
| qwen3:8b | 10.00 | 0.00 | 10-10 |
| gemma3 | 10.00 | 0.00 | 10-10 |
| llama3.2 | 7.80 | 1.10 | 4-10 |
| mistral | 8.63 | 1.19 | 5-10 |

### Sycophancy Detection

| Model | Organization | Unique Scores | Verdict |
|-------|-------------|---------------|---------|
| deepseek-r1:8b | DeepSeek | 1 | ❌ SYCOPHANTIC |
| qwen3:8b | Alibaba | 1 | ❌ SYCOPHANTIC |
| gemma3 | Google | 1 | ❌ SYCOPHANTIC |
| llama3.2 | Meta | 4 | ✅ RELIABLE |
| mistral | Mistral AI | 3 | ✅ RELIABLE |

### Speed (30 rows)

| Model | Time/Row | Total Time |
|-------|----------|------------|
| gemma3 | 6.03s | 180s |
| llama3.2 | 7.04s | 211s |
| mistral | 7.80s | 234s |
| deepseek-r1:8b | 21.31s | 639s |
| qwen3:8b | 41.88s | 1256s |

**Key finding**: 3 out of 5 models — including both reasoning models (deepseek-r1, qwen3) — exhibited sycophantic bias, scoring all rows at maximum quality regardless of data issues. `llama3.2` offers the best balance of reliability, flag accuracy, and speed.

---

## Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running

### Install dependencies

```bash
git clone https://github.com/Satishchandra24/LLMDataBench.git
cd LLMDataBench
pip install -r requirements.txt
```

### Pull models

```bash
ollama pull llama3.2
ollama pull mistral
ollama pull qwen3:8b
ollama pull gemma3
ollama pull deepseek-r1:8b
```

---

## Usage

### Run evaluation on a single model

```bash
python3 main.py /path/to/dataset.csv --model llama3.2
```

### Run with options

```bash
python3 main.py /path/to/dataset.csv \
  --model llama3.2 \
  --rows 100 \
  --concurrency 5 \
  --output-dir output/my_run
```

### Run across multiple models

```bash
python3 main.py data.csv --model llama3.2 --output-dir output/healthcare
python3 main.py data.csv --model mistral --output-dir output/healthcare
python3 main.py data.csv --model qwen3:8b --output-dir output/healthcare
```

### Compare models

```bash
python3 compare.py output/healthcare
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `csv` | required | Path to CSV file |
| `--model` | llama3.2 | Ollama model name |
| `--rows` | None (all) | Number of rows to sample |
| `--concurrency` | 5 | Async concurrency limit |
| `--output-dir` | output/ | Directory to save results |

---

## Output Files

Each run produces:

| File | Description |
|------|-------------|
| `scores_{model}.csv` | Original data + score, flags, reasoning per row |
| `report_{model}.md` | Full markdown quality report |
| `meta_{model}.json` | Timing metadata and summary stats |
| `orchestrator_{model}.json` | Auto-generated evaluation config |
| `comparison.csv` | Cross-model comparison (from compare.py) |

---

## Project Structure

```
LLMDataBench/
├── main.py                    # Entry point
├── compare.py                 # Cross-model benchmark comparison
├── agents/
│   ├── orchestrator_agent.py  # Dataset type detection + prompt generation
│   ├── schema_agent.py        # Column structure analysis
│   ├── meaning_agent.py       # Rule inference from data
│   ├── evaluator_agent.py     # Async row evaluation
│   └── report_agent.py        # Report generation
├── utils/
│   └── csv_loader.py          # CSV loading utilities
├── requirements.txt
└── README.md
```

---

## Supported Dataset Types

LLMDataBench automatically adapts to different dataset types:

| Type | Example | Evaluation Focus |
|------|---------|-----------------|
| Structured | Patient records, financial data | Value validity, range, format, consistency |
| Conversational | Medical chatbots, support logs | Relevance, completeness, safety, coherence |
| Mixed | Survey data with free text | Combined structured + semantic evaluation |

---

## Requirements

```
langchain
langchain-ollama
langchain-community
langchain-core
langgraph
faiss-cpu
pypdf
beautifulsoup4
pandas
tabulate
```

---

## Hardware Recommendations

Tested on Apple M4 Mac Mini 16GB. Recommended specs:

| Model Size | Min RAM | Recommended |
|-----------|---------|-------------|
| 3B (llama3.2) | 8GB | 8GB+ |
| 7B (mistral) | 8GB | 16GB+ |
| 8B (qwen3, gemma3) | 10GB | 16GB+ |
| 14B (phi4) | 16GB | 32GB+ |

---

## Citation

If you use LLMDataBench in your research, please cite:

```bibtex
@misc{llmdatabench2026,
  title={LLMDataBench: Benchmarking Local LLMs as Domain-Specific Data Quality Judges},
  author={Satishchandra},
  year={2026},
  url={https://github.com/Satishchandra24/LLMDataBench}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

*Built with ❤️ for privacy-preserving AI evaluation*