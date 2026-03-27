"""
medquad_converter.py — Converts MedQuAD labeled dataset to LLMDataBench ground truth format

Usage:
    python3 medquad_converter.py \
        --qrels MedQuAD-CSVs/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt \
        --answers MedQuAD-CSVs/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-2479-Answers-retrieved-from-MedQuAD.csv \
        --output output/ground_truth_medquad.csv \
        --rows 200
"""

import os
import re
import argparse
import pandas as pd


# Map MedQuAD 1-4 labels to LLMDataBench 0-10 scores
LABEL_MAP = {
    "1-Incorrect":   2,   # Wrong answer → 0-2
    "2-Related":     4,   # Related but not direct → 3-5
    "3-Incomplete":  6,   # Partially answers → 6-7
    "4-Excellent":   9,   # Fully answers → 8-10
}

# Map labels to human-readable flags
FLAG_MAP = {
    "1-Incorrect":   "Answer is incorrect or irrelevant to the question",
    "2-Related":     "Answer is related but does not directly address the question",
    "3-Incomplete":  "Answer partially addresses the question but is missing key information",
    "4-Excellent":   "No issues found",
}

# Map labels to reasoning
REASONING_MAP = {
    "1-Incorrect":   "Expert judged this answer as incorrect — does not answer the medical question",
    "2-Related":     "Expert judged this answer as related — touches the topic but misses the question",
    "3-Incomplete":  "Expert judged this answer as incomplete — partially correct but lacks detail",
    "4-Excellent":   "Expert judged this answer as excellent — fully and accurately answers the question",
}


def parse_qrels(qrels_path: str) -> pd.DataFrame:
    """Parse the qrels txt file into a dataframe."""
    records = []
    with open(qrels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                question_id = parts[0]
                label       = parts[1]
                answer_id   = parts[2]
                records.append({
                    "question_id": question_id,
                    "label":       label,
                    "answer_id":   answer_id
                })

    return pd.DataFrame(records)


def parse_answers(answers_path: str) -> pd.DataFrame:
    """Parse the answers CSV into a dataframe."""
    df = pd.read_csv(answers_path)

    # extract question and answer from the combined text field
    questions  = []
    answers    = []
    focus_list = []

    for _, row in df.iterrows():
        text = str(row.get("Answer", ""))

        # extract question
        q_match = re.search(r'Question:\s*(.*?)(?:\n|$)', text)
        question = q_match.group(1).strip() if q_match else ""

        # extract focus (medical topic)
        f_match = re.search(r'Focus:\s*(.*?)(?:\n|$)', text)
        focus = f_match.group(1).strip() if f_match else ""

        # extract answer — everything after "Answer:" or after the question block
        a_match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
        if a_match:
            answer = a_match.group(1).strip()[:1000]  # cap at 1000 chars
        else:
            # fallback — take text after first two lines
            lines  = text.split("\n")
            answer = " ".join(lines[2:]).strip()[:1000]

        questions.append(question)
        answers.append(answer)
        focus_list.append(focus)

    df["question_text"] = questions
    df["answer_text"]   = answers
    df["focus"]         = focus_list

    return df


def convert(qrels_path: str, answers_path: str, output_path: str, rows: int = None):
    print("Loading qrels...")
    qrels = parse_qrels(qrels_path)
    print(f"  Loaded {len(qrels)} labeled pairs")
    print(f"  Label distribution:\n{qrels['label'].value_counts().to_string()}\n")

    print("Loading answers...")
    answers = parse_answers(answers_path)
    print(f"  Loaded {len(answers)} answers\n")

    # merge on answer_id
    merged = qrels.merge(answers, left_on="answer_id", right_on="AnswerID", how="inner")
    print(f"Merged {len(merged)} labeled QA pairs")

    # sample if requested
    if rows and rows < len(merged):
        # stratified sample — keep label distribution balanced
        merged = (
            merged.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(
                min(len(x), max(1, rows // merged["label"].nunique())),
                random_state=42
            ))
            .reset_index(drop=True)
        )
        print(f"Sampled {len(merged)} rows (stratified by label)\n")

    # map labels to scores, flags, reasoning
    merged["human_score"]     = merged["label"].map(LABEL_MAP)
    merged["human_flags"]     = merged["label"].map(FLAG_MAP)
    merged["human_reasoning"] = merged["label"].map(REASONING_MAP)
    merged["row_index"]       = merged.index

    # select and rename final columns to match LLMDataBench format
    output_df = merged[[
        "row_index",
        "question_id",
        "focus",
        "question_text",
        "answer_text",
        "label",
        "human_score",
        "human_flags",
        "human_reasoning"
    ]].rename(columns={
        "question_text": "Patient",   # maps to chatbot dataset Patient column
        "answer_text":   "Doctor",    # maps to chatbot dataset Doctor column
        "focus":         "Description"
    })

    # save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved {len(output_df)} rows to {output_path}")
    print(f"\nFinal label distribution:")
    print(merged["label"].value_counts().to_string())

    print(f"\nScore distribution:")
    score_dist = output_df["human_score"].value_counts().sort_index()
    for score, count in score_dist.items():
        bar = "█" * count
        print(f"  {score:>2}/10 | {bar} {count}")

    print(f"\nColumn mapping:")
    print(f"  question_text → Patient  (patient question)")
    print(f"  answer_text   → Doctor   (doctor response)")
    print(f"  focus         → Description (medical topic)")
    print(f"\nReady to use as ground truth in LLMDataBench!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MedQuAD to LLMDataBench ground truth format")
    parser.add_argument("--qrels",   required=True, help="Path to qrels txt file")
    parser.add_argument("--answers", required=True, help="Path to answers CSV file")
    parser.add_argument("--output",  default="output/ground_truth_medquad.csv", help="Output path")
    parser.add_argument("--rows",    default=200, type=int, help="Number of rows to sample (default 200)")
    args = parser.parse_args()

    convert(
        qrels_path   = args.qrels,
        answers_path = args.answers,
        output_path  = args.output,
        rows         = args.rows,
    )