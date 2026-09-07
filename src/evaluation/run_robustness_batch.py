import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.generate_robustness_rewrite import (
    generate_rewrite,
    MODEL_NAME,
    TEMPERATURE,
    MAX_ATTEMPTS,
)
from src.evaluation.semantic_screen import (
    SemanticScreener,
    SIMILARITY_THRESHOLD,
    empty_semantic_rubric,
    evaluate_semantic_rubric,
)


INPUT_PATH = Path("results/fusion/robustness_sample_100.csv")
OUTPUT_DIR = Path("results/fusion/robustness_generated")
OUTPUT_PATH = OUTPUT_DIR / "robustness_transformations.jsonl"

TRANSFORMATIONS = [
    "professional",
    "cue_reduced",
    "combined",
]


def load_existing_records():
    records = {}

    if not OUTPUT_PATH.exists():
        return records

    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            key = (
                record["sample_id"],
                record["transformation_type"],
            )
            records[key] = record

    return records


def append_record(record):
    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    screener = SemanticScreener()
    existing = load_existing_records()

    total = len(df) * len(TRANSFORMATIONS)
    completed = len(existing)

    print(f"INPUT_ROWS: {len(df)}")
    print(f"TRANSFORMATIONS: {len(TRANSFORMATIONS)}")
    print(f"TARGET_RECORDS: {total}")
    print(f"EXISTING_RECORDS: {completed}")
    print(f"OUTPUT: {OUTPUT_PATH}")

    for _, row in df.iterrows():
        sample_id = str(row["sample_id"])
        original_text = str(row["text"])
        source = str(row["source"])
        confidence_band = str(row["confidence_band"])

        for transformation_type in TRANSFORMATIONS:
            key = (sample_id, transformation_type)

            if key in existing:
                print(
                    f"SKIP {sample_id} {transformation_type} "
                    f"(already recorded)"
                )
                continue

            print(
                f"GENERATE {sample_id} {transformation_type}"
            )

            started_at = datetime.now(timezone.utc).isoformat()

            result = generate_rewrite(
                original_text=original_text,
                transformation_type=transformation_type,
            )

            record = {
                "sample_id": sample_id,
                "source": source,
                "confidence_band": confidence_band,
                "transformation_type": transformation_type,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "max_attempts": MAX_ATTEMPTS,
                "started_at_utc": started_at,
                "generation_status": result["status"],
                "generation_reason": result["reason"],
                "generation_records": result.get(
                    "generation_records", []
                ),
                "placeholder_validation": result.get(
                    "validation"
                ),
                "cosine_similarity": None,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "similarity_status": "NOT_ASSESSED",
                "semantic_rubric": empty_semantic_rubric(),
                "semantic_rubric_status": "NOT_ASSESSED",
                "semantic_rubric_decision": evaluate_semantic_rubric(
                    empty_semantic_rubric()
                ),
                "rewritten_text": None,
            }

            if result["status"] == "pass":
                rewritten_text = result["rewritten_text"]

                screening = screener.screen(
                    original_text,
                    rewritten_text,
                )

                record["cosine_similarity"] = screening[
                    "cosine_similarity"
                ]
                record["similarity_status"] = (
                    "PASS_SCREENING"
                    if screening["similarity_passed"]
                    else "REJECT_SIMILARITY"
                )
                record["rewritten_text"] = rewritten_text

            if result["reason"] == "rate_limit_exceeded":
                print(
                    f"RATE_LIMIT {sample_id} {transformation_type} "
                    f"- not persisted; retry on next run"
                )
                return

            append_record(record)
            existing[key] = record

            completed += 1

            print(
                f"RECORDED {sample_id} {transformation_type} "
                f"({completed}/{total}) "
                f"status={record['similarity_status']}"
            )


if __name__ == "__main__":
    main()
