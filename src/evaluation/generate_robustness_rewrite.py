import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


INPUT_PATH = "results/fusion/robustness_sample_100.csv"
MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.0
MAX_ATTEMPTS = 2


def get_client():
    load_dotenv(".env")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def protect_content(text):
    protected = {}

    # Protect URLs in ordinary and tokenized/spaced forms.
    url_pattern = re.compile(
        r"(?:"
        r"https?\s*:\s*/\s*/\s*"
        r"(?:[A-Za-z0-9-]+\s*\.\s*)+"
        r"[A-Za-z]{2,}"
        r"(?:\s*/\s*[^\s<>\[\]]*)?"
        r"|"
        r"www\s*\.\s*"
        r"(?:[A-Za-z0-9-]+\s*\.\s*)+"
        r"[A-Za-z]{2,}"
        r"(?:\s*/\s*[^\s<>\[\]]*)?"
        r"|"
        r"https?://[^\s<>\[\]]+"
        r")",
        flags=re.IGNORECASE,
    )

    def replace_url(match):
        key = f"__URL_{len(protected) + 1:03d}__"
        protected[key] = match.group(0)
        return key

    protected_text = url_pattern.sub(replace_url, text)

    # Protect email addresses.
    email_pattern = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    def replace_email(match):
        key = f"__EMAIL_{len(protected) + 1:03d}__"
        protected[key] = match.group(0)
        return key

    protected_text = email_pattern.sub(replace_email, protected_text)

    # Protect percentages.
    percentage_pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*%"
    )

    def replace_percentage(match):
        key = f"__PCT_{len(protected) + 1:03d}__"
        protected[key] = match.group(0)
        return key

    protected_text = percentage_pattern.sub(
        replace_percentage,
        protected_text,
    )

    return protected_text, protected


def restore_content(protected_text, protected):
    restored = protected_text

    for placeholder, original_value in protected.items():
        restored = restored.replace(
            placeholder,
            original_value,
        )

    return restored


def validate_placeholders(original_text, rewritten_text):
    placeholder_pattern = re.compile(
        r"__(?:URL|EMAIL|PCT)_\d{3}__"
    )

    original_placeholders = placeholder_pattern.findall(original_text)
    rewritten_placeholders = placeholder_pattern.findall(rewritten_text)

    sequence_match = (
        original_placeholders == rewritten_placeholders
    )

    set_match = (
        set(original_placeholders)
        == set(rewritten_placeholders)
    )

    counts_match = all(
        rewritten_placeholders.count(key) == 1
        for key in set(original_placeholders)
    )

    return {
        "original_count": len(original_placeholders),
        "rewritten_count": len(rewritten_placeholders),
        "sequence_match": sequence_match,
        "set_match": set_match,
        "counts_match": counts_match,
        "passed": set_match and counts_match,
        "missing": sorted(
            set(original_placeholders)
            - set(rewritten_placeholders)
        ),
        "unexpected": sorted(
            set(rewritten_placeholders)
            - set(original_placeholders)
        ),
    }


SYSTEM_PROMPT = """
You are creating a controlled robustness-test transformation of an email
for defensive machine-learning evaluation.

Rewrite the email so that it sounds more natural, polished, and professional.

CRITICAL:
The tokens beginning with __URL_, __EMAIL_, and __PCT_ are immutable
placeholders.

You MUST:
1. Keep every placeholder exactly unchanged.
2. Keep every placeholder exactly once.
3. Do not rename, split, merge, delete, or invent placeholders.
4. Preserve the underlying intent and requested actions.
5. Do not add new facts, claims, instructions, threats, prices, dates,
   organizations, URLs, or other details.
6. Do not remove existing substantive content.
7. Improve only grammar, readability, organization, and naturalness.
8. Do not summarize or shorten the message.
9. Return valid JSON only.

JSON format:
{
  "rewritten_text": "..."
}
"""


def generate_rewrite(
    original_text,
    transformation_type="professional",
):
    client = get_client()

    protected_text, protected = protect_content(
        original_text
    )

    transformation_instructions = {
        "professional": """Improve grammar, readability, organization, and naturalness. Keep the original tone and level of urgency unless required for grammatical clarity. Do not intentionally remove suspicious cues.""",
        "cue_reduced": """Improve grammar and naturalness while reducing obvious scam-style linguistic cues such as exaggerated urgency, awkward phrasing, excessive punctuation, or overtly threatening language. Preserve the underlying objective, requested action, entities, claims, and substantive behavioral details. Do not remove the malicious or suspicious intent itself.""",
        "combined": """First improve grammar, readability, organization, and naturalness, then reduce obvious scam-style linguistic cues such as exaggerated urgency, awkward phrasing, excessive punctuation, or overtly threatening language. Preserve the underlying objective, requested action, entities, claims, and substantive behavioral details. Do not remove the malicious or suspicious intent itself.""",
    }

    if transformation_type not in transformation_instructions:
        raise ValueError(f"Unsupported transformation_type: {transformation_type}")

    user_prompt = f"""
Transformation type: {transformation_type}

Transformation instructions:
{transformation_instructions[transformation_type]}

Rewrite this email according to the requested transformation.

The transformation MUST preserve:
- the underlying malicious or suspicious objective
- the requested action
- all important entities and claims
- all financial or credential-related intent
- all authority or impersonation context
- all important behavioral details

The transformation MUST NOT:
- introduce new behavior
- introduce new facts
- introduce new instructions
- introduce new URLs
- introduce new prices or dates
- remove substantive content
- summarize the original

Protected data:

{protected_text}

The placeholders are protected data. Treat them exactly like immutable
database tokens and copy them unchanged into the rewritten text.

Return JSON only.
"""

    generation_records = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        current_system_prompt = SYSTEM_PROMPT

        if attempt == 2:
            current_system_prompt += """
IMPORTANT RETRY INSTRUCTION:
The previous attempt failed validation.

Return exactly one complete JSON object and nothing else.
Do not use markdown fences.
Do not add commentary.
Do not stop before closing the JSON object.
Preserve every immutable placeholder exactly once.
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": current_system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=4096,
            )
        except Exception as exc:
            error_text = str(exc)

            record = {
                "attempt": attempt,
                "api_error": type(exc).__name__,
                "api_error_message": error_text[:1000],
            }
            generation_records.append(record)

            if "413" in error_text:
                return {
                    "status": "discard",
                    "reason": "request_too_large",
                    "protected_text": protected_text,
                    "protected": protected,
                    "generation_records": generation_records,
                }

            if "429" in error_text:
                return {
                    "status": "discard",
                    "reason": "rate_limit_exceeded",
                    "protected_text": protected_text,
                    "protected": protected,
                    "generation_records": generation_records,
                }

            raise

        raw = (
            response.choices[0].message.content or ""
        ).strip()

        record = {
            "attempt": attempt,
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": getattr(
                response.usage,
                "prompt_tokens",
                None,
            ),
            "completion_tokens": getattr(
                response.usage,
                "completion_tokens",
                None,
            ),
            "raw_characters": len(raw),
        }

        try:
            result = json.loads(raw)
            record["json_valid"] = True
        except json.JSONDecodeError:
            record["json_valid"] = False
            generation_records.append(record)

            if attempt == MAX_ATTEMPTS:
                return {
                    "status": "discard",
                    "reason": "json_validation_failed",
                    "protected_text": protected_text,
                    "protected": protected,
                    "generation_records": generation_records,
                }

            continue

        rewritten_protected = (
            result.get("rewritten_text", "")
            or ""
        ).strip()

        if not rewritten_protected:
            record["json_valid"] = False
            generation_records.append(record)

            if attempt == MAX_ATTEMPTS:
                return {
                    "status": "discard",
                    "reason": "empty_rewrite",
                    "protected_text": protected_text,
                    "protected": protected,
                    "generation_records": generation_records,
                }

            continue

        validation = validate_placeholders(
            protected_text,
            rewritten_protected,
        )

        record["placeholder_passed"] = validation["passed"]
        record["sequence_match"] = validation["sequence_match"]

        generation_records.append(record)

        if not validation["passed"]:
            if attempt == MAX_ATTEMPTS:
                return {
                    "status": "discard",
                    "reason": "placeholder_validation_failed",
                    "protected_text": protected_text,
                    "protected": protected,
                    "generation_records": generation_records,
                    "validation": validation,
                }

            continue

        rewritten = restore_content(
            rewritten_protected,
            protected,
        )

        restoration_ok = (
            all(
                value in rewritten
                for value in protected.values()
            )
        )

        if not restoration_ok:
            if attempt == MAX_ATTEMPTS:
                return {
                    "status": "discard",
                    "reason": "restoration_validation_failed",
                    "protected_text": protected_text,
                    "protected": protected,
                    "generation_records": generation_records,
                    "validation": validation,
                }

            continue

        return {
            "status": "pass",
            "reason": "validated",
            "rewritten_text": rewritten,
            "rewritten_protected": rewritten_protected,
            "protected_text": protected_text,
            "protected": protected,
            "validation": validation,
            "generation_records": generation_records,
        }

    return {
        "status": "discard",
        "reason": "unexpected_generation_failure",
        "protected_text": protected_text,
        "protected": protected,
        "generation_records": generation_records,
    }


def main():
    df = pd.read_csv(INPUT_PATH)
    row = df.iloc[0]

    result = generate_rewrite(
        original_text=str(row["text"]),
        transformation_type="professional",
    )

    print("=== SINGLE-SAMPLE GENERATION TEST ===")
    print("Source:", row["source"])
    print("Original probability:", f"{row['text_probability']:.6f}")
    print("Status:", result["status"])
    print("Reason:", result["reason"])

    print()
    print("=== GENERATION RECORDS ===")

    for record in result["generation_records"]:
        print(record)

    if result["status"] == "discard":
        if "validation" in result:
            print()
            print("=== VALIDATION ===")
            print(result["validation"])
        return

    validation = result["validation"]

    print()
    print("=== PLACEHOLDER VALIDATION ===")
    print("Original placeholders:", validation["original_count"])
    print("Rewritten placeholders:", validation["rewritten_count"])
    print("Exact sequence match:", validation["sequence_match"])
    print("Placeholder set match:", validation["set_match"])
    print("Each placeholder exactly once:", validation["counts_match"])

    print()
    print("=== RESTORATION VALIDATION ===")
    print("All protected values restored successfully.")

    print()
    print("=== REWRITTEN TEXT ===")
    print(result["rewritten_text"][:5000])


if __name__ == "__main__":
    main()
