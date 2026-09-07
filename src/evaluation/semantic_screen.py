from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85

RUBRIC_FIELDS = [
    "objective_retained",
    "requested_action_retained",
    "entities_claims_retained",
    "financial_intent_retained",
    "credential_intent_retained",
    "authority_impersonation_retained",
    "important_behavior_retained",
    "new_behavior_introduced",
]


class SemanticScreener:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def cosine_similarity(self, original: str, rewritten: str) -> float:
        embeddings = self.model.encode(
            [original, rewritten],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return float(np.dot(embeddings[0], embeddings[1]))

    def screen(
        self,
        original: str,
        rewritten: str,
    ) -> Dict[str, object]:
        similarity = self.cosine_similarity(original, rewritten)

        return {
            "cosine_similarity": similarity,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "similarity_passed": similarity >= SIMILARITY_THRESHOLD,
            "screening_status": (
                "PASS_SCREENING"
                if similarity >= SIMILARITY_THRESHOLD
                else "REVIEW_REQUIRED"
            ),
            "semantic_rubric_status": "NOT_ASSESSED",
        }


def empty_semantic_rubric() -> Dict[str, str]:
    return {field: "NOT_ASSESSED" for field in RUBRIC_FIELDS}


def evaluate_semantic_rubric(rubric: Dict[str, str]) -> Dict[str, object]:
    allowed_values = {"YES", "NO", "N/A", "NOT_ASSESSED"}

    unknown_fields = set(rubric) - set(RUBRIC_FIELDS)
    missing_fields = set(RUBRIC_FIELDS) - set(rubric)

    invalid_values = {
        field: value
        for field, value in rubric.items()
        if value not in allowed_values
    }

    if unknown_fields or missing_fields or invalid_values:
        return {
            "valid": False,
            "status": "INVALID_RUBRIC",
            "rejection_reason": "rubric_schema_invalid",
        }

    if any(value == "NOT_ASSESSED" for value in rubric.values()):
        return {
            "valid": True,
            "status": "NOT_ASSESSED",
            "rejection_reason": None,
        }

    rejection_fields = [
        field
        for field in RUBRIC_FIELDS
        if field != "new_behavior_introduced"
        and rubric[field] == "NO"
    ]

    if rubric["new_behavior_introduced"] == "YES":
        rejection_fields.append("new_behavior_introduced")

    if rejection_fields:
        return {
            "valid": True,
            "status": "REJECT",
            "rejection_reason": "semantic_preservation_failed",
            "rejection_fields": rejection_fields,
        }

    return {
        "valid": True,
        "status": "ACCEPT",
        "rejection_reason": None,
        "rejection_fields": [],
    }


if __name__ == "__main__":
    screener = SemanticScreener()
    original = (
        "Your account requires immediate verification. "
        "Click the link below to confirm your credentials."
    )
    rewritten = (
        "Please verify your account by using the link below "
        "to confirm your login credentials."
    )

    result = screener.screen(original, rewritten)
    result["rubric"] = empty_semantic_rubric()
    print(result)
