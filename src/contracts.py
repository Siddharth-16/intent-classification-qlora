from __future__ import annotations

import json
from pathlib import Path
from typing import Collection

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
RANDOM_SEED = 42
OOS_LABEL = "oos"

DEVELOPMENT_SPLIT = "validation"
FINAL_SPLIT = "test"

METRICS = (
    "overall_accuracy",
    "macro_f1_all_labels",
    "in_scope_accuracy",
    "oos_precision",
    "oos_recall",
    "oos_f1",
    "in_scope_false_rejection_rate",
    "valid_label_rate",
)


def normalize_generated_label(raw_output: str) -> str:
    return raw_output.strip()


def parse_generated_label(
    raw_output: str, allowed_labels: Collection[str]
) -> tuple[str | None, bool]:
    """Return the canonical label and whether the output is valid"""

    normalized = normalize_generated_label(raw_output)
    if normalized in allowed_labels:
        return normalized, True
    return None, False


def build_evaluation_contract(labels: list[str]) -> dict[str, object]:
    """Create the machine readable evaluation contract"""

    return {
        "model_id": MODEL_ID,
        "random_seed": RANDOM_SEED,
        "development_split": DEVELOPMENT_SPLIT,
        "final_split": FINAL_SPLIT,
        "label_count": len(labels),
        "oos_label": OOS_LABEL,
        "labels": labels,
        "output_contract": {
            "expected": "exactly one canonical label",
            "normalization": "surrounding whitespace only",
            "invalid_output_policy": (
                "count as incorrect and include in valid-label-rate calculation"
            ),
        },
        "primary_metric": "macro_f1_all_labels",
        "metrics": list(METRICS),
        "experiment_name_pattern": "{approach}__{split}__seed-{seed}",
        "test_policy": (
            "Run only after prompts, parsing, model configuration and "
            "checkpoint selection are frozen. Do not tune after viewing test results."
        ),
    }


def write_evaluation_contract(path: Path, labels: list[str]) -> None:
    """Persist the contract used by every later experiment"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_evaluation_contract(labels), indent=2) + "\n",
        encoding="utf-8",
    )

