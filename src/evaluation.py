from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from src.contracts import OOS_LABEL

INVALID_PREDICTION = "__invalid_prediction__"


def evaluate_predictions(
    references: Sequence[str],
    predictions: Sequence[str | None],
    labels: Sequence[str],
    oos_label: str = OOS_LABEL,
) -> dict[str, int | float]:
    """Compute the frozen metrics, counting invalid predictions as incorrect"""

    if not references:
        raise ValueError("Cannot evaluate an empty prediction set")

    if len(references) != len(predictions):
        raise ValueError(
            "Reference and prediction lengths differ: "
            f"{len(references)} != {len(predictions)}"
        )

    if len(labels) != len(set(labels)):
        raise ValueError("The label list contains duplicates")

    allowed_labels = set(labels)

    if oos_label not in allowed_labels:
        raise ValueError(f"OOS label {oos_label!r} is missing from the label list")

    unexpected_references = sorted(set(references) - allowed_labels)
    if unexpected_references:
        raise ValueError(
            f"References contain unknown labels: {unexpected_references}"
        )

    # LLM outputs that violate the label contract will become None,
    # mapping them to a sentinel so they count as incorrect
    canonical_predictions = [
        prediction if prediction in allowed_labels else INVALID_PREDICTION
        for prediction in predictions
    ]

    valid_prediction_count = sum(
        prediction in allowed_labels for prediction in predictions
    )

    in_scope_indices = [
        index
        for index, reference in enumerate(references)
        if reference != oos_label
    ]

    if not in_scope_indices:
        raise ValueError("Evaluation data contains no in-scope examples")

    in_scope_correct = sum(
        references[index] == canonical_predictions[index]
        for index in in_scope_indices
    )

    in_scope_false_rejections = sum(
        canonical_predictions[index] == oos_label
        for index in in_scope_indices
    )

    # Treating OOS detection as a binary classification problem
    oos_references = [
        reference == oos_label for reference in references
    ]
    oos_predictions = [
        prediction == oos_label for prediction in canonical_predictions
    ]

    oos_precision, oos_recall, oos_f1, _ = (
        precision_recall_fscore_support(
            oos_references,
            oos_predictions,
            average="binary",
            pos_label=True,
            zero_division=0,
        )
    )

    sample_count = len(references)
    in_scope_count = len(in_scope_indices)

    return {
        "sample_count": sample_count,
        "valid_prediction_count": valid_prediction_count,
        "invalid_prediction_count": (
            sample_count - valid_prediction_count
        ),
        "overall_accuracy": float(
            accuracy_score(references, canonical_predictions)
        ),
        "macro_f1_all_labels": float(
            f1_score(
                references,
                canonical_predictions,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "in_scope_accuracy": in_scope_correct / in_scope_count,
        "oos_precision": float(oos_precision),
        "oos_recall": float(oos_recall),
        "oos_f1": float(oos_f1),
        "in_scope_false_rejection_rate": (
            in_scope_false_rejections / in_scope_count
        ),
        "valid_label_rate": valid_prediction_count / sample_count,
    }


def write_metrics(path: Path, payload: dict[str, object]) -> None:
    """Write a deterministic, human-readable metrics artifact"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )