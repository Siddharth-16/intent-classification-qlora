from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.contracts import DEVELOPMENT_SPLIT, RANDOM_SEED
from src.evaluation import evaluate_predictions, write_metrics

APPROACH = "tfidf"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one processed JSONL split"""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.data` first."
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}"
                ) from error

            if not isinstance(record.get("text"), str):
                raise ValueError(
                    f"Invalid text in {path}:{line_number}"
                )

            if not isinstance(record.get("label"), str):
                raise ValueError(
                    f"Invalid label in {path}:{line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def read_labels(path: Path) -> list[str]:
    """Load canonical labels in numeric ID order"""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.data` first."
        )

    mapping = json.loads(path.read_text(encoding="utf-8"))
    label_to_id = mapping.get("label_to_id")

    if not isinstance(label_to_id, dict):
        raise ValueError(f"Invalid label mapping in {path}")

    labels = [
        label
        for label, _ in sorted(
            label_to_id.items(),
            key=lambda item: item[1],
        )
    ]

    if not all(isinstance(label, str) for label in labels):
        raise ValueError(f"Invalid labels in {path}")

    return labels


def build_model() -> Pipeline:
    """Create the fixed, untuned classical baseline"""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=1_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def run_baseline(
    processed_dir: Path,
    label_mapping_path: Path,
    metrics_dir: Path,
) -> Path:
    """Fit on train, evaluate on validation, and save the result"""

    train_records = read_jsonl(
        processed_dir / "train.jsonl"
    )
    validation_records = read_jsonl(
        processed_dir / f"{DEVELOPMENT_SPLIT}.jsonl"
    )
    labels = read_labels(label_mapping_path)

    train_texts = [record["text"] for record in train_records]
    train_labels = [record["label"] for record in train_records]

    validation_texts = [
        record["text"] for record in validation_records
    ]
    validation_labels = [
        record["label"] for record in validation_records
    ]

    model = build_model()

    fit_started = time.perf_counter()
    model.fit(train_texts, train_labels)
    fit_seconds = time.perf_counter() - fit_started

    inference_started = time.perf_counter()
    predictions = model.predict(validation_texts).tolist()
    inference_seconds = time.perf_counter() - inference_started

    metrics = evaluate_predictions(
        references=validation_labels,
        predictions=predictions,
        labels=labels,
    )

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    experiment_name = (
        f"{APPROACH}__{DEVELOPMENT_SPLIT}__seed-{RANDOM_SEED}"
    )

    result: dict[str, object] = {
        "experiment_name": experiment_name,
        "approach": APPROACH,
        "dataset": "DeepPavlov/clinc_oos",
        "dataset_config": "plus",
        "train_split": "train",
        "evaluation_split": DEVELOPMENT_SPLIT,
        "random_seed": RANDOM_SEED,
        "train_rows": len(train_records),
        "evaluation_rows": len(validation_records),
        "feature_count": len(
            vectorizer.get_feature_names_out()
        ),
        "configuration": {
            "tfidf": {
                "analyzer": "word",
                "lowercase": True,
                "ngram_range": [1, 2],
                "min_df": 2,
                "sublinear_tf": True,
                "stop_words": None,
            },
            "logistic_regression": {
                "C": 1.0,
                "effective_penalty": "l2",
                "l1_ratio": 0.0,
                "solver": "lbfgs",
                "max_iter": 1_000,
                "class_weight": None,
            },
        },
        "timing": {
            "fit_seconds": round(fit_seconds, 6),
            "inference_seconds": round(
                inference_seconds, 6
            ),
            "mean_inference_ms_per_example": round(
                inference_seconds
                * 1_000
                / len(validation_records),
                6,
            ),
        },
        "optimizer_iterations": int(
            classifier.n_iter_.max()
        ),
        "metrics": metrics,
    }

    output_path = metrics_dir / f"{experiment_name}.json"
    write_metrics(output_path, result)

    print(f"Saved metrics: {output_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run TF-IDF + logistic regression "
            "on CLINC-OOS validation."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=Path(
            "artifacts/metrics/label_mapping.json"
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("artifacts/metrics"),
    )
    return parser


def main() -> None:
    """Run the baseline from the command line"""

    args = build_parser().parse_args()

    run_baseline(
        processed_dir=args.processed_dir,
        label_mapping_path=args.label_mapping,
        metrics_dir=args.metrics_dir,
    )


if __name__ == "__main__":
    main()