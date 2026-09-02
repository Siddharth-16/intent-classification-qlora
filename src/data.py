from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from datasets import DatasetDict, load_dataset

from src.contracts import OOS_LABEL, write_evaluation_contract

HF_DATASET_NAME = "DeepPavlov/clinc_oos"
HF_DATASET_CONFIG = "plus"
HF_DATASET_REVISION = "d76f3a952f3dd39124311cc9db4648d31cbb8774"
HF_DATASET_URL = "https://huggingface.co/datasets/DeepPavlov/clinc_oos"

EXPECTED_SPLIT_SIZES = {
    "train": 15_250,
    "validation": 3_100,
    "test": 5_500,
}
EXPECTED_OOS_COUNTS = {
    "train": 250,
    "validation": 100,
    "test": 1_000,
}
EXPECTED_IN_SCOPE_PER_LABEL = {
    "train": 100,
    "validation": 20,
    "test": 30,
}
EXPECTED_IN_SCOPE_LABELS = 150

TRACKED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "peft",
    "scikit-learn",
    "torch",
    "transformers",
    "trl",
)


def extract_labels(source_data: DatasetDict) -> list[str]:
    """Return a deterministic label order with ``oos`` placed last."""

    train_labels = set(source_data["train"]["label_text"])
    if not all(isinstance(label, str) and label for label in train_labels):
        raise ValueError("Training data contains an invalid label_text value")

    if OOS_LABEL not in train_labels:
        raise ValueError(f"Training data does not contain {OOS_LABEL!r}")

    in_scope_labels = sorted(train_labels - {OOS_LABEL})
    if len(in_scope_labels) != EXPECTED_IN_SCOPE_LABELS:
        raise ValueError(
            f"Expected {EXPECTED_IN_SCOPE_LABELS} in-scope labels, "
            f"found {len(in_scope_labels)}"
        )

    return in_scope_labels + [OOS_LABEL]


def build_splits(source_data: DatasetDict) -> dict[str, list[dict[str, str]]]:
    """Convert the Hub dataset into the project's modeling-record format."""

    actual_splits = set(source_data)
    expected_splits = set(EXPECTED_SPLIT_SIZES)
    if actual_splits != expected_splits:
        raise ValueError(
            "Unexpected Hugging Face splits; "
            f"expected={sorted(expected_splits)}, found={sorted(actual_splits)}"
        )

    splits: dict[str, list[dict[str, str]]] = {}
    for split_name in EXPECTED_SPLIT_SIZES:
        records: list[dict[str, str]] = []

        for index, row in enumerate(source_data[split_name]):
            text = row["text"]
            label = row["label_text"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Found empty or invalid text in {split_name}")
            if not isinstance(label, str) or not label:
                raise ValueError(f"Found invalid label_text in {split_name}")

            records.append(
                {
                    "id": f"{split_name}-{index:05d}",
                    "text": text,
                    "label": label,
                }
            )

        splits[split_name] = records

    return splits


def validate_splits(
    splits: dict[str, list[dict[str, str]]], labels: list[str]
) -> None:
    """Fail loudly if the pinned CLINC-OOS Plus contract changes."""

    if set(splits) != set(EXPECTED_SPLIT_SIZES):
        raise ValueError(f"Unexpected processed splits: {sorted(splits)}")

    expected_labels = set(labels)
    for split_name, records in splits.items():
        expected_size = EXPECTED_SPLIT_SIZES[split_name]
        if len(records) != expected_size:
            raise ValueError(
                f"{split_name} expected {expected_size} rows, "
                f"found {len(records)}"
            )

        counts = Counter(record["label"] for record in records)
        if set(counts) != expected_labels:
            missing = sorted(expected_labels - set(counts))
            extra = sorted(set(counts) - expected_labels)
            raise ValueError(
                f"{split_name} label mismatch; missing={missing}, extra={extra}"
            )

        expected_oos = EXPECTED_OOS_COUNTS[split_name]
        if counts[OOS_LABEL] != expected_oos:
            raise ValueError(
                f"{split_name} expected {expected_oos} OOS rows, "
                f"found {counts[OOS_LABEL]}"
            )

        expected_per_label = EXPECTED_IN_SCOPE_PER_LABEL[split_name]
        incorrect_counts = {
            label: count
            for label, count in counts.items()
            if label != OOS_LABEL and count != expected_per_label
        }
        if incorrect_counts:
            raise ValueError(
                f"{split_name} has unexpected in-scope label counts: "
                f"{incorrect_counts}"
            )


def percentile(values: list[int], proportion: float) -> int:
    """Return a deterministic nearest-rank percentile."""

    ordered = sorted(values)
    rank = max(1, int(proportion * len(ordered) + 0.999999))
    return ordered[rank - 1]


def normalized_text(text: str) -> str:
    """Normalize text only for duplicate diagnostics, never for modeling."""

    return " ".join(text.lower().split())


def duplicate_report(
    splits: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    """Describe normalized duplicates within and across the fixed splits."""

    text_indexes: dict[str, dict[str, list[dict[str, str]]]] = {}
    within_split: dict[str, int] = {}

    for split_name, records in splits.items():
        index: dict[str, list[dict[str, str]]] = {}
        for record in records:
            index.setdefault(normalized_text(record["text"]), []).append(record)
        text_indexes[split_name] = index
        within_split[split_name] = sum(len(group) - 1 for group in index.values())

    cross_split: dict[str, object] = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlaps = sorted(set(text_indexes[left]) & set(text_indexes[right]))
        pairs: list[dict[str, object]] = []

        for text in overlaps:
            for left_record in text_indexes[left][text]:
                for right_record in text_indexes[right][text]:
                    pairs.append(
                        {
                            "left_id": left_record["id"],
                            "right_id": right_record["id"],
                            "left_label": left_record["label"],
                            "right_label": right_record["label"],
                            "same_label": (
                                left_record["label"] == right_record["label"]
                            ),
                        }
                    )

        cross_split[f"{left}_{right}"] = {
            "overlap_count": len(overlaps),
            "same_label_count": sum(bool(pair["same_label"]) for pair in pairs),
            "conflicting_label_count": sum(
                not bool(pair["same_label"]) for pair in pairs
            ),
            "record_pairs": pairs,
        }

    return {"within_split": within_split, "cross_split": cross_split}


def text_statistics(records: Iterable[dict[str, str]]) -> dict[str, float | int]:
    """Summarize text lengths without depending on a model tokenizer."""

    rows = list(records)
    word_counts = [len(record["text"].split()) for record in rows]
    character_counts = [len(record["text"]) for record in rows]
    return {
        "word_count_min": min(word_counts),
        "word_count_median": statistics.median(word_counts),
        "word_count_mean": round(statistics.fmean(word_counts), 3),
        "word_count_p95": percentile(word_counts, 0.95),
        "word_count_max": max(word_counts),
        "character_count_min": min(character_counts),
        "character_count_median": statistics.median(character_counts),
        "character_count_mean": round(statistics.fmean(character_counts), 3),
        "character_count_p95": percentile(character_counts, 0.95),
        "character_count_max": max(character_counts),
    }


def build_data_report(
    splits: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    """Build the dataset-integrity and descriptive-statistics report."""

    split_reports: dict[str, object] = {}
    for split_name, records in splits.items():
        label_counts = Counter(record["label"] for record in records)
        split_reports[split_name] = {
            "rows": len(records),
            "in_scope_rows": len(records) - label_counts[OOS_LABEL],
            "oos_rows": label_counts[OOS_LABEL],
            "oos_fraction": round(label_counts[OOS_LABEL] / len(records), 6),
            "unique_labels": len(label_counts),
            "text_statistics": text_statistics(records),
        }

    return {
        "dataset": "CLINC-OOS Plus",
        "source": "Hugging Face Hub",
        "source_repository": HF_DATASET_URL,
        "source_dataset": HF_DATASET_NAME,
        "source_config": HF_DATASET_CONFIG,
        "source_revision": HF_DATASET_REVISION,
        "in_scope_label_count": EXPECTED_IN_SCOPE_LABELS,
        "total_label_count": EXPECTED_IN_SCOPE_LABELS + 1,
        "label_order": "alphabetical in-scope labels followed by oos",
        "split_usage": {
            "train": "fit model parameters",
            "validation": "prompt, configuration, and checkpoint decisions",
            "test": "one final frozen evaluation; never tune from results",
        },
        "splits": split_reports,
        "duplicates": duplicate_report(splits),
    }


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, str]]) -> None:
    """Write modeling records as JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_environment_report() -> dict[str, object]:
    """Record the runtime without importing optional GPU libraries."""

    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "note": (
            "Regenerate this file inside the final Colab/CUDA runtime before "
            "reporting training or inference results."
        ),
    }


def prepare_dataset(data_dir: Path, metrics_dir: Path) -> dict[str, object]:
    """Download, validate, normalize, and document CLINC-OOS Plus."""

    source_data = load_dataset(
        HF_DATASET_NAME,
        HF_DATASET_CONFIG,
        revision=HF_DATASET_REVISION,
        cache_dir=str(data_dir / "cache"),
    )
    if not isinstance(source_data, DatasetDict):
        raise TypeError("Expected load_dataset() to return a DatasetDict")

    labels = extract_labels(source_data)
    splits = build_splits(source_data)
    validate_splits(splits, labels)

    processed_dir = data_dir / "processed"
    for split_name, records in splits.items():
        write_jsonl(processed_dir / f"{split_name}.jsonl", records)

    label_mapping = {
        "label_to_id": {label: index for index, label in enumerate(labels)},
        "id_to_label": {str(index): label for index, label in enumerate(labels)},
    }
    report = build_data_report(splits)

    write_json(metrics_dir / "data_report.json", report)
    write_json(metrics_dir / "label_mapping.json", label_mapping)
    write_json(metrics_dir / "environment.json", build_environment_report())
    write_evaluation_contract(metrics_dir / "evaluation_contract.json", labels)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Download and validate CLINC-OOS Plus from Hugging Face."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for cached and processed data (default: data).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("artifacts/metrics"),
        help="Directory for small tracked metadata artifacts.",
    )
    return parser


def main() -> None:
    """Run data preparation from the command line."""

    args = build_parser().parse_args()
    report = prepare_dataset(args.data_dir, args.metrics_dir)
    split_summary = {
        split_name: details["rows"]
        for split_name, details in report["splits"].items()
    }

    print("CLINC-OOS Plus validation passed.")
    print(json.dumps(split_summary, indent=2))
    print(f"Data report: {args.metrics_dir / 'data_report.json'}")


if __name__ == "__main__":
    main()
