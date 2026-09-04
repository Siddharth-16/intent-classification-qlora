from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from src.contracts import (
    MODEL_ID,
    MODEL_REVISION,
    OOS_LABEL,
)
from src.prompting import (
    PROMPT_VERSION,
    build_training_example,
)

TRAIN_PATH = Path("data/processed/train.jsonl")
VALIDATION_PATH = Path("data/processed/validation.jsonl")
LABEL_MAPPING_PATH = Path("artifacts/metrics/label_mapping.json")
REPORT_PATH = Path("artifacts/metrics/training_data_report.json")


def read_jsonl(path: Path) -> list[dict[str, str]]:
    """Read processed modeling records"""

    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def read_labels(path: Path) -> list[str]:
    """Load canonical labels in numeric ID order"""

    mapping = json.loads(
        path.read_text(encoding="utf-8")
    )["label_to_id"]

    return [
        label
        for label, _ in sorted(
            mapping.items(),
            key=lambda item: item[1],
        )
    ]


def build_sft_dataset(
    records: list[dict[str, str]],
    labels: list[str],
) -> Dataset:
    """Convert modeling records to conversational prompt-completion format"""

    allowed_labels = set(labels)

    unknown_labels = sorted(
        {
            record["label"]
            for record in records
            if record["label"] not in allowed_labels
        }
    )
    if unknown_labels:
        raise ValueError(
            f"Records contain unknown labels: {unknown_labels}"
        )

    examples = [
        build_training_example(
            text=record["text"],
            label=record["label"],
            labels=labels,
        )
        for record in records
    ]

    return Dataset.from_list(examples)


def nearest_rank_percentile(
    values: list[int],
    proportion: float,
) -> int:
    """Calculate a deterministic nearest-rank percentile"""

    ordered = sorted(values)
    rank = max(1, math.ceil(proportion * len(ordered)))
    return ordered[rank - 1]


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    """Summarize tokenized sequence lengths"""

    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 3),
        "p95": nearest_rank_percentile(values, 0.95),
        "max": max(values),
    }


def count_chat_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> int:
    """Render chat messages and count their actual token IDs"""

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    return len(encoded["input_ids"])


def audit_dataset(
    dataset: Dataset,
    tokenizer: Any,
) -> dict[str, object]:
    """Measure prompt and complete-sequence token lengths"""

    prompt_lengths: list[int] = []
    sequence_lengths: list[int] = []

    for example in dataset:
        prompt_messages = example["prompt"]
        completion_messages = example["completion"]

        prompt_length = count_chat_tokens(
            tokenizer=tokenizer,
            messages=prompt_messages,
            add_generation_prompt=True,
        )

        sequence_length = count_chat_tokens(
            tokenizer=tokenizer,
            messages=prompt_messages + completion_messages,
            add_generation_prompt=False,
        )

        prompt_lengths.append(prompt_length)
        sequence_lengths.append(sequence_length)

    return {
        "prompt_tokens": summarize_lengths(prompt_lengths),
        "complete_sequence_tokens": summarize_lengths(
            sequence_lengths
        ),
    }


def next_multiple_of_128(value: int) -> int:
    """Round a token length upward to a convenient training boundary"""

    return math.ceil(value / 128) * 128


def main() -> None:
    """Build and audit the QLoRA training datasets"""

    labels = read_labels(LABEL_MAPPING_PATH)
    train_records = read_jsonl(TRAIN_PATH)
    validation_records = read_jsonl(VALIDATION_PATH)

    train_dataset = build_sft_dataset(
        train_records,
        labels,
    )
    validation_dataset = build_sft_dataset(
        validation_records,
        labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    train_audit = audit_dataset(
        train_dataset,
        tokenizer,
    )
    validation_audit = audit_dataset(
        validation_dataset,
        tokenizer,
    )

    maximum_length = max(
        train_audit["complete_sequence_tokens"]["max"],
        validation_audit["complete_sequence_tokens"]["max"],
    )
    recommended_max_length = next_multiple_of_128(
        maximum_length
    )

    label_counts = Counter(
        record["label"] for record in train_records
    )

    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "format": "conversational_prompt_completion",
        "loss_scope": "completion_only",
        "label_count": len(labels),
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "train_oos_rows": label_counts[OOS_LABEL],
        "train_token_lengths": train_audit,
        "validation_token_lengths": validation_audit,
        "maximum_observed_tokens": maximum_length,
        "recommended_max_length": recommended_max_length,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()