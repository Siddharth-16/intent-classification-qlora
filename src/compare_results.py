from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.contracts import (
    DEVELOPMENT_SPLIT,
    FINAL_SPLIT,
    METRICS,
    RANDOM_SEED,
)
from src.evaluation import write_metrics
from src.frozen_evaluation import (
    EVALUATION_CONFIG_PATH,
    load_frozen_evaluation_config,
)

METRICS_DIR = Path("artifacts/metrics")
TRAINING_REPORT_PATH = (
    METRICS_DIR / f"qwen_qlora__seed-{RANDOM_SEED}.json"
)
APPROACHES = (
    "tfidf",
    "qwen_zero_shot",
    "qwen_qlora",
)


def read_json(path: Path) -> dict[str, Any]:
    """Read one metrics artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics artifact: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_path(approach: str, split: str) -> Path:
    """Return the canonical metrics path for one experiment."""

    return (
        METRICS_DIR
        / f"{approach}__{split}__seed-{RANDOM_SEED}.json"
    )


def percentage_point_change(
    candidate: dict[str, int | float],
    baseline: dict[str, int | float],
) -> dict[str, float]:
    """Calculate candidate-minus-baseline changes in percentage points."""

    return {
        metric: round(
            100
            * (
                float(candidate[metric])
                - float(baseline[metric])
            ),
            6,
        )
        for metric in METRICS
    }


def mean_latency_ms(
    approach: str,
    artifact: dict[str, Any],
) -> float:
    """Read the comparable per-example inference latency."""

    timing = artifact["timing"]

    if approach == "tfidf":
        return float(
            timing["mean_inference_ms_per_example"]
        )

    return float(
        timing["batched_mean_ms_per_example"]
    )


def validate_artifacts(
    artifacts: dict[str, dict[str, Any]],
    split: str,
) -> int:
    """Ensure every comparison uses the same evaluation records."""

    sample_counts = set()
    dataset_ids = set()
    dataset_configs = set()
    random_seeds = set()

    for approach, artifact in artifacts.items():
        if artifact["approach"] != approach:
            raise ValueError(
                f"Approach mismatch in {approach} artifact"
            )

        if artifact["evaluation_split"] != split:
            raise ValueError(
                f"Split mismatch in {approach} artifact"
            )

        sample_counts.add(
            int(artifact["metrics"]["sample_count"])
        )
        dataset_ids.add(artifact["dataset"])
        dataset_configs.add(artifact["dataset_config"])
        random_seeds.add(int(artifact["random_seed"]))

        if (
            int(artifact["evaluation_rows"])
            != int(artifact["metrics"]["sample_count"])
        ):
            raise ValueError(
                f"Row-count mismatch in {approach} artifact"
            )

    if len(sample_counts) != 1:
        raise ValueError(
            "Compared experiments used different sample counts"
        )

    if (
        len(dataset_ids) != 1
        or len(dataset_configs) != 1
        or random_seeds != {RANDOM_SEED}
    ):
        raise ValueError(
            "Compared experiments do not share dataset and seed metadata"
        )

    return sample_counts.pop()


def build_comparison(split: str) -> dict[str, object]:
    """Build the three-way experiment comparison."""

    if split == FINAL_SPLIT:
        load_frozen_evaluation_config()

    artifacts = {
        approach: read_json(metrics_path(approach, split))
        for approach in APPROACHES
    }
    sample_count = validate_artifacts(artifacts, split)

    metrics = {
        approach: artifact["metrics"]
        for approach, artifact in artifacts.items()
    }
    latencies = {
        approach: mean_latency_ms(approach, artifact)
        for approach, artifact in artifacts.items()
    }

    qlora_metrics = metrics["qwen_qlora"]
    training_report = read_json(TRAINING_REPORT_PATH)
    adapter = artifacts["qwen_qlora"]["adapter"]

    return {
        "evaluation_split": split,
        "sample_count": sample_count,
        "random_seed": RANDOM_SEED,
        "primary_metric": "macro_f1_all_labels",
        "comparison_direction": (
            "qwen_qlora minus baseline; values are percentage points"
        ),
        "results": metrics,
        "percentage_point_change": {
            "qwen_qlora_vs_qwen_zero_shot": (
                percentage_point_change(
                    qlora_metrics,
                    metrics["qwen_zero_shot"],
                )
            ),
            "qwen_qlora_vs_tfidf": (
                percentage_point_change(
                    qlora_metrics,
                    metrics["tfidf"],
                )
            ),
        },
        "invalid_output_change": {
            "qwen_zero_shot_count": metrics[
                "qwen_zero_shot"
            ]["invalid_prediction_count"],
            "qwen_qlora_count": qlora_metrics[
                "invalid_prediction_count"
            ],
            "reduction_percent": round(
                100
                * (
                    float(
                        metrics["qwen_zero_shot"][
                            "invalid_prediction_count"
                        ]
                    )
                    - float(
                        qlora_metrics[
                            "invalid_prediction_count"
                        ]
                    )
                )
                / float(
                    metrics["qwen_zero_shot"][
                        "invalid_prediction_count"
                    ]
                ),
                6,
            ),
        },
        "inference": {
            "mean_ms_per_example": latencies,
            "qlora_latency_ratio_vs_zero_shot": round(
                latencies["qwen_qlora"]
                / latencies["qwen_zero_shot"],
                6,
            ),
            "qlora_latency_ratio_vs_tfidf": round(
                latencies["qwen_qlora"]
                / latencies["tfidf"],
                6,
            ),
            "measurement_note": (
                "Qwen measurements used the same GPU and batch size; "
                "TF-IDF used CPU inference and is not hardware-equivalent."
            ),
        },
        "training": {
            "elapsed_gpu_hours": round(
                float(
                    training_report[
                        "measured_training_seconds"
                    ]
                )
                / 3600,
                6,
            ),
            "trainable_parameters": training_report[
                "parameters"
            ]["trainable_parameters"],
            "trainable_percentage": training_report[
                "parameters"
            ]["trainable_percentage"],
            "adapter_size_mib": adapter["size_mib"],
            "adapter_weights_sha256": adapter[
                "weights_sha256"
            ],
        },
        "frozen_configuration": str(
            EVALUATION_CONFIG_PATH
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare TF-IDF, zero-shot Qwen, and QLoRA results."
        )
    )
    parser.add_argument(
        "--split",
        choices=(
            DEVELOPMENT_SPLIT,
            FINAL_SPLIT,
        ),
        default=FINAL_SPLIT,
    )
    return parser


def main() -> None:
    """Build and save the requested comparison."""

    args = build_parser().parse_args()
    comparison = build_comparison(args.split)
    output_path = (
        METRICS_DIR
        / f"comparison__{args.split}__seed-{RANDOM_SEED}.json"
    )

    write_metrics(output_path, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()