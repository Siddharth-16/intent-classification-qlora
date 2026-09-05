from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.contracts import (
    DEVELOPMENT_SPLIT,
    FINAL_SPLIT,
    MODEL_ID,
    MODEL_REVISION,
    RANDOM_SEED,
)
from src.data import (
    HF_DATASET_CONFIG,
    HF_DATASET_NAME,
    HF_DATASET_REVISION,
)
from src.prompting import PROMPT_VERSION

EVALUATION_CONFIG_PATH = Path(
    "configs/evaluation.json"
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON document"""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required configuration: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file"""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_frozen_evaluation_config(
    path: Path = EVALUATION_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the final evaluation manifest"""

    config = read_json(path)

    expected_values = {
        "dataset.id": (
            config["dataset"]["id"],
            HF_DATASET_NAME,
        ),
        "dataset.config": (
            config["dataset"]["config"],
            HF_DATASET_CONFIG,
        ),
        "dataset.revision": (
            config["dataset"]["revision"],
            HF_DATASET_REVISION,
        ),
        "model.id": (
            config["model"]["id"],
            MODEL_ID,
        ),
        "model.revision": (
            config["model"]["revision"],
            MODEL_REVISION,
        ),
        "prompt_version": (
            config["prompt_version"],
            PROMPT_VERSION,
        ),
        "random_seed": (
            config["random_seed"],
            RANDOM_SEED,
        ),
        "development_split": (
            config["development_split"],
            DEVELOPMENT_SPLIT,
        ),
        "final_split": (
            config["final_split"],
            FINAL_SPLIT,
        ),
    }

    for name, (actual, expected) in (
        expected_values.items()
    ):
        if actual != expected:
            raise ValueError(
                f"Frozen {name} mismatch: "
                f"{actual!r} != {expected!r}"
            )

    if config.get("configuration_frozen") is not True:
        raise ValueError(
            "Final evaluation configuration is not frozen"
        )

    expected_generation = {
        "decoding": "greedy",
        "do_sample": False,
        "batch_size": 8,
        "max_new_tokens": 16,
    }
    if config["generation"] != expected_generation:
        raise ValueError(
            "Frozen generation configuration changed"
        )

    expected_quantization = {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "bfloat16",
    }
    if config["quantization"] != expected_quantization:
        raise ValueError(
            "Frozen quantization configuration changed"
        )

    expected_output_contract = {
        "expected": "exactly one canonical label",
        "normalization": (
            "surrounding whitespace only"
        ),
        "invalid_output_policy": "count as incorrect",
    }
    if (
        config["output_contract"]
        != expected_output_contract
    ):
        raise ValueError(
            "Frozen output contract changed"
        )

    return config


def validate_frozen_adapter(
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """Validate and return the frozen adapter paths"""

    adapter_dir = Path(config["adapter"]["path"])
    weights_path = (
        adapter_dir
        / config["adapter"]["weights_file"]
    )
    validation_artifact = Path(
        config["checkpoint_selection"][
            "validation_artifact"
        ]
    )

    for path in (
        adapter_dir,
        weights_path,
        validation_artifact,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen artifact: {path}"
            )

    actual_sha256 = sha256_file(weights_path)
    expected_sha256 = config["adapter"][
        "weights_sha256"
    ]

    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Adapter checksum mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )

    return adapter_dir, weights_path