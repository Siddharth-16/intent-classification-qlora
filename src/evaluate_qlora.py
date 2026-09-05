from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from src.baseline_zero_shot import (
    generate_batch,
    select_records,
    write_jsonl,
)
from src.contracts import (
    DEVELOPMENT_SPLIT,
    FINAL_SPLIT,
    MODEL_ID,
    MODEL_REVISION,
    RANDOM_SEED,
    parse_generated_label,
)
from src.evaluation import (
    evaluate_predictions,
    write_metrics,
)
from src.prompting import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_prompt,
)
from src.training_data import (
    LABEL_MAPPING_PATH,
    VALIDATION_PATH,
    read_jsonl,
    read_labels,
)
from src.frozen_evaluation import (
    load_frozen_evaluation_config,
    validate_frozen_adapter,
)

APPROACH = "qwen_qlora"

CONFIG_PATH = Path("configs/qlora.json")
ADAPTER_PATH = Path(
    "artifacts/adapters/qwen_qlora__seed-42"
)
TRAINING_REPORT_PATH = Path(
    "artifacts/metrics/qwen_qlora__seed-42.json"
)
TEST_PATH = Path("data/processed/test.jsonl")
METRICS_DIR = Path("artifacts/metrics")
SMOKE_METRICS_DIR = Path("artifacts/logs/smoke")
PREDICTIONS_DIR = Path("artifacts/predictions")


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON document"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one local file"""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def directory_size_bytes(path: Path) -> int:
    """Return the recursive size of a directory"""

    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )


def validate_adapter(
    training_config: dict[str, Any],
) -> dict[str, object]:
    """Validate adapter and training metadata before evaluation"""

    adapter_config_path = (
        ADAPTER_PATH / "adapter_config.json"
    )
    adapter_weights_path = (
        ADAPTER_PATH / "adapter_model.safetensors"
    )

    required_paths = (
        ADAPTER_PATH,
        adapter_config_path,
        adapter_weights_path,
        TRAINING_REPORT_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required evaluation artifact is missing: {path}"
            )

    peft_config = PeftConfig.from_pretrained(
        str(ADAPTER_PATH)
    )
    training_report = read_json(TRAINING_REPORT_PATH)

    if peft_config.base_model_name_or_path != MODEL_ID:
        raise ValueError(
            "Adapter base model mismatch: "
            f"{peft_config.base_model_name_or_path!r} "
            f"!= {MODEL_ID!r}"
        )

    if training_report["model_id"] != MODEL_ID:
        raise ValueError(
            "Training report model ID does not match "
            "the evaluation contract"
        )

    if (
        training_report["model_revision"]
        != MODEL_REVISION
    ):
        raise ValueError(
            "Training report model revision does not match "
            "the evaluation contract"
        )

    if (
        training_report["prompt_version"]
        != PROMPT_VERSION
    ):
        raise ValueError(
            "Training and evaluation prompt versions differ"
        )

    if training_report["is_smoke_run"]:
        raise ValueError(
            "The selected adapter came from a smoke run"
        )

    if (
        training_report["training_config"]
        != training_config
    ):
        raise ValueError(
            "Current QLoRA configuration differs from "
            "the recorded training configuration"
        )

    return {
        "path": str(ADAPTER_PATH),
        "weights_file": adapter_weights_path.name,
        "weights_sha256": sha256_file(
            adapter_weights_path
        ),
        "size_mib": round(
            directory_size_bytes(ADAPTER_PATH)
            / (1024**2),
            6,
        ),
        "base_model": (
            peft_config.base_model_name_or_path
        ),
        "task_type": str(peft_config.task_type),
    }


def run(
    limit: int | None,
    batch_size: int,
    max_new_tokens: int,
    evaluation_split: str,
) -> None:
    """Evaluate the trained adapter on the selected split"""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for QLoRA evaluation"
        )

    if batch_size < 1:
        raise ValueError("--batch-size must be positive")

    if max_new_tokens < 1:
        raise ValueError(
            "--max-new-tokens must be positive"
        )

    set_seed(RANDOM_SEED)

    training_config = read_json(CONFIG_PATH)
    adapter_metadata = validate_adapter(
        training_config
    )

    frozen_config: dict[str, Any] | None = None

    if evaluation_split == FINAL_SPLIT:
        if limit is not None:
            raise ValueError(
                "--limit is prohibited for final test evaluation"
            )

        frozen_config = (
            load_frozen_evaluation_config()
        )
        frozen_adapter_dir, _ = (
            validate_frozen_adapter(frozen_config)
        )

        if frozen_adapter_dir != ADAPTER_PATH:
            raise ValueError(
                "Evaluator adapter path differs from "
                "the frozen adapter path"
            )

        generation = frozen_config["generation"]
        batch_size = int(generation["batch_size"])
        max_new_tokens = int(
            generation["max_new_tokens"]
        )

        print(
            "Using frozen test settings: "
            f"batch_size={batch_size}, "
            f"max_new_tokens={max_new_tokens}"
        )

    split_paths = {
        DEVELOPMENT_SPLIT: VALIDATION_PATH,
        FINAL_SPLIT: TEST_PATH,
    }

    records = select_records(
        read_jsonl(split_paths[evaluation_split]),
        limit,
    )
    labels = read_labels(LABEL_MAPPING_PATH)
    allowed_labels = set(labels)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    compute_dtype_name = str(
        compute_dtype
    ).replace("torch.", "")

    if frozen_config is None:
        quantization = {
            "load_in_4bit": True,
            "quant_type": training_config[
                "bnb_4bit_quant_type"
            ],
            "double_quantization": training_config[
                "bnb_4bit_use_double_quant"
            ],
            "compute_dtype": compute_dtype_name,
        }
    else:
        quantization = frozen_config[
            "quantization"
        ]

        if (
            quantization["compute_dtype"]
            != compute_dtype_name
        ):
            raise RuntimeError(
                "Runtime compute dtype differs from "
                "the frozen configuration"
            )

        if (
            quantization["quant_type"]
            != training_config[
                "bnb_4bit_quant_type"
            ]
            or quantization[
                "double_quantization"
            ]
            != training_config[
                "bnb_4bit_use_double_quant"
            ]
        ):
            raise ValueError(
                "Frozen quantization differs from "
                "the training configuration"
            )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization[
            "load_in_4bit"
        ],
        bnb_4bit_quant_type=quantization[
            "quant_type"
        ],
        bnb_4bit_use_double_quant=quantization[
            "double_quantization"
        ],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    load_started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={
            "": torch.cuda.current_device()
        },
        low_cpu_mem_usage=True,
    )
    base_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        is_trainable=False,
    )
    model.config.use_cache = True
    model.eval()

    model_load_seconds = (
        time.perf_counter() - load_started
    )

    prompts = [
        build_prompt(
            text=record["text"],
            labels=labels,
            tokenizer=tokenizer,
        )
        for record in records
    ]

    # Warm up CUDA before collecting timing and memory.
    generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts[:1],
        max_new_tokens=max_new_tokens,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    raw_outputs: list[str] = []
    inference_started = time.perf_counter()

    for start in range(
        0,
        len(prompts),
        batch_size,
    ):
        batch_prompts = prompts[
            start : start + batch_size
        ]

        raw_outputs.extend(
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
            )
        )

        completed = min(
            start + batch_size,
            len(prompts),
        )
        print(
            f"Processed {completed}/{len(prompts)}",
            flush=True,
        )

    torch.cuda.synchronize()

    inference_seconds = (
        time.perf_counter() - inference_started
    )

    predictions: list[str | None] = []
    prediction_records: list[dict[str, object]] = []

    for record, raw_output in zip(
        records,
        raw_outputs,
        strict=True,
    ):
        prediction, valid_output = (
            parse_generated_label(
                raw_output,
                allowed_labels,
            )
        )

        reference = record["label"]
        predictions.append(prediction)

        prediction_records.append(
            {
                "id": record["id"],
                "text": record["text"],
                "reference": reference,
                "raw_output": raw_output,
                "prediction": prediction,
                "valid_output": valid_output,
                "correct": prediction == reference,
            }
        )

    references = [
        record["label"] for record in records
    ]

    metrics = evaluate_predictions(
        references=references,
        predictions=predictions,
        labels=labels,
    )

    base_name = (
        f"{APPROACH}__{evaluation_split}"
        f"__seed-{RANDOM_SEED}"
    )
    experiment_name = (
        base_name
        if limit is None
        else f"{base_name}__limit-{limit}"
    )

    metrics_directory = (
        METRICS_DIR
        if limit is None
        else SMOKE_METRICS_DIR
    )
    metrics_path = (
        metrics_directory
        / f"{experiment_name}.json"
    )
    predictions_path = (
        PREDICTIONS_DIR
        / f"{experiment_name}.jsonl"
    )

    write_jsonl(
        predictions_path,
        prediction_records,
    )

    result: dict[str, object] = {
        "experiment_name": experiment_name,
        "approach": APPROACH,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter": adapter_metadata,
        "checkpoint_selection": (
            "final adapter after one training epoch"
        ),
        "training_report": str(
            TRAINING_REPORT_PATH
        ),
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "dataset": "DeepPavlov/clinc_oos",
        "dataset_config": "plus",
        "evaluation_split": evaluation_split,
        "is_smoke_run": limit is not None,
        "selection": (
            "full_split"
            if limit is None
            else "seeded_random_subset"
        ),
        "random_seed": RANDOM_SEED,
        "evaluation_rows": len(records),
        "label_count": len(labels),
        "generation": {
            "decoding": "greedy",
            "do_sample": False,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
        },
        "quantization": quantization,
        "timing": {
            "model_load_seconds": round(
                model_load_seconds,
                6,
            ),
            "inference_seconds": round(
                inference_seconds,
                6,
            ),
            "examples_per_second": round(
                len(records) / inference_seconds,
                6,
            ),
            "batched_mean_ms_per_example": round(
                inference_seconds
                * 1000
                / len(records),
                6,
            ),
        },
        "memory": {
            "gpu_name": torch.cuda.get_device_name(0),
            "model_footprint_gib": round(
                model.get_memory_footprint()
                / (1024**3),
                6,
            ),
            "peak_cuda_memory_gib": round(
                torch.cuda.max_memory_allocated()
                / (1024**3),
                6,
            ),
        },
        "metrics": metrics,
    }

    write_metrics(metrics_path, result)

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
    )

    incorrect_examples = [
        record
        for record in prediction_records
        if not record["correct"]
    ][:5]

    if incorrect_examples:
        print("First incorrect predictions:")
        print(
            json.dumps(
                incorrect_examples,
                indent=2,
                ensure_ascii=False,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the validation-only CLI"""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained QLoRA adapter "
            "on a CLINC-OOS evaluation split."
        )
    )
    parser.add_argument(
        "--split",
        choices=(
            DEVELOPMENT_SPLIT,
            FINAL_SPLIT,
        ),
        default=DEVELOPMENT_SPLIT,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )

    return parser


def main() -> None:
    """Run the command-line evaluator"""

    args = build_parser().parse_args()

    run(
        limit=args.limit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        evaluation_split=args.split,
    )


if __name__ == "__main__":
    main()