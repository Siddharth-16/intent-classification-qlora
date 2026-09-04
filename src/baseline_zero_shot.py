from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from src.contracts import (
    DEVELOPMENT_SPLIT,
    MODEL_ID,
    MODEL_REVISION,
    RANDOM_SEED,
    parse_generated_label,
)
from src.evaluation import evaluate_predictions, write_metrics
from src.prompting import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_prompt,
)

APPROACH = "qwen_zero_shot"

VALIDATION_PATH = Path("data/processed/validation.jsonl")
LABEL_MAPPING_PATH = Path("artifacts/metrics/label_mapping.json")
METRICS_DIR = Path("artifacts/metrics")
SMOKE_METRICS_DIR = Path("artifacts/logs/smoke")
PREDICTIONS_DIR = Path("artifacts/predictions")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a processed JSONL file"""

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


def select_records(
    records: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select a reproducible random subset for smoke testing"""

    if limit is None:
        return records

    if not 1 <= limit <= len(records):
        raise ValueError(
            f"--limit must be between 1 and {len(records)}"
        )

    generator = random.Random(RANDOM_SEED)
    indices = sorted(
        generator.sample(range(len(records)), limit)
    )

    return [records[index] for index in indices]


def build_prompt(
    text: str,
    labels: list[str],
    tokenizer,
) -> str:
    label_block = "\n".join(f"- {label}" for label in labels)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"ALLOWED_LABELS:\n{label_block}\n\n"
                f"USER_REQUEST:\n{text}\n\n"
                "Return one exact label from ALLOWED_LABELS:"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Generate one batch using deterministic greedy decoding"""

    inputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    prompt_width = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, prompt_width:]

    return tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )


def write_jsonl(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    """Write local prediction records for error analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def run(
    limit: int | None,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    """Run the validation-only zero-shot baseline"""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this baseline"
        )

    set_seed(RANDOM_SEED)

    records = select_records(
        read_jsonl(VALIDATION_PATH),
        limit,
    )
    labels = read_labels(LABEL_MAPPING_PATH)
    allowed_labels = set(labels)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="auto",
        dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    model.eval()

    model_load_seconds = (
        time.perf_counter() - load_started
    )

    prompts = [
        build_prompt(
            record["text"],
            labels,
            tokenizer
        )
        for record in records
    ]

    # Warm up CUDA before measuring inference
    generate_batch(
        model,
        tokenizer,
        prompts[:1],
        max_new_tokens,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    raw_outputs: list[str] = []
    inference_started = time.perf_counter()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[
            start : start + batch_size
        ]

        raw_outputs.extend(
            generate_batch(
                model,
                tokenizer,
                batch_prompts,
                max_new_tokens,
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

        predictions.append(prediction)

        prediction_records.append(
            {
                "id": record["id"],
                "text": record["text"],
                "reference": record["label"],
                "raw_output": raw_output,
                "prediction": prediction,
                "valid_output": valid_output,
            }
        )

    references = [
        record["label"] for record in records
    ]

    metrics = evaluate_predictions(
        references,
        predictions,
        labels,
    )

    base_name = (
        f"{APPROACH}__{DEVELOPMENT_SPLIT}"
        f"__seed-{RANDOM_SEED}"
    )

    experiment_name = (
        base_name
        if limit is None
        else f"{base_name}__limit-{limit}"
    )

    metrics_dir = (
        METRICS_DIR
        if limit is None
        else SMOKE_METRICS_DIR
    )

    metrics_path = (
        metrics_dir / f"{experiment_name}.json"
    )
    predictions_path = (
        PREDICTIONS_DIR / f"{experiment_name}.jsonl"
    )

    write_jsonl(
        predictions_path,
        prediction_records,
    )

    result: dict[str, object] = {
        "experiment_name": experiment_name,
        "approach": APPROACH,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": "DeepPavlov/clinc_oos",
        "dataset_config": "plus",
        "evaluation_split": DEVELOPMENT_SPLIT,
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
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quantization": True,
            "compute_dtype": str(
                compute_dtype
            ).replace("torch.", ""),
        },
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
                * 1_000
                / len(records),
                6,
            ),
        },
        "memory": {
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
            "gpu_name": torch.cuda.get_device_name(0),
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

    invalid_examples = [
        record
        for record in prediction_records
        if not record["valid_output"]
    ][:5]

    if invalid_examples:
        print("First invalid outputs:")
        print(
            json.dumps(
                invalid_examples,
                indent=2,
                ensure_ascii=False,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the validation-only CLI"""

    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot Qwen classification "
            "on validation only."
        )
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
    """Run the command-line program"""

    args = build_parser().parse_args()

    run(
        limit=args.limit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()