from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from src.contracts import (
    MODEL_ID,
    MODEL_REVISION,
    RANDOM_SEED,
)
from src.evaluation import write_metrics
from src.prompting import PROMPT_VERSION
from src.training_data import (
    LABEL_MAPPING_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    build_sft_dataset,
    read_jsonl,
    read_labels,
)

CONFIG_PATH = Path("configs/qlora.json")
DATA_REPORT_PATH = Path(
    "artifacts/metrics/training_data_report.json"
)

CHECKPOINTS_DIR = Path("artifacts/checkpoints")
ADAPTERS_DIR = Path("artifacts/adapters")
METRICS_DIR = Path("artifacts/metrics")
SMOKE_LOGS_DIR = Path("artifacts/logs/smoke")

SMOKE_ROWS = 128
SMOKE_STEPS = 10


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON document"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value: Any) -> Any:
    """Convert training results into JSON-serializable values"""

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()

    if hasattr(value, "item"):
        return value.item()

    return value


def parameter_report(model: Any) -> dict[str, int | float]:
    """Count parameters correctly, including packed 4-bit weights"""

    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = (
            model.get_nb_trainable_parameters()
        )
    else:
        total = sum(
            parameter.numel()
            for parameter in model.parameters()
        )
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": round(
            100 * trainable / total,
            6,
        ),
    }


def directory_size_bytes(path: Path) -> int:
    """Return the recursive size of a directory"""

    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )


def validate_max_length(
    config: dict[str, Any],
    data_report: dict[str, Any],
) -> None:
    """Ensure training cannot truncate an observed example"""

    configured = int(config["max_length"])
    observed = int(
        data_report["maximum_observed_tokens"]
    )

    if configured < observed:
        raise ValueError(
            f"Configured max_length={configured} would "
            f"truncate examples requiring {observed} tokens"
        )


def run(smoke: bool) -> None:
    """Train QLoRA adapters or perform a short smoke run"""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA training")

    config = read_json(CONFIG_PATH)
    data_report = read_json(DATA_REPORT_PATH)
    validate_max_length(config, data_report)

    set_seed(RANDOM_SEED)

    labels = read_labels(LABEL_MAPPING_PATH)
    train_records = read_jsonl(TRAIN_PATH)
    full_train_rows = len(train_records)

    train_dataset = build_sft_dataset(
        train_records,
        labels,
    )

    if smoke:
        train_dataset = (
            train_dataset
            .shuffle(seed=RANDOM_SEED)
            .select(
                range(
                    min(
                        SMOKE_ROWS,
                        len(train_dataset),
                    )
                )
            )
        )
        validation_dataset = None
    else:
        validation_records = read_jsonl(
            VALIDATION_PATH
        )
        validation_dataset = build_sft_dataset(
            validation_records,
            labels,
        )

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_bf16 = compute_dtype == torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "<|im_end|>" not in tokenizer.get_vocab():
        raise ValueError(
            "Qwen end-of-message token is missing "
            "from the tokenizer vocabulary"
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config[
            "bnb_4bit_quant_type"
        ],
        bnb_4bit_use_double_quant=config[
            "bnb_4bit_use_double_quant"
        ],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={
            "": torch.cuda.current_device()
        },
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(
            config["lora_dropout"]
        ),
        target_modules=config[
            "lora_target_modules"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    run_name = (
        f"qwen_qlora__seed-{RANDOM_SEED}"
        + ("__smoke" if smoke else "")
    )

    output_dir = CHECKPOINTS_DIR / run_name
    adapter_dir = ADAPTERS_DIR / run_name

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        max_length=int(config["max_length"]),
        completion_only_loss=True,
        eos_token="<|im_end|>",
        packing=False,
        num_train_epochs=float(
            config["num_train_epochs"]
        ),
        max_steps=SMOKE_STEPS if smoke else -1,
        per_device_train_batch_size=int(
            config["per_device_train_batch_size"]
        ),
        per_device_eval_batch_size=int(
            config["per_device_eval_batch_size"]
        ),
        gradient_accumulation_steps=int(
            config["gradient_accumulation_steps"]
        ),
        learning_rate=float(
            config["learning_rate"]
        ),
        lr_scheduler_type=config[
            "lr_scheduler_type"
        ],
        warmup_steps=(
            1
            if smoke
            else int(config["warmup_steps"])
        ),
        weight_decay=float(
            config["weight_decay"]
        ),
        max_grad_norm=float(
            config["max_grad_norm"]
        ),
        optim=config["optimizer"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        use_cache=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=True,
        logging_strategy="steps",
        logging_steps=(
            1
            if smoke
            else int(config["logging_steps"])
        ),
        logging_first_step=True,
        eval_strategy=(
            "no"
            if smoke
            else "epoch"
        ),
        save_strategy=(
            "no"
            if smoke
            else "epoch"
        ),
        save_total_limit=1,
        prediction_loss_only=True,
        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        dataloader_num_workers=0,
        include_num_input_tokens_seen=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    parameters = parameter_report(trainer.model)

    print(json.dumps(parameters, indent=2))
    trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()

    train_result = trainer.train()

    torch.cuda.synchronize()
    measured_training_seconds = (
        time.perf_counter() - training_started
    )

    adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    trainer.model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
    )

    effective_batch_size = (
        int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"])
    )

    expected_full_steps = math.ceil(
        full_train_rows / effective_batch_size
    )

    projection: dict[str, int | float] | None = None

    if smoke:
        seconds_per_step = (
            measured_training_seconds
            / SMOKE_STEPS
        )
        projection = {
            "expected_full_optimizer_steps": (
                expected_full_steps
            ),
            "observed_seconds_per_optimizer_step": (
                round(seconds_per_step, 6)
            ),
            "estimated_full_training_minutes": round(
                expected_full_steps
                * seconds_per_step
                / 60,
                3,
            ),
        }

    result = {
        "experiment_name": run_name,
        "is_smoke_run": smoke,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "dataset": "DeepPavlov/clinc_oos",
        "dataset_config": "plus",
        "training_split": "train",
        "validation_split": (
            None if smoke else "validation"
        ),
        "training_rows": len(train_dataset),
        "full_training_rows": full_train_rows,
        "label_count": len(labels),
        "training_format": (
            "conversational_prompt_completion"
        ),
        "loss_scope": "completion_only",
        "training_config": config,
        "smoke_steps": (
            SMOKE_STEPS if smoke else None
        ),
        "effective_batch_size": effective_batch_size,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": config[
                "bnb_4bit_quant_type"
            ],
            "double_quantization": config[
                "bnb_4bit_use_double_quant"
            ],
            "compute_dtype": str(
                compute_dtype
            ).replace("torch.", ""),
        },
        "parameters": parameters,
        "training_metrics": json_safe(
            train_result.metrics
        ),
        "loss_history": json_safe(
            trainer.state.log_history
        ),
        "measured_training_seconds": round(
            measured_training_seconds,
            6,
        ),
        "projection": projection,
        "memory": {
            "gpu_name": torch.cuda.get_device_name(0),
            "model_footprint_gib": round(
                trainer.model.get_memory_footprint()
                / (1024**3),
                6,
            ),
            "peak_cuda_memory_gib": round(
                torch.cuda.max_memory_allocated()
                / (1024**3),
                6,
            ),
        },
        "adapter": {
            "path": str(adapter_dir),
            "size_mib": round(
                directory_size_bytes(adapter_dir)
                / (1024**2),
                6,
            ),
        },
    }

    metrics_dir = (
        SMOKE_LOGS_DIR
        if smoke
        else METRICS_DIR
    )
    metrics_path = metrics_dir / f"{run_name}.json"

    write_metrics(metrics_path, result)

    print(f"Saved adapter: {adapter_dir}")
    print(f"Saved training report: {metrics_path}")
    print(
        json.dumps(
            {
                "training_metrics": result[
                    "training_metrics"
                ],
                "projection": projection,
                "memory": result["memory"],
                "adapter": result["adapter"],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI"""

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Qwen for CLINC150 "
            "intent classification using QLoRA."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run 10 optimizer steps on "
            "128 shuffled training examples."
        ),
    )

    return parser


def main() -> None:
    """Run training"""

    args = build_parser().parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
