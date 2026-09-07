# Intent Classification with QLoRA

Parameter-efficient fine-tuning of Qwen2.5-1.5B-Instruct for 151-class intent classification on CLINC-OOS Plus. The project compares the same frozen test split across a classical TF-IDF baseline, zero-shot Qwen, and a QLoRA-trained Qwen adapter.

The fine-tuned model achieved **94.30% macro F1** and **91.62% accuracy** on 5,500 held-out examples. It improved macro F1 by **45.16 percentage points over zero-shot Qwen** and **10.12 points over TF-IDF + Logistic Regression**.

## Results

All reported values come from the untouched test split after the prompt, parser, model revision, adapter checkpoint, quantization, and decoding configuration were frozen. Macro F1 is the primary metric because the dataset contains 150 in-scope intents plus an out-of-scope (`oos`) class with different proportions across splits.

| Approach | Accuracy | Macro F1 | In-scope accuracy | OOS precision | OOS recall | OOS F1 | Valid-label rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B zero-shot | 43.07% | 49.14% | 51.93% | 19.63% | 3.20% | 5.50% | 91.82% |
| TF-IDF + Logistic Regression | 80.64% | 84.19% | 88.04% | 81.83% | 47.30% | 59.95% | **100.00%** |
| **Qwen2.5-1.5B + QLoRA** | **91.62%** | **94.30%** | **97.07%** | **98.82%** | **67.10%** | **79.93%** | 99.22% |

QLoRA reduced invalid generated labels from 450 to 43, a **90.4% reduction** relative to zero-shot Qwen. Against the stronger TF-IDF baseline, it gained 10.98 accuracy points, 10.12 macro-F1 points, and 19.98 OOS-F1 points.

## Why this project

General-purpose LLMs are flexible, but zero-shot generation can be unreliable and unnecessarily expensive for a narrow routing problem. Intent classification provides a constrained setting in which fine-tuning quality can be measured directly rather than judged with subjective generation metrics. The comparison includes both the unchanged base model and a strong classical baseline so the improvement and operational cost are visible.

## Dataset

The pipeline downloads [`DeepPavlov/clinc_oos`](https://huggingface.co/datasets/DeepPavlov/clinc_oos) from the Hugging Face Hub using the pinned `plus` configuration and source revision.

| Split | Rows | In-scope rows | OOS rows | Usage |
| --- | ---: | ---: | ---: | --- |
| Train | 15,250 | 15,000 | 250 | Fit TF-IDF and QLoRA parameters |
| Validation | 3,100 | 3,000 | 100 | Prompt, configuration, and checkpoint decisions |
| Test | 5,500 | 4,500 | 1,000 | One final frozen evaluation |

The 151 canonical labels consist of 150 in-scope intents and `oos`. The data audit also checks split sizes, label distributions, empty text, normalized duplicates, and cross-split overlap before modeling.

## Experimental design

### Baselines

- **TF-IDF + Logistic Regression:** word unigrams and bigrams, sublinear term frequency, `min_df=2`, L2-regularized Logistic Regression.
- **Zero-shot Qwen:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) receives the full canonical label set and generates one label using greedy decoding.

### QLoRA fine-tuning

- Base model: Qwen2.5-1.5B-Instruct at a pinned revision
- Training examples: 15,250
- Objective: completion-only causal-language-model loss on the canonical label
- Quantization: 4-bit NF4 with double quantization and bfloat16 computation
- LoRA: rank 16, alpha 32, dropout 0.05, all linear layers
- Optimizer: paged AdamW 8-bit
- Schedule: one epoch, cosine decay, 60 warmup steps, learning rate `2e-4`
- Effective batch size: 8
- Maximum sequence length: 1,024 tokens

Only **18,464,768 parameters (1.182%)** were trainable. The resulting adapter is **35.27 MiB**.

## Evaluation protocol

The comparison uses the same label order, prompt, greedy decoder, strict parser, seed, and examples for both Qwen approaches. A generated output is valid only if stripping surrounding whitespace produces one exact canonical label; invalid outputs count as incorrect.

The workflow deliberately separates model development from final reporting:

1. Train models only on `train`.
2. Make prompt and configuration decisions using `validation`.
3. Freeze the adapter SHA-256 and evaluation settings in [`configs/evaluation.json`](configs/evaluation.json).
4. Commit the frozen configuration before accessing `test`.
5. Evaluate every approach once and generate the comparison artifact.

Metrics include overall accuracy, macro F1 across all 151 labels, in-scope accuracy, OOS precision/recall/F1, in-scope false-rejection rate, and valid-label rate.

## Efficiency

Training took **2.05 GPU-hours** on an NVIDIA RTX 4070. Peak CUDA memory during training was **3.24 GiB**; QLoRA evaluation peaked at **2.27 GiB**.

| Approach | Mean inference latency | Measurement device |
| --- | ---: | --- |
| TF-IDF + Logistic Regression | 0.029 ms/example | CPU |
| Qwen zero-shot | 110.03 ms/example | RTX 4070, batch size 8 |
| Qwen + QLoRA | 161.18 ms/example | RTX 4070, batch size 8 |

The Qwen latency measurements are directly comparable because they used the same GPU and batch size. TF-IDF was measured on CPU, so its latency is included as an operational reference rather than a hardware-controlled comparison. QLoRA delivers the best classification quality, but TF-IDF remains much cheaper and may be the better production choice when latency dominates the quality requirement.

## Reproduce the pipeline

### 1. Create the environment

Python 3.12 and an NVIDIA CUDA environment are recommended for the Qwen experiments. Install a compatible PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the pinned project dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-gpu.txt
```

Dataset preparation and TF-IDF can run without a GPU using `requirements.txt`.

### 2. Prepare and audit the Hugging Face dataset

```bash
python -m src.data
```

Processed data is written under `data/processed/`; small audit and environment reports are written under `artifacts/metrics/`.

### 3. Run development baselines

```bash
python -m src.baseline_tfidf --split validation
python -m src.baseline_zero_shot --split validation
```

### 4. Audit training examples and run QLoRA

```bash
python -m src.training_data
python -m src.train_qlora --smoke
python -m src.train_qlora
python -m src.evaluate_qlora --split validation
```

Generated adapters, checkpoints, predictions, processed data, and smoke logs are intentionally excluded from Git. The committed metric artifacts record the reported run. A newly trained adapter should first be evaluated on validation and assigned a new frozen checksum before test is accessed.

### 5. Run a new frozen final evaluation

After selecting the adapter and freezing `configs/evaluation.json`:

```bash
python -m src.baseline_tfidf --split test
python -m src.baseline_zero_shot --split test
python -m src.evaluate_qlora --split test
python -m src.compare_results --split test
```

Do not change prompts, parsing, hyperparameters, or checkpoint selection after viewing test results.

## Repository layout

```text
configs/
  qlora.json                 QLoRA hyperparameters
  evaluation.json            Frozen final-evaluation manifest
src/
  data.py                    Hugging Face ingestion and data audit
  prompting.py               Shared zero-shot and training prompt
  baseline_tfidf.py          Classical baseline
  baseline_zero_shot.py      Base-Qwen evaluation
  training_data.py           SFT formatting and token-length audit
  train_qlora.py             Quantized LoRA training
  evaluate_qlora.py          Adapter evaluation
  evaluation.py              Shared metrics
  frozen_evaluation.py       Frozen-config and checksum validation
  compare_results.py         Three-way benchmark comparison
artifacts/metrics/            Tracked reports and final metrics
```

## Limitations

- The experiment uses one dataset, one base model, one seed, and one full training run; it does not estimate variance across seeds.
- OOS recall is **67.1%**, meaning 329 of 1,000 test OOS requests were still assigned an in-scope label.
- QLoRA produced 43 invalid labels and was 1.46 times slower than zero-shot inference in this setup.
- Passing all 151 labels in every prompt produces roughly 940-token prompts, which is inefficient for a routing system.
- The upstream fixed splits contain two exact normalized train-test text overlaps with conflicting labels. They do not create favorable label leakage, but they demonstrate dataset-label ambiguity.
- This is an evaluation-focused portfolio project, not a production service or a claim that an LLM is always preferable to a classical classifier.

## Possible extensions

- Replace free-form generation with constrained decoding or label scoring.
- Retrieve a smaller candidate-label set before classification to reduce prompt length.
- Improve OOS detection with a calibrated confidence threshold and a validation-only tuning protocol.
- Repeat the experiment across multiple seeds and another intent dataset.

## Key artifacts

- [`comparison__test__seed-42.json`](artifacts/metrics/comparison__test__seed-42.json): final three-way test comparison
- [`qwen_qlora__seed-42.json`](artifacts/metrics/qwen_qlora__seed-42.json): training configuration, loss history, memory, and parameter counts
- [`evaluation_contract.json`](artifacts/metrics/evaluation_contract.json): metric and output contract
- [`data_report.json`](artifacts/metrics/data_report.json): source revision, split statistics, and duplicate audit
- [`environment.json`](artifacts/metrics/environment.json): exact package and CUDA environment