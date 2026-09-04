from __future__ import annotations

from typing import Any

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """## 1. Role / Persona

You are a deterministic intent classification system with expertise in mapping short user requests to a fixed intent taxonomy. You are not a conversational assistant.

## 2. Context & Background

You will receive:
- ALLOWED_LABELS: the complete set of valid intent labels.
- USER_REQUEST: one user utterance to classify.

The labels are canonical identifiers written in snake_case. The label `oos` means that the request is outside the scope of every other allowed label.

## 3. Task Definition

Determine the meaning of USER_REQUEST and select the single label from ALLOWED_LABELS that most closely represents the user's intent.

## 4. Constraints

- Select exactly one label from ALLOWED_LABELS.
- Copy the selected label exactly as written.
- Never invent, rename, paraphrase, combine, or correct labels.
- Prefer the most specific semantically matching in-scope label.
- Use `oos` only when no other allowed label semantically matches the request.
- Do not select `oos` merely because the request is ambiguous or informal.
- Do not provide reasoning, explanations, confidence scores, punctuation, quotes, or formatting.

## 5. Output Format

Return exactly one line containing one canonical label copied verbatim from ALLOWED_LABELS."""


def build_user_content(text: str, labels: list[str]) -> str:
    """Build the dynamic user portion of the classification prompt."""

    label_block = "\n".join(f"- {label}" for label in labels)

    return (
        f"ALLOWED_LABELS:\n{label_block}\n\n"
        f"USER_REQUEST:\n{text}\n\n"
        "Return one exact label from ALLOWED_LABELS:"
    )


def build_prompt_messages(
    text: str,
    labels: list[str],
) -> list[dict[str, str]]:
    """Build the system and user messages shared by all Qwen experiments."""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_content(text, labels),
        },
    ]


def build_prompt(
    text: str,
    labels: list[str],
    tokenizer: Any,
) -> str:
    """Render an inference prompt using Qwen's chat template"""

    return tokenizer.apply_chat_template(
        build_prompt_messages(text, labels),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_training_example(
    text: str,
    label: str,
    labels: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Build one conversational prompt-completion training example"""

    if label not in labels:
        raise ValueError(f"Unknown training label: {label!r}")

    return {
        "prompt": build_prompt_messages(text, labels),
        "completion": [
            {
                "role": "assistant",
                "content": label,
            }
        ],
    }