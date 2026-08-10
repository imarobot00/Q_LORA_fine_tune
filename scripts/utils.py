"""Shared utilities for the RF-QLoRA project.

Kept dependency-light: heavy libraries (torch, transformers) are imported lazily
inside the functions that need them, so this module can be imported in a
CPU-only / minimal environment (e.g. by scripts/00_inspect_data.py).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert RF, DSP, and wireless communications engineer. "
    "Answer precisely and show the key steps. When code is required, return "
    "correct, runnable Python."
)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 3407) -> int:
    """Seed Python, NumPy and (if available) Torch. Returns the seed used."""
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #
def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file (one JSON object per line). Blank lines are skipped."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    """Write an iterable of dicts to a JSONL file (creates parent dirs)."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Chat / prompt formatting
# --------------------------------------------------------------------------- #
def _build_response(record: dict, task_type: str) -> str:
    """Render the assistant turn from a raw dataset record."""
    if task_type == "code":
        out = (record.get("output") or "").strip()
        return f"```python\n{out}\n```" if out else ""

    if task_type == "numeric":
        parts: list[str] = []
        reasoning = (record.get("reasoning") or "").strip()
        if reasoning:
            parts.append(reasoning)
        final_answer = record.get("final_answer", None)
        unit = (record.get("unit") or "").strip()
        if final_answer is not None:
            suffix = f" {unit}" if unit else ""
            parts.append(f"**Final answer:** {final_answer}{suffix}".rstrip())
        code = (record.get("solver_code") or "").strip()
        if code:
            parts.append(f"```python\n{code}\n```")
        return "\n\n".join(parts)

    # fallback for any other/unknown task type
    return (record.get("output") or record.get("solver_code") or "").strip()


def record_to_messages(record: dict, include_response: bool = True) -> list[dict]:
    """Convert a raw dataset record into a chat `messages` list.

    Works for both `code` and `numeric` task types. Set include_response=False
    to build an inference prompt (system + user only).
    """
    task_type = record.get("task_type", "")
    user = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()
    if input_text:
        user += "\n\nInput:\n" + input_text

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if include_response:
        messages.append({"role": "assistant", "content": _build_response(record, task_type)})
    return messages


def messages_to_text(messages: list[dict]) -> str:
    """Flatten messages into plain text (used for tokenizer-free length estimates)."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


# --------------------------------------------------------------------------- #
# Tokenizer helpers (optional — degrade gracefully if transformers is absent)
# --------------------------------------------------------------------------- #
def try_load_tokenizer(model_name: str = DEFAULT_MODEL, verbose: bool = True) -> Optional[Any]:
    """Load a HF tokenizer, returning None if transformers is missing or offline."""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # transformers not installed
        if verbose:
            print(f"[utils] transformers unavailable ({exc}); token counts will be approximate.")
        return None
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:  # no network / not cached
        if verbose:
            print(f"[utils] could not load tokenizer '{model_name}' ({exc}); using approximate counts.")
        return None


def count_tokens(tokenizer: Any, messages: list[dict]) -> int:
    """Exact token count of a rendered chat conversation.

    Robust across transformers versions: v5 returns a BatchEncoding
    (dict with ``input_ids``), older versions return a flat ``list[int]``.
    """
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(out, "input_ids"):
        ids = out["input_ids"]
    elif isinstance(out, dict):
        ids = out["input_ids"]
    else:
        ids = out
    if len(ids) and isinstance(ids[0], (list, tuple)):  # batched output
        ids = ids[0]
    return len(ids)
