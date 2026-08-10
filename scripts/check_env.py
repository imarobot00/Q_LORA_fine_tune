#!/usr/bin/env python3
"""Environment / GPU verification for the RF-QLoRA training stack.

Run this on the machine that will do the fine-tuning (a CUDA GPU box). It
reports Python, Torch + CUDA availability, bf16 support, and whether the core
training libraries are importable.

Usage:
    python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import platform


def _version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "installed (no __version__)")
    except Exception as exc:  # noqa: BLE001
        return f"NOT INSTALLED ({exc.__class__.__name__})"


def main() -> None:
    print("=" * 60)
    print("ENVIRONMENT CHECK — RF-QLoRA")
    print("=" * 60)
    print(f"python   : {platform.python_version()}  ({platform.platform()})")

    cuda_ok = False
    bf16_ok = False
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        print(f"torch    : {torch.__version__}")
        print(f"cuda avail: {cuda_ok}")
        if cuda_ok:
            print(f"device    : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"vram      : {props.total_memory / 1024**3:.1f} GiB")
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                bf16_ok = False
            print(f"bf16      : {bf16_ok}  (use bfloat16 if True, else float16)")
    except Exception as exc:  # noqa: BLE001
        print(f"torch    : NOT INSTALLED ({exc})")

    print("-" * 60)
    for lib in ("unsloth", "transformers", "trl", "peft", "bitsandbytes",
                "accelerate", "datasets", "huggingface_hub"):
        print(f"{lib:<16}: {_version(lib)}")

    print("=" * 60)
    if not cuda_ok:
        print("RESULT: WARN — no CUDA GPU visible. QLoRA training needs a CUDA GPU")
        print("        (bitsandbytes 4-bit + Unsloth). Use a GPU box or Colab/Kaggle.")
    else:
        print("RESULT: OK — CUDA GPU visible. Ready for QLoRA training.")
        if not bf16_ok:
            print("        Note: bf16 unsupported on this GPU -> set compute dtype float16.")


if __name__ == "__main__":
    main()
