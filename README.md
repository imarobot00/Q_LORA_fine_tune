# RF-QLoRA — Domain Fine-Tuning of a 7B LLM for RF / DSP

QLoRA fine-tuning of **Qwen2.5-7B-Instruct** on a small RF / DSP instruction
dataset, with a rigorous, leakage-audited evaluation. This is a *research-grade*
experiment: the deliverable is a **valid, reproducible** base-vs-fine-tuned
comparison — not a target number hit by any means necessary.

> **Status:** Week 11 in progress. Day 1 (project setup + data inspection) complete.
> Training runs on an external CUDA GPU (this dev machine has no GPU).

## Why this is a small-data project (measured, not assumed)

An automated audit of the raw data (`scripts/00_inspect_data.py` →
`results/data_inspection.json`) shows the "~600 samples" are heavily duplicated:

| Metric | Value |
|---|---|
| Raw records | 638 |
| **Unique instruction texts** | **88** |
| Pure-duplicate records (safe to drop) | 28 |
| Same-instruction / different-solution groups | 36 |
| Near-duplicate pairs (Jaccard ≥ 0.90) | 1,593 (291 records) |
| Reused ids | 81 ids across 622 records (⇒ `id` is **not** unique) |
| Domain mix | DSP ~62% · RF-Fundamentals 16% · Modulation 12% · Wireless ~10% · **SDR 0%** |
| Token length (full example) | p50 198 · p95 321 · p99 579 · max 896 |

So the effective dataset is **~88 unique problems**. The strategy therefore
prioritizes **generalization over memorization**: strong regularization, few
epochs, leakage-safe *group-wise* splitting (by unique instruction), and honest
reporting of whatever improvement is actually measured.

## Project structure

```
.
├── data/
│   ├── raw/data.json            # source dataset (638 JSONL records, read-only)
│   ├── processed/               # cleaned + chat-formatted train/val (Tue)
│   └── benchmark/               # 100 hand-built held-out RF MCQs (Tue)
├── configs/
│   ├── model.yaml               # base model + 4-bit NF4 quantization
│   ├── lora_base.yaml           # LoRA r/alpha/dropout/target modules
│   └── train_base.yaml          # lr/epochs/batch/accum/warmup (E0)
├── scripts/
│   ├── utils.py                 # seeding, JSONL IO, chat formatting, tokenizer
│   ├── 00_inspect_data.py       # data audit (stats/dupes/token lengths)
│   └── check_env.py             # torch/CUDA/lib verification (run on GPU box)
├── evaluation/                  # MCQ eval + leakage audit + comparison (Wed+)
├── results/
│   ├── data_inspection.json     # machine-readable audit snapshot
│   ├── baseline/                # base-model eval (Wed)
│   ├── experiments/             # per-run config + losses + eval
│   └── plots/                   # loss curves, accuracy bars
├── notebooks/                   # scratch exploration
├── requirements.txt             # GPU training stack (torch/unsloth/bnb/...)
├── requirements-data.txt        # CPU-only data + analysis stack
├── LICENSE                      # Apache-2.0 (matches Qwen2.5)
└── README.md
```

## Setup

### 1) CPU data / analysis environment (this repo's local work)
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements-data.txt
```

### 2) GPU training environment (required for QLoRA — external CUDA box)
Python 3.10–3.11 recommended. Install PyTorch for your CUDA build first, then
Unsloth, then the rest (see the header of `requirements.txt`), and verify:
```bash
python scripts/check_env.py        # expects: CUDA GPU visible
```

## Reproduce so far
```bash
python scripts/00_inspect_data.py  # audit raw data -> results/data_inspection.json
```

## Methodology (7-day plan)
Mon setup + data audit · Tue leakage-safe splits + 100-question MCQ benchmark ·
Wed base-model baseline + eval harness · Thu first QLoRA run · Fri controlled
experiments (rank / lr / epochs) · Sat final base-vs-QLoRA eval + error analysis ·
Sun ship adapter + model card to the Hugging Face Hub.

## License
Apache-2.0 — see [LICENSE](LICENSE). Chosen to match the Qwen2.5 base model.
