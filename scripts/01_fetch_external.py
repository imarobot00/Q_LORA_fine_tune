#!/usr/bin/env python3
"""Download + convert external EE/physics datasets into this project's JSONL schema.

Path A (keep the RF/EE domain): augment the small local dataset with
  * STEM-AI-mtl/Electrical-engineering  — electrical-engineering Q&A (~1.1k)
  * camel-ai/physics                    — filtered to electromagnetism / optics / waves

Each external row is converted to the same fields the local data uses
(instruction / input / output / id / topic_id / area / task_type / source / verified)
so the rest of the pipeline (and scripts/00_inspect_data.py) treats them uniformly.
External rows are marked `verified: false` and `task_type: "qa"`.

Outputs (default data/external/):
  stem_ee.jsonl, camel_physics_rf.jsonl, external_merged.jsonl
  (+ all_combined.jsonl when --with-raw is given)

Usage:
  python scripts/01_fetch_external.py                       # defaults
  python scripts/01_fetch_external.py --max-physics 1500 --with-raw
  python scripts/00_inspect_data.py --data data/external/external_merged.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# HF's default 10s read timeout is too short on flaky links; raise it before hf imports.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from utils import read_jsonl, set_seed, write_jsonl  # noqa: E402

# Keep only physics rows whose topic/sub-topic looks RF-relevant.
PHYSICS_KEYWORDS = (
    "electromagnet", "optic", "wave", "radiation", "antenna",
    "signal", "oscillat", "interference", "diffraction", "microwave",
)


def _clean(s) -> str:
    return (s or "").strip() if isinstance(s, str) else ""


def convert_stem_ee(max_rows: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("STEM-AI-mtl/Electrical-engineering", split="train")
    rows: list[dict] = []
    for i, r in enumerate(ds):
        # The real question lives in `input`; `instruction` is a generic system prompt.
        question = _clean(r.get("input")) or _clean(r.get("instruction"))
        answer = _clean(r.get("output"))
        if not question or not answer:
            continue
        rows.append({
            "instruction": question,
            "input": "",
            "output": answer,
            "id": f"eeqa-{len(rows):04d}",
            "topic_id": "ee-qa",
            "area": "electrical_engineering",
            "task_type": "qa",
            "source": "STEM-AI-mtl/Electrical-engineering",
            "verified": False,
        })
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def convert_camel_physics(max_rows: int = 2000,
                          keywords: tuple[str, ...] = PHYSICS_KEYWORDS) -> list[dict]:
    import json
    import zipfile

    from huggingface_hub import hf_hub_download

    # camel-ai/physics is a single physics.zip of many JSON files. Download it once
    # with resume (robust to flaky links) and cache it; then filter locally.
    zip_path = hf_hub_download(repo_id="camel-ai/physics", filename="physics.zip",
                               repo_type="dataset")
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            try:
                with zf.open(name) as f:
                    obj = json.load(f)
            except Exception:
                continue
            for r in (obj if isinstance(obj, list) else [obj]):
                topic = _clean(r.get("topic;") or r.get("topic")).lower()
                sub = _clean(r.get("sub_topic")).lower()
                if not any(k in (topic + " " + sub) for k in keywords):
                    continue
                q = _clean(r.get("message_1"))
                a = _clean(r.get("message_2"))
                if not q or not a:
                    continue
                rows.append({
                    "instruction": q,
                    "input": "",
                    "output": a,
                    "id": f"phys-{len(rows):04d}",
                    "topic_id": _clean(r.get("sub_topic")) or "physics",
                    "area": "physics_" + (topic.split()[0] if topic else "general"),
                    "task_type": "qa",
                    "source": "camel-ai/physics",
                    "verified": False,
                })
                if max_rows and len(rows) >= max_rows:
                    return rows
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + convert external EE/physics datasets.")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "external"))
    ap.add_argument("--max-physics", type=int, default=2000, help="max RF-relevant physics rows")
    ap.add_argument("--max-ee", type=int, default=None, help="max EE rows (default: all)")
    ap.add_argument("--skip-physics", action="store_true",
                    help="skip camel-ai/physics (use if that download is unreliable)")
    ap.add_argument("--with-raw", action="store_true",
                    help="also write all_combined.jsonl = local raw data + external")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import datasets  # noqa: F401
    except Exception:
        sys.exit("ERROR: the `datasets` library is required. Install with:\n"
                 "  uv pip install datasets   (or: pip install datasets)")

    print("Downloading STEM-AI-mtl/Electrical-engineering ...")
    ee = convert_stem_ee(args.max_ee)
    write_jsonl(str(out / "stem_ee.jsonl"), ee)
    print(f"  -> {len(ee):>5} rows  ->  {out / 'stem_ee.jsonl'}")

    print(f"Downloading camel-ai/physics (filtered to RF topics, max {args.max_physics}) ...")
    if args.skip_physics:
        print("  skipped (--skip-physics).")
        phys: list[dict] = []
    else:
        try:
            phys = convert_camel_physics(args.max_physics)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: physics fetch failed ({type(exc).__name__}: {str(exc)[:160]}).")
            print("  Continuing with EE only — re-run later, or pass --skip-physics.")
            phys = []
    write_jsonl(str(out / "camel_physics_rf.jsonl"), phys)
    print(f"  -> {len(phys):>5} rows  ->  {out / 'camel_physics_rf.jsonl'}")

    merged = ee + phys
    write_jsonl(str(out / "external_merged.jsonl"), merged)
    print(f"  -> {len(merged):>5} rows  ->  {out / 'external_merged.jsonl'}")

    if args.with_raw:
        raw_path = ROOT / "data" / "raw" / "data.json"
        raw = read_jsonl(str(raw_path)) if raw_path.exists() else []
        combined = raw + merged
        write_jsonl(str(out / "all_combined.jsonl"), combined)
        print(f"  -> {len(combined):>5} rows  ->  {out / 'all_combined.jsonl'}"
              f"  (raw {len(raw)} + external {len(merged)})")

    print("\nNext — inspect the result to see the new UNIQUE-instruction count:")
    print(f"  python scripts/00_inspect_data.py --data {out / 'external_merged.jsonl'}")
    if args.with_raw:
        print(f"  python scripts/00_inspect_data.py --data {out / 'all_combined.jsonl'}")


if __name__ == "__main__":
    main()
