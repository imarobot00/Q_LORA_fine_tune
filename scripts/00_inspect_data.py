#!/usr/bin/env python3
"""First-pass audit of the raw RF dataset (Monday, Week 11).

Reports:
  * sample count + malformed-line count
  * task_type and domain (`area`) distribution
  * field / schema completeness
  * exact, normalized and near-duplicate instructions + duplicate ids
  * token-length distribution (true Qwen tokens if `transformers` is installed,
    otherwise a word/char approximation) and a suggested `max_seq_length`

Writes a machine-readable summary to results/data_inspection.json and prints a
human-readable report.

Usage:
    python scripts/00_inspect_data.py
    python scripts/00_inspect_data.py --data data/raw/data.json --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from utils import (  # noqa: E402
    DEFAULT_MODEL,
    count_tokens,
    messages_to_text,
    record_to_messages,
    try_load_tokenizer,
)

# Coarse suggestion mapping from fine-grained `area` tags to benchmark domains.
# This is a STARTING suggestion — review it against the raw `area` counts below.
AREA_TO_DOMAIN = {
    "modulation": "Modulation",
    "filter_design": "DSP",
    "spectral_analysis": "DSP",
    "windowing": "DSP",
    "resampling": "DSP",
    "sampling_theory": "DSP",
    "convolution_correlation": "DSP",
    "quantization": "DSP",
    "link_budget": "Wireless",
    "capacity": "Wireless",
    "noise": "Wireless",
    "decibels_power": "RF Fundamentals",
}

REQUIRED_FIELDS = {
    "code": ["instruction", "output"],
    "numeric": ["instruction", "solver_code"],
}

_NORM_RE = re.compile(r"[^a-z0-9\s]")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_tolerant(path: Path) -> tuple[list[dict], list[tuple[int, str]]]:
    rows: list[dict] = []
    errors: list[tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as exc:
                errors.append((lineno, str(exc)))
    return rows, errors


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = _NORM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return set(normalize(text).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dist(values: list[int]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "mean": round(float(arr.mean()), 1),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def recommend_seq_len(p95: int) -> int:
    for cand in (512, 768, 1024, 1536, 2048, 3072, 4096):
        if cand >= p95:
            return cand
    return 4096


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the raw RF dataset.")
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "data.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--near-dup-threshold", type=float, default=0.90)
    ap.add_argument("--out", default=str(ROOT / "results" / "data_inspection.json"))
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    rows, errors = load_tolerant(data_path)
    n = len(rows)

    section("DATASET OVERVIEW")
    print(f"file:            {data_path}")
    print(f"valid records:   {n}")
    print(f"malformed lines: {len(errors)}")
    for lineno, msg in errors[:5]:
        print(f"  - line {lineno}: {msg}")

    # ----- distributions ---------------------------------------------------- #
    task_types = Counter(r.get("task_type", "<missing>") for r in rows)
    areas = Counter(r.get("area", "<missing>") for r in rows)
    sources = Counter(r.get("source", "<missing>") for r in rows)
    verified = Counter(bool(r.get("verified", False)) for r in rows)
    domains = Counter(AREA_TO_DOMAIN.get(r.get("area", ""), "Other/Unmapped") for r in rows)

    section("TASK TYPE")
    for k, v in task_types.most_common():
        print(f"  {k:<12} {v:>4}  ({v / n:5.1%})")

    section("RAW AREA (fine-grained) — use this as ground truth for domains")
    for k, v in areas.most_common():
        mapped = AREA_TO_DOMAIN.get(k, "Other/Unmapped")
        print(f"  {k:<26} {v:>4}  ({v / n:5.1%})  -> {mapped}")

    section("SUGGESTED BENCHMARK DOMAINS (coarse map — review before trusting)")
    for k, v in domains.most_common():
        print(f"  {k:<18} {v:>4}  ({v / n:5.1%})")
    if domains.get("Other/Unmapped"):
        print("  NOTE: some areas are unmapped; edit AREA_TO_DOMAIN in this script.")
    print("  NOTE: no `area` maps to 'SDR'. Decide how SDR is represented in the")
    print("        100-question benchmark (it may be a cross-cutting category).")

    section("SOURCE / VERIFIED")
    for k, v in sources.most_common():
        print(f"  source={k:<8} {v:>4}")
    for k, v in verified.most_common():
        print(f"  verified={str(k):<5} {v:>4}")

    # ----- field completeness ---------------------------------------------- #
    section("FIELD / SCHEMA COMPLETENESS")
    missing_report: dict[str, int] = Counter()
    empty_instruction = 0
    empty_response = 0
    empty_input = 0
    missing_id = 0
    for r in rows:
        tt = r.get("task_type", "")
        for field in REQUIRED_FIELDS.get(tt, ["instruction"]):
            val = r.get(field, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing_report[f"{tt}:{field}"] += 1
        if not (r.get("instruction") or "").strip():
            empty_instruction += 1
        if not (r.get("input") or "").strip():
            empty_input += 1
        if not r.get("id"):
            missing_id += 1
        resp = record_to_messages(r, include_response=True)[-1]["content"]
        if not resp.strip():
            empty_response += 1

    print(f"  empty instruction:      {empty_instruction}")
    print(f"  empty assistant answer: {empty_response}")
    print(f"  missing id:             {missing_id}")
    print(f"  empty `input` field:    {empty_input}  (expected for most 'code' tasks)")
    if missing_report:
        print("  missing required fields (by task:field):")
        for k, v in missing_report.most_common():
            print(f"    - {k}: {v}")
    else:
        print("  all required per-task fields present.")

    # ----- duplicates ------------------------------------------------------- #
    section("DUPLICATES")
    ids = [r.get("id") for r in rows if r.get("id")]
    dup_ids = {k: v for k, v in Counter(ids).items() if v > 1}
    print(f"  duplicate ids:                {len(dup_ids)} ids reused"
          f" ({sum(dup_ids.values())} records)")
    if dup_ids:
        example = ", ".join(f"{k}×{v}" for k, v in list(dup_ids.items())[:8])
        print(f"    e.g. {example}")

    instructions = [(r.get("instruction") or "").strip() for r in rows]
    exact_groups = {k: v for k, v in Counter(instructions).items() if v > 1 and k}
    exact_dup_records = sum(v - 1 for v in exact_groups.values())
    print(f"  exact-duplicate instructions: {len(exact_groups)} groups,"
          f" {exact_dup_records} redundant records")

    norm_instr = [normalize(x) for x in instructions]
    norm_groups = {k: v for k, v in Counter(norm_instr).items() if v > 1 and k}
    norm_dup_records = sum(v - 1 for v in norm_groups.values())
    print(f"  normalized-duplicate instr.:  {len(norm_groups)} groups,"
          f" {norm_dup_records} redundant records")

    unique_instr = len({x for x in instructions if x})
    print(f"  UNIQUE instruction texts:     {unique_instr}  (of {n} records)")

    # Within exact-duplicate groups: identical answers (pure dup) vs different
    # solutions to the same question (candidate augmentation).
    grp_idx: dict[str, list[int]] = {}
    for idx, instr in enumerate(instructions):
        if instr in exact_groups:
            grp_idx.setdefault(instr, []).append(idx)
    pure_groups = pure_redundant = multi_groups = 0
    for idxs in grp_idx.values():
        responses = {record_to_messages(rows[i], True)[-1]["content"] for i in idxs}
        if len(responses) == 1:
            pure_groups += 1
            pure_redundant += len(idxs) - 1
        else:
            multi_groups += 1
    print(f"    -> {pure_groups} groups are PURE duplicates"
          f" ({pure_redundant} records safe to drop)")
    print(f"    -> {multi_groups} groups: SAME instruction, DIFFERENT solution"
          f" (keep one, or treat extras as augmentation)")

    # near-duplicates (Jaccard on unigram token sets) — O(n^2), fine for ~600
    sets = [token_set(x) for x in instructions]
    thr = args.near_dup_threshold
    near_pairs = 0
    near_items: set[int] = set()
    examples: list[tuple[str, str, float]] = []
    for i in range(n):
        if not sets[i]:
            continue
        for j in range(i + 1, n):
            if not sets[j]:
                continue
            sim = jaccard(sets[i], sets[j])
            if sim >= thr and instructions[i] != instructions[j]:
                near_pairs += 1
                near_items.add(i)
                near_items.add(j)
                if len(examples) < 5:
                    examples.append((rows[i].get("id", f"#{i}"),
                                     rows[j].get("id", f"#{j}"), round(sim, 3)))
    print(f"  near-duplicate pairs (J>={thr:.2f}): {near_pairs}"
          f"  ({len(near_items)} records involved, non-exact)")
    for a, b, sim in examples:
        print(f"    e.g. {a} ~ {b}  (jaccard={sim})")

    # ----- token lengths ---------------------------------------------------- #
    section("TOKEN-LENGTH DISTRIBUTION (full prompt + response)")
    tokenizer = try_load_tokenizer(args.model)
    # Chat templating needs jinja2; if it's missing, fall back gracefully.
    if tokenizer is not None:
        try:
            count_tokens(tokenizer, record_to_messages(rows[0], include_response=True))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] chat-template tokenization unavailable "
                  f"({exc.__class__.__name__}); using approximate counts.")
            tokenizer = None
    method = "qwen-tokenizer" if tokenizer is not None else "approx(words/chars)"
    total_lengths: list[int] = []
    response_lengths: list[int] = []
    for r in rows:
        msgs = record_to_messages(r, include_response=True)
        resp_text = msgs[-1]["content"]
        if tokenizer is not None:
            total_lengths.append(count_tokens(tokenizer, msgs))
            response_lengths.append(len(tokenizer.encode(resp_text)))
        else:
            text = messages_to_text(msgs)
            approx = max(len(text.split()), math.ceil(len(text) / 4))
            total_lengths.append(approx)
            response_lengths.append(max(len(resp_text.split()),
                                        math.ceil(len(resp_text) / 4)))

    total_dist = dist(total_lengths)
    resp_dist = dist(response_lengths)
    print(f"  method: {method}")
    print(f"  full example : {total_dist}")
    print(f"  response only: {resp_dist}")
    rec = recommend_seq_len(total_dist.get("p95", 2048))
    print(f"  -> suggested max_seq_length: {rec}  (>= p95={total_dist.get('p95')})")
    if tokenizer is None:
        print("  WARNING: approximate counts. Re-run on the GPU box with `transformers`")
        print("           installed to get exact Qwen token lengths before training.")

    # ----- write summary ---------------------------------------------------- #
    summary = {
        "data_file": str(data_path),
        "n_records": n,
        "malformed_lines": len(errors),
        "task_types": dict(task_types),
        "areas": dict(areas),
        "suggested_domains": dict(domains),
        "sources": dict(sources),
        "verified": {str(k): v for k, v in verified.items()},
        "completeness": {
            "empty_instruction": empty_instruction,
            "empty_response": empty_response,
            "missing_id": missing_id,
            "empty_input": empty_input,
            "missing_required": dict(missing_report),
        },
        "duplicates": {
            "duplicate_ids": len(dup_ids),
            "duplicate_id_records": sum(dup_ids.values()),
            "exact_instruction_groups": len(exact_groups),
            "exact_redundant_records": exact_dup_records,
            "unique_instructions": unique_instr,
            "pure_dup_groups": pure_groups,
            "pure_dup_droppable_records": pure_redundant,
            "same_instruction_diff_solution_groups": multi_groups,
            "normalized_groups": len(norm_groups),
            "normalized_redundant_records": norm_dup_records,
            "near_dup_threshold": thr,
            "near_dup_pairs": near_pairs,
            "near_dup_records_involved": len(near_items),
        },
        "token_length": {
            "method": method,
            "full_example": total_dist,
            "response_only": resp_dist,
            "recommended_max_seq_length": rec,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    section("DONE")
    print(f"summary written to: {out_path}")
    print("Review the raw `area` counts and duplicates before building splits (Tuesday).")


if __name__ == "__main__":
    main()
