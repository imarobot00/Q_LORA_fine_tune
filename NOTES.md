# RF-QLoRA — Engineering Notes

Running research log. For every major step, record **What / Why / Risk / Verify**.

---

## Week 11 · Day 1 (Monday) — Fundamentals, setup, data inspection

### Environment reality
- **No local GPU** (Fedora, Python 3.13.13, no NVIDIA driver). QLoRA training
  (4-bit bitsandbytes + Unsloth) needs a CUDA GPU, so training will run on an
  external GPU box / Colab / Kaggle. This machine handles data + analysis work.
- `transformers==5.3.0` and `numpy==2.3.3` are available locally → exact Qwen
  tokenization and the full data audit run here without a GPU.

### QLoRA, in my own words
- **What:** freeze the 7B base, quantize it to 4-bit (NF4), and train small LoRA
  adapters (rank-16 low-rank updates) on top. Only the adapters get gradients.
- **Why NF4 + double quant:** NF4's levels are spaced for a normal weight
  distribution (better than int4 for the same 4 bits); double quantization also
  quantizes the block scales, saving ~0.4 bits/param. Matmuls upcast to bf16.
- **Why it fits one GPU:** 7B in 4-bit ≈ 4–5 GB; the adapter is ~tens of MB.

### Step log (What / Why / Risk / Verify)
1. **Repo scaffold + configs** — dir tree, `configs/*.yaml`, Apache-2.0 LICENSE
   (matches Qwen2.5). *Why:* config-driven runs → comparable experiments.
   *Risk:* config/code drift. *Verify:* files exist; YAML parses.
2. **`utils.py`** — seeding, JSONL IO, chat formatting, tokenizer helper.
   *Why:* one code path for every script. *Risk:* heavy imports breaking CPU use
   → mitigated by lazy torch/transformers imports. *Verify:* inspection imports it fine.
3. **Tokenizer token-count bug** — `apply_chat_template(tokenize=True)` returns a
   `BatchEncoding` in transformers 5.x, so `len()` counted 2 keys, not tokens.
   *Fix:* read `out["input_ids"]`. *Verify:* full-example p95 went 2 → 321 (sane).
4. **`00_inspect_data.py`** — ran on the real 638 records; findings below.

### Data inspection findings (`results/data_inspection.json`)
- **638 records**, 0 malformed. 411 numeric (64%), 227 code (36%). All
  `source=llm`, all `verified=true`, all required fields present.
- **Duplication is the headline problem:**
  - **Only 88 UNIQUE instruction texts** across 638 records.
  - 42 exact-duplicate instruction groups (550 redundant records). Of these:
    **6 pure duplicates** (28 records droppable) and **36 same-instruction /
    different-solution** groups (multiple `solver_code`/`output` for one problem).
  - 1,593 near-duplicate pairs (Jaccard ≥ 0.90), 291 records involved.
  - `id` is **not unique** (81 ids reused, e.g. `gen-0007` ×13) → never split or
    dedup on `id`.
- **Domain (`area`) mix is imbalanced:** DSP ~62% (sampling, quantization,
  resampling, filter, windowing, conv, spectral), RF-Fundamentals 16%
  (decibels_power), Modulation 12%, Wireless ~10% (capacity/link_budget/noise).
  **No SDR data at all.**
- **Token lengths (Qwen tokenizer):** full example p50 198 / p95 321 / p99 579 /
  max 896; response-only p95 226 / max 812. → use `max_seq_length = 1024`
  (covers the 896 max; 512 would clip the longest ~1%).

### Implications (feed into Tuesday)
- Effective dataset ≈ **88 unique problems**, not 600 → *extreme* small data.
- **Split by unique instruction**, not by record/`id`; otherwise the same problem
  (different solution) leaks across train/val and inflates validation numbers.
- Dedup policy to decide: (a) one record per unique instruction (~88), or
  (b) keep distinct solutions as light augmentation (~610 records) but STILL
  group-split by instruction. Currently leaning (b) with grouped splitting.
- The 100-question benchmark cannot be a clean 20×5 balance across
  {Modulation, DSP, Wireless, SDR, RF-Fundamentals}: there is no SDR signal and
  Wireless is thin. Either match benchmark domains to what exists, or author
  SDR/Wireless questions that test *base* knowledge (and report them as
  generalization, not learned content).
- Reset expectations on +10pt: with ~88 unique problems, gains may be small or
  format-level. The clean, honest experiment is the deliverable.

### Blocked on GPU (do on the GPU box)
- [ ] Install training stack (`requirements.txt`) → `python scripts/check_env.py`
- [ ] Load `Qwen2.5-7B-Instruct` in 4-bit + one generation (end-to-end path)
