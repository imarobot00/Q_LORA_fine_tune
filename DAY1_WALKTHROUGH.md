# Day 1 Walkthrough — What I Did, Explained Simply

This document explains **everything** that happened during the Week 11 Day 1 (Monday)
setup, in plain language. If you feel lost, read this top to bottom once — it goes
step by step and explains every word.

---

## 0. TL;DR (the 30-second version)

- The goal of the whole project: take a smart general AI model and **gently teach it your
  RF/DSP domain**, then prove (fairly) whether it got better.
- Today was **setup + inspecting your data**. No AI was trained yet.
- I built the project folders, wrote helper code, and wrote a script that **examines your
  dataset**.
- **Biggest discovery:** your file has **638 rows, but only 88 are actually unique** — the
  rest are duplicates or near-duplicates. That changes the plan a lot.
- **The fix:** I added two real public datasets (electrical-engineering + physics) and merged
  them with yours, going from **88 unique → 3,207 unique** examples — still in your RF domain.
- Your computer **has no GPU**, so the actual AI training must happen later on a different
  machine (a GPU box, Google Colab, or Kaggle). Everything that *doesn't* need a GPU is done.
- I saved all the work with **git** (a save point). Nothing was uploaded to the internet.

---

## 1. What is this project, in plain English?

You are doing **QLoRA fine-tuning**. Let me break that phrase down with an analogy.

Imagine a brilliant university professor (that's the **base AI model**, `Qwen2.5-7B-Instruct`).
They already know a huge amount, including a lot of math and physics. You are **not**
re-teaching them from scratch. You just want them to get a little sharper and speak in the
style of your RF (radio frequency) domain.

- **Fine-tuning** = giving the professor extra practice on your specific topic.
- **LoRA** = instead of rewriting the professor's entire brain (expensive), you hand them a
  small stack of **sticky notes** with adjustments. The brain stays the same; the sticky
  notes nudge their answers. These sticky notes are tiny and easy to share.
- **QLoRA** = the "Q" is for **Quantized**. To make the professor fit on a normal computer,
  you compress their brain from a giant encyclopedia into a **pocket-sized summary** (4-bit).
  Then you still add the sticky notes on top.

The end product you will publish is just the **sticky notes** (called an "adapter") plus a
report card showing whether they helped.

**Why this matters today:** Everything I built is the workshop and tools to do this safely
and to *prove* it fairly later.

---

## 2. The big picture of what I did today

Think of it like preparing a kitchen before cooking:

1. **Checked what appliances you have** (Is there a GPU? What Python? etc.).
2. **Built the shelves and drawers** (the project folder structure).
3. **Wrote reusable kitchen tools** (helper code in `scripts/utils.py`).
4. **Wrote a "food inspector"** (`scripts/00_inspect_data.py`) to examine your ingredients
   (your dataset).
5. **Ran the inspector**, found a **bug**, fixed it, and ran it again.
6. **Discovered the ingredients were mostly duplicates** (the 88-vs-638 finding).
7. **Set up a clean, separate toolbox** (a Python "virtual environment").
8. **Hit one more small snag** (a missing library called `jinja2`), fixed it.
9. **Created a save point** with git.
10. **After inspecting**, I realized 88 unique was too few, so I **downloaded two real public
    datasets** and merged them in, growing it to ~3,200 unique examples (see Section 7).

No cooking (AI training) yet — that needs the GPU oven you don't have here.

---

## 3. Step by step: exactly what happened

### Step 1 — I checked your computer

I ran one command that asked your system a bunch of questions. Here is what it told us:

| Question | Answer | What it means for us |
|---|---|---|
| Operating system | Linux (Fedora) | Fine. |
| Python version | 3.13.13 | Works for data tools; training libs prefer 3.10–3.11. |
| `uv` installed? | Yes (0.11.18) | `uv` is a fast tool to manage Python packages. |
| **GPU present?** | **No** | **Big deal** — training needs a GPU; this machine can't train. |
| numpy installed? | Yes (2.3.3) | numpy does number-crunching; needed for the inspector. |
| git installed? | Yes | We can make save points. |
| Lines in your data | **638** | Your dataset has 638 rows. |

> During this check, Fedora popped up "Install NVIDIA drivers? [N/y]". I answered **N** (no),
> because installing system GPU drivers on your machine is not something I should do silently,
> and it wouldn't give you a real GPU anyway.

**Why GPU matters:** The compression trick (4-bit) and the training math run on special
hardware called a **GPU** (a graphics card made by NVIDIA). Your machine doesn't have one, so
the *training* step waits until you're on a machine that does.

### Step 2 — I built the folder structure

I created a tidy set of folders so every kind of file has a home. Here's the layout:

```
QLORA_Fine_Tune/
├── data/
│   ├── raw/data.json      <- YOUR original dataset (never edited)
│   ├── processed/         <- cleaned data will go here (Tuesday)
│   └── benchmark/         <- the 100-question test will go here (Tuesday)
├── configs/               <- settings files (knobs for training)
├── scripts/               <- the actual code
├── evaluation/            <- test/grading code (later this week)
├── results/               <- outputs, scores, reports
├── notebooks/             <- scratch/experiment space
├── requirements.txt       <- list of tools needed on the GPU machine
├── requirements-data.txt  <- list of tools needed here (no GPU)
├── LICENSE                <- legal permission file (Apache-2.0)
├── README.md              <- the project's front page
└── NOTES.md               <- my running lab notebook
```

Empty folders can't be saved by git, so I put a tiny placeholder file named `.gitkeep` in each
empty one. That's the only reason those files exist.

### Step 3 — I wrote the helper code (`scripts/utils.py`)

This is a **toolbox** other scripts borrow from, so we don't rewrite the same code everywhere.
It has four kinds of tools (explained simply in [Section 5](#5-the-important-code-explained-simply)):

- Set a "random seed" (makes results repeatable).
- Read and write your data files.
- Turn one data row into a **chat conversation** (system/user/assistant).
- Count how many **tokens** (word-pieces) a conversation is.

### Step 4 — I wrote the data inspector (`scripts/00_inspect_data.py`)

This script **reads your 638 rows and reports on them**: how many, what topics, how long,
and — crucially — **how many are duplicates**. It writes a summary to
`results/data_inspection.json` and prints a readable report.

### Step 5 — I ran it, found a bug, fixed it

The first run produced a **wrong number**: it said every conversation was "2 tokens long,"
which is impossible. That was a **bug** (explained in [Section 5.3](#53-the-bug-i-found-and-fixed-important)).
I investigated, found the cause, fixed it, and re-ran. After the fix the numbers were sensible
(most conversations are ~200 word-pieces; the longest is 896).

### Step 6 — The big discovery

The fixed inspector revealed the **duplication problem** (full details in
[Section 6](#6-what-the-inspection-found-and-why-it-matters)). Short version: **638 rows, but
only 88 truly different questions.**

### Step 7 — I made a clean, separate toolbox (a "virtual environment")

A **virtual environment** (`.venv`) is like a **separate, labeled toolbox** just for this
project, so this project's tools don't clash with the rest of your computer. I created it with
`uv` and installed only the tools needed for the non-GPU work.

### Step 8 — One more small snag (`jinja2`), fixed

Inside the fresh toolbox, the inspector crashed because a small library called **`jinja2`** was
missing. (`jinja2` fills in templates — the chat format uses it.) I added it to the tools list,
made the script **not crash** if it's ever missing again, installed it, and re-ran. Success.

### Step 9 — I saved everything with git

**git** is like save points in a video game. I made one save point ("commit") containing all
21 files, with the message *"project scaffold, CPU data env, and data inspection (Week 11
Day 1)."*

**Important:** I only saved **locally on your computer**. I did **not** upload/push anything to
GitHub — that's your decision to make.

---

## 4. Every file explained (what it is and why it exists)

| File | What it is | Why it exists |
|---|---|---|
| `scripts/utils.py` | Shared toolbox of small functions | So all scripts reuse the same logic |
| `scripts/00_inspect_data.py` | The dataset inspector | To understand your data before using it |
| `scripts/check_env.py` | GPU/tools checker | Run it on the GPU machine to confirm it's ready |
| `configs/model.yaml` | Model + compression settings | The "which model + how to compress" knobs |
| `configs/lora_base.yaml` | Sticky-note (LoRA) settings | Size/strength of the adjustments |
| `configs/train_base.yaml` | Training settings | Speed, repetitions, batch size, etc. |
| `requirements.txt` | Tool list for the **GPU** machine | So training installs cleanly later |
| `requirements-data.txt` | Tool list for **this** machine | So the data work installs cleanly |
| `LICENSE` | Legal permissions (Apache-2.0) | Lets others use your work; matches Qwen2.5 |
| `README.md` | Project front page | First thing people (and future-you) read |
| `NOTES.md` | My lab notebook | Records what/why/risks/checks each day |
| `.gitignore` | "Don't save these" list | Keeps huge/temp files out of git |
| `results/data_inspection.json` | The inspector's saved report | A snapshot of your data's health |
| `data/raw/data.json` | **Your** dataset | The raw material (kept untouched) |

### What are those `configs/*.yaml` files? (the "knobs")

YAML files are just **settings written in plain text**. Instead of burying numbers inside the
code, we keep them here so experiments are easy and fair (change one knob, rerun). Examples:

`configs/model.yaml` (which model + how to compress it):
```yaml
base_model: Qwen/Qwen2.5-7B-Instruct   # the "professor" we start from
load_in_4bit: true                     # compress the brain to 4-bit (fits on a GPU)
bnb_4bit_quant_type: nf4               # a smart 4-bit format tuned for AI weights
max_seq_length: 1024                   # longest conversation we'll handle (fits your data)
```

`configs/lora_base.yaml` (the sticky notes):
```yaml
r: 16              # "thickness" of the sticky notes (more = more capacity, more risk)
lora_alpha: 16     # how strongly the notes are applied
lora_dropout: 0.05 # randomly ignore a few notes while learning (prevents cramming)
```

`configs/train_base.yaml` (how the practice session runs):
```yaml
learning_rate: 2.0e-4   # how big each learning step is
num_train_epochs: 2     # how many times to go through the data
per_device_train_batch_size: 4  # how many examples at once
```

You don't need to memorize these — just know they're **adjustable knobs**, and we chose safe
starting values (not final answers).

---

## 5. The important code, explained simply

I'll show the key pieces and explain each in plain English. (Some are lightly trimmed for
clarity.)

### 5.1 Turning one data row into a chat

Your data rows look like this (one row):
```json
{"instruction": "Convert 20 dBm to watts...", "reasoning": "P(W) = ...",
 "final_answer": 0.1, "solver_code": "print(...)", "task_type": "numeric", "area": "decibels_power"}
```

AI chat models expect a **conversation** with three roles: a **system** message (instructions
to the AI), a **user** message (the question), and an **assistant** message (the answer). This
function builds that:

```python
def record_to_messages(record, include_response=True):
    user = record.get("instruction", "").strip()          # the question
    if record.get("input"):                                # extra input, if any
        user += "\n\nInput:\n" + record["input"]
    messages = [
        {"role": "system",    "content": DEFAULT_SYSTEM_PROMPT},   # "You are an RF/DSP expert..."
        {"role": "user",      "content": user},                    # the question
    ]
    if include_response:
        messages.append({"role": "assistant", "content": _build_response(record, ...)})  # the answer
    return messages
```

In words: **take a row, produce a 3-part conversation** (expert instructions → question →
answer). The answer is built differently for "code" rows (wrap the code) vs "numeric" rows
(reasoning + final answer + code).

### 5.2 Counting "tokens" (word-pieces)

AI models don't read words; they read **tokens** — little chunks of text (roughly ¾ of a word
each). We count tokens so we know how long conversations are, which decides a setting called
`max_seq_length`.

```python
def count_tokens(tokenizer, messages):
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    # newer libraries return a bundle (dict); older ones return a plain list.
    if hasattr(out, "input_ids") or isinstance(out, dict):
        ids = out["input_ids"]   # pull the actual token numbers out of the bundle
    else:
        ids = out
    return len(ids)              # how many tokens
```

The `tokenizer` is the tool that chops text into tokens. `apply_chat_template` formats the
conversation the exact way Qwen expects, then turns it into token numbers.

### 5.3 The bug I found and fixed (important!)

**Symptom:** the inspector said every conversation was **2 tokens** long. Impossible.

**Cause:** Newer versions of the `transformers` library changed what
`apply_chat_template(tokenize=True)` gives back. It used to return a **plain list of tokens**
(so `len()` = number of tokens). Now it returns a **bundle** that looks like:
```python
{"input_ids": [151644, 8948, 198, ...], "attention_mask": [1, 1, 1, ...]}
```
That bundle has exactly **2 labels** (`input_ids` and `attention_mask`). My old code did
`len(bundle)`, which counted the **2 labels** instead of the tokens inside. Hence "2".

**Fix:** reach inside the bundle and count `input_ids` instead (that's the `ids = out["input_ids"]`
line above). After the fix, the longest conversation correctly showed **896 tokens**, not 2.

**Lesson (worth remembering):** when a number looks obviously wrong, don't trust it — open it up
and check what the tool actually returned.

### 5.4 Finding duplicates

To find near-duplicate questions, the script turns each question into a **set of words** and
measures overlap using a simple formula called **Jaccard similarity**:

```python
def jaccard(a, b):                 # a and b are sets of words
    return len(a & b) / len(a | b) # (shared words) / (total distinct words)
```

- `a & b` = words in **both** questions (overlap).
- `a | b` = words in **either** question (total).
- Result is between 0 (nothing in common) and 1 (identical words).

If two questions score **0.90 or higher**, we flag them as near-duplicates. That's how we found
1,593 near-duplicate pairs.

---

## 6. What the inspection found (and why it matters)

Here are the real numbers from your dataset:

| What we measured | Result |
|---|---|
| Total rows | **638** |
| **Truly unique questions** | **88** |
| Exact copies that are 100% identical (safe to delete) | 28 rows |
| Same question but a *different* solution | 36 groups |
| Near-duplicate pairs (≥90% word overlap) | 1,593 |
| Reused ID labels | 81 IDs repeat (so the `id` field is **not** a reliable unique key) |
| Topic balance | DSP ~62%, RF-Fundamentals 16%, Modulation 12%, Wireless ~10%, **SDR 0%** |
| Conversation length | usually ~200 tokens; longest 896 → we set `max_seq_length = 1024` |

**Why this is the most important finding of the day:**

1. **You have ~88 unique problems, not 600.** That is *very* little data. It means we must be
   extra careful to avoid the AI just **memorizing** answers instead of actually learning. Our
   whole plan leans toward "learn the pattern," not "cram the answers."

2. **The same question sometimes appears in slightly different forms.** If a question ends up in
   both the "study set" and the "test set," the AI has effectively **seen the exam answers** —
   that's called **data leakage**, and it makes results fake. Tomorrow (Tuesday) we must split
   the data **by unique question**, so no question sneaks into both sides.

3. **The `id` labels repeat**, so we can't rely on them to tell rows apart. We'll dedupe by the
   actual question text instead.

4. **There is no "SDR" data at all**, and some topics are thin. So the 100-question test can't be
   perfectly balanced across all 5 topics — we'll adjust the test to match reality.

None of this is bad news — it's exactly what a data inspection is for. Better to know now than
to get fake results later.

---

## 7. The fix: I added real datasets

88 unique questions is far too few to fine-tune on. Instead of giving up on your RF domain, I
**added two real, public datasets** that fit it and merged them with your data.

### What I added

| Source | What it is | Rows kept |
|---|---|---|
| Your original data | RF/DSP code + numeric problems | 638 |
| `STEM-AI-mtl/Electrical-engineering` | Electrical-engineering Q&A | 1,131 |
| `camel-ai/physics` (filtered) | Physics Q&A — kept only electromagnetism / optics / waves | 2,000 |

### The new numbers

| Before | After |
|---|---|
| 638 rows, **88 unique** | 3,769 rows, **3,207 unique** |

That's roughly a **36x jump** in unique training material — and it's all still in the
RF / electrical-engineering / physics family, so your Week 11 plan still holds.

### How I did it (one script)

I wrote `scripts/01_fetch_external.py`. It:

1. Downloads both datasets from Hugging Face (a public library of datasets).
2. Converts them into **your exact format** (`instruction / input / output / ...`), and marks
   them `verified: false` and `task_type: "qa"` so you can always tell them apart from your own.
3. Saves them in `data/external/`, including a combined file `all_combined.jsonl`.

You can run it yourself anytime:

```bash
uv pip install -r requirements-data.txt
.venv/bin/python scripts/01_fetch_external.py --with-raw
```

Handy flags: `--max-physics 1500` (how much physics to keep), `--skip-physics` (if that one
download misbehaves).

### One hiccup (and how I fixed it)

The physics dataset is one big `physics.zip`, and your internet kept **timing out** mid-download
(you saw lots of "read operation timed out" and "connection reset" messages). I changed the
script to **download the whole file once with resume** — it picks up where it left off instead
of restarting — with a longer timeout, and made a physics failure **non-fatal** (you always keep
the EE data). After that it downloaded cleanly.

### Quick note on git (you asked earlier)

Your `scripts/` and `configs/` folders are **tracked and already in your GitHub repo** — they
are **not** ignored. The only things ignored are `instructions.md`, `.venv/`, caches, model
weights, and now `data/external/` (because the script can regenerate it). So nothing important
is missing.

---

## 8. What's blocked (no GPU) and what to do about it

Two Monday tasks need a GPU and are therefore **paused** (not skipped):

- Loading the compressed `Qwen2.5-7B-Instruct` model.
- Running one test generation to confirm it works.

When you get onto a machine with an NVIDIA GPU (your own, **Google Colab**, or **Kaggle**), do:

```bash
# 1) install the training tools (see the notes at the top of requirements.txt)
# 2) confirm the GPU is visible:
python scripts/check_env.py     # should say: "CUDA GPU visible"
```

`check_env.py` is the little script I wrote for exactly this — it prints your GPU name, memory,
and whether all the training libraries are installed.

---

## 9. Glossary (plain-English definitions)

- **Model / base model:** the pre-trained AI we start from (`Qwen2.5-7B-Instruct`). "7B" = 7
  billion internal numbers ("parameters").
- **Fine-tuning:** giving the model extra practice on your topic.
- **LoRA / adapter:** small "sticky note" adjustments trained on top of the frozen model.
- **Quantization / 4-bit / NF4:** compressing the model so it fits on a GPU with little quality loss.
- **QLoRA:** LoRA on top of a 4-bit-compressed model.
- **Token:** a chunk of text (~¾ of a word) that the model actually reads.
- **Tokenizer:** the tool that splits text into tokens.
- **`max_seq_length`:** the longest conversation (in tokens) we allow.
- **JSONL:** a file with **one JSON record per line** (your `data.json` is this format).
- **Overfitting / memorizing:** when the model learns exact answers instead of the general idea — bad.
- **Data leakage:** when test questions accidentally appear in the study data — makes scores fake.
- **Duplicate / near-duplicate:** identical / almost-identical rows.
- **Virtual environment (`.venv`):** a separate toolbox of Python packages for this project only.
- **`uv` / `pip`:** tools that install Python packages.
- **git / commit:** version control / a save point of your files.
- **GPU:** the graphics-card hardware needed to train AI efficiently.
- **YAML:** a plain-text settings format (the `configs/*.yaml` files).
- **`jinja2`:** a template-filler library the chat formatter needs.

---

## 10. What's next (Tuesday)

Now that you have ~3,200 unique examples (RF/DSP + EE + physics), the plan continues:

1. **Decide the domain mix:** 638 RF/DSP + 1,131 EE + 2,000 physics. Choose how much of each to
   keep so the model stays RF-focused (physics can easily dominate if left unchecked).
2. **Clean + dedupe** the combined data (drop exact and near-duplicates).
3. **Split by unique question** into a study set (train) and a check set (validation) — with
   **no leakage**.
4. **Hand-write the 100-question multiple-choice test**, matched to the topics you keep.
5. **Prove there's no overlap** between the test and the training data.

Everything so far was about building a trustworthy foundation — and getting enough real data —
so the eventual training and grading are **fair and believable**. The point isn't just to get a
number, but to get a number you can **defend**.

---

*If any part of this is still confusing, tell me which section and I'll re-explain it even more
simply.*
