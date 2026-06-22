# Supermarket Operations Assistant — Student Starter

## Your Task

You have been given a working RAG chatbot for a supermarket supply-chain assistant. Your job is to observe how the system behaves, modify it, and then recover the original behavior using version control.

By the end of this exercise you will have demonstrated that **prompts are part of the system state** — changing the prompt changes the answer, even when the data and model stay the same.

---

## What You Have Been Given

A fully working pipeline with:

- A supermarket inventory dataset (`data/inventory.csv`)
- A system prompt defining the assistant's rules (`prompts/base_rules.txt`)
- A vector index built from the inventory (`artifacts/chroma_db/`)
- A RAG chatbot (`src/chat.py`)
- A DVC pipeline that rebuilds the prompt artifact and vector index whenever a dependency changes (`dvc.yaml`)

The repository is already at tag `inventory-v1`. Do not modify the inventory data.

---

## Setup

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Initialise the local git repo and create the `inventory-v1` tag.** Required only if you plan to do the rollback parts (Parts 3 and 4). Pick one option:

*Option A — Run the helper script (no Git account needed):*

```bash
bash setup.sh
```

Sets a repo-local `user.email` and `user.name`, makes the baseline commit, and creates the `inventory-v1` tag. Re-running it is a no-op.

*Option B — Initialise it yourself (uses your own Git identity):*

```bash
git init
git add .
git commit -m "Baseline (inventory-v1)"
git tag inventory-v1
```

Requires Git configured with your name and email (globally via `git config --global` or signed in to your environment).

**Confirm the pipeline is up to date:**

```bash
dvc repro
```

If DVC reports nothing to reproduce, you are ready.

---

## Part 1 — Run the System and Record the Baseline Answer

Start the chatbot:

```bash
python src/chat.py
```

Ask this question exactly:

```
Should we reorder milk?
```

**Write down the full answer you receive.** You will need it later.

When you are done, type `exit` to quit.

---

## Part 2 — Change the Prompt

Open `prompts/base_rules.txt` in a text editor.

Replace the contents with the following:

```
You are a supermarket inventory assistant.

Rules:
- Answer using only the provided data.
- Do not provide recommendations or operational advice.
- Only report facts from the inventory data.
```

Save the file, then rebuild the pipeline:

```bash
dvc repro
```

Start the chatbot again and ask the same question:

```bash
python src/chat.py
```

```
Should we reorder milk?
```

**Write down this new answer.**

> **Question to answer:** What is different about the two answers? What caused the change?

Commit your changes:

```bash
git add prompts/base_rules.txt artifacts/system_prompt.txt
git commit -m "Prompt v2: remove operational recommendations"
git tag prompt-v2
```

---

## Part 3 — Revert and Reproduce the Original Answer

You must now recover the original system behavior using version control.

Restore the earlier state:

```bash
git checkout inventory-v1
dvc checkout --force
dvc repro
```

Start the chatbot and ask the same question again:

```bash
python src/chat.py
```

```
Should we reorder milk?
```

> **Question to answer:** Does this answer match what you recorded in Part 1? Why or why not?

---

## Part 4 — Trace the Lineage

You are given one of these two answers:

```
(A) Milk inventory is below the reorder point.
    You should place a reorder with supplier FarmFresh.
    Lead time is 2 days.

(B) Milk inventory is below the reorder point.
    Supplier: FarmFresh. Lead time: 2 days.
```

Without running the chatbot, determine which prompt version produced the answer you were given.

Use these commands to investigate:

```bash
# See the commit history for the prompt file
git log prompts/base_rules.txt

# Compare what changed between the two versions
dvc diff inventory-v1 prompt-v2

# Inspect the recorded file hashes from the last pipeline run
cat dvc.lock
```

**Write up your reasoning.** You should be able to state:

- Which prompt version was active
- How you know — what evidence in Git or DVC supports your conclusion

---

## Deliverables

Submit the following:

1. **The two answers** you recorded in Parts 1 and 2, with a one-paragraph explanation of what changed and why.
2. **A confirmation** that Part 3 succeeded — the original answer was reproduced after rollback.
3. **Your lineage trace** from Part 4 — which prompt version produced the given answer, and the evidence you used to conclude this.

---

## Hints

- `dvc repro` only reruns stages whose dependencies have changed. If you edit `base_rules.txt`, it will rerun `build_prompt` and then `build_index`.
- `dvc checkout` restores DVC-tracked files (like the vector index) to match the current Git commit. Always run it after `git checkout`.
- The generated system prompt lives at `artifacts/system_prompt.txt`. You can read it directly to confirm which rules are active before running the chatbot.
- If something looks wrong, run `dvc status` to see what DVC thinks is out of date.

---

## Key Commands

| Command | What it does |
|---|---|
| `dvc repro` | Rebuild any stages whose inputs have changed |
| `dvc checkout --force` | Restore DVC-tracked outputs to match the current commit |
| `dvc status` | Show which stages are out of date |
| `dvc diff <tag1> <tag2>` | Show file-level changes between two versions |
| `git log prompts/base_rules.txt` | Show commit history for the prompt file |
| `cat dvc.lock` | Inspect recorded file hashes for the last pipeline run |
| `cat artifacts/system_prompt.txt` | Read the currently active system prompt |
