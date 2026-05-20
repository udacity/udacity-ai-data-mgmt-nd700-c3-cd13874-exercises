# Exercise: Prompt Versioning with DVC

## Supermarket Operations Assistant — Prompt Lineage Lab

This exercise extends the Supermarket RAG demo to teach a concept most GenAI tutorials skip: **prompts are part of the system state and must be versioned just like data and embeddings**.

You will observe how changing the system prompt changes the model's answers, and then use DVC and Git to revert to a previous prompt version and reproduce the original behavior.

---

## Learning Objective

A GenAI system's output is determined by three elements:

```
DATA  +  PROMPT  +  MODEL / EMBEDDINGS  =  ANSWER
```

Change any one of them, and the answer changes. This exercise focuses on **prompt lineage** — tracing which version of the prompt produced a given output, and recovering earlier behavior through rollback.

---

## Background: What Changes Between Versions

The operations team modifies the assistant's rules between v1 and v2. The inventory data and embeddings stay the same. Only the prompt changes.

### Prompt v1 — Operational assistant

```
You are a supermarket operations assistant.

Rules:
- Answer questions using inventory data.
- Provide supplier and lead time information when relevant.
- If an item is below reorder point, warn about low stock.
- Provide operational recommendations.
```

### Prompt v2 — Facts only, no recommendations

```
You are a supermarket inventory assistant.

Rules:
- Answer using only the provided data.
- Do not provide recommendations or operational advice.
- Only report facts from the inventory data.
```

### The effect on answers

**Question:** `Should we reorder milk?`

| Version | Answer |
|---|---|
| **Prompt v1** | Milk inventory is below the reorder point. You should place a reorder with supplier FarmFresh. Lead time is 2 days. |
| **Prompt v2** | Milk inventory is below the reorder point. Supplier: FarmFresh. Lead time: 2 days. |

Same data. Same embeddings. Different prompt → different answer. The recommendation disappears because the prompt was changed to forbid it.

---

## Tasks

### Task 1 — Run the system with Prompt v1

Make sure you are on the `inventory-v1` tag and run the pipeline:

```bash
git checkout inventory-v1
dvc checkout
dvc repro
python src/chat.py
```

Ask the following question and **record the answer**:

```
Should we reorder milk?
```

---

### Task 2 — Modify the prompt and observe the change

Edit `prompts/base_rules.txt` and replace the contents with the Prompt v2 rules above.

Rebuild the pipeline and ask the same question:

```bash
dvc repro
python src/chat.py
```

```
Should we reorder milk?
```

Note how the answer has changed. The recommendation is gone.

Commit the change:

```bash
git add prompts/base_rules.txt
git commit -m "Prompt v2: remove operational recommendations"
git tag prompt-v2
```

---

### Task 3 — Revert the prompt and reproduce the original answer

Restore the earlier system state:

```bash
git checkout inventory-v1
dvc checkout --force
dvc repro
python src/chat.py
```

Ask the question again:

```
Should we reorder milk?
```

The original answer — including the recommendation — should reappear.

---

## Extension: Lineage Investigation

Given only an answer, can you determine which prompt version produced it?

Your instructor will provide one of these two answers:

```
(A) Milk inventory is below the reorder point.
    You should place a reorder with supplier FarmFresh.

(B) Milk inventory is below the reorder point.
    Supplier: FarmFresh. Lead time: 2 days.
```

Use the following tools to trace which prompt version was active:

**Inspect Git history for the prompt file:**

```bash
git log prompts/base_rules.txt
```

**Compare DVC state between tags:**

```bash
dvc diff inventory-v1 prompt-v2
```

**Inspect the DVC lock file** to see exactly which file hashes were recorded:

```bash
cat dvc.lock
```

You should be able to identify, for any given answer:

- Which prompt version was active
- Which dataset version was loaded
- Which embedding index was used

---

## Advanced Extension: Automatic Generation Logging

Modify `src/chat.py` to log every question and answer to `logs/generation_log.json` with the following fields:

```json
{
  "question": "Should we reorder milk?",
  "answer": "...",
  "prompt_hash": "<sha256 of system_prompt.txt>",
  "dataset_version": "<git tag or commit>",
  "timestamp": "2026-03-07T10:00:00Z"
}
```

With this log in place, you can trace any answer back to the exact system state that produced it — without relying on memory or manual notes.

---

## Key Takeaways

| Lesson | What this exercise demonstrates |
|---|---|
| Prompts affect output | Same data + different prompt = different answer |
| Prompts must be versioned | DVC + Git track prompt changes alongside data changes |
| Rollback restores behavior | `git checkout` + `dvc checkout` + `dvc repro` fully restores earlier outputs |
| Lineage enables auditing | You can trace any answer back to the prompt, data, and index that produced it |

---

## Files Relevant to This Exercise

```
prompts/
└── base_rules.txt        # Edit this to change the prompt

artifacts/
└── system_prompt.txt     # Generated from base_rules.txt + inventory (DVC output)

dvc.lock                  # Records file hashes for every stage output
logs/
└── generation_log.json   # (Advanced) Per-query lineage log
```

---

## Commands Reference

| Command | Purpose |
|---|---|
| `dvc repro` | Rebuild any stages whose dependencies have changed |
| `dvc checkout` | Restore DVC-tracked files to match the current Git commit |
| `dvc diff <tag1> <tag2>` | Show what changed between two versions |
| `git log prompts/base_rules.txt` | Show commit history for the prompt file |
| `cat dvc.lock` | Inspect recorded file hashes for the last pipeline run |
