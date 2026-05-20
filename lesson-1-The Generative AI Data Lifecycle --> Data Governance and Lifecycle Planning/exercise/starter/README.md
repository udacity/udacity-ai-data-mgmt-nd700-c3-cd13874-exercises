# Refactoring Exercise: Solar Generation Chatbot

## Overview

In this exercise, you are given a working solar generation chatbot and started code for a new improved chatbot. Your task is to refactor the original code, using the starter code as a templete to start with and implement the improved chatbot version that handles sensor sentinel values, produces audit logs, and retrieves richer context for its AI responses.

The pipeline loads solar generation reports from a CSV, validates the data using Great Expectations, generates embeddings stored in ChromaDB, and exposes an AI chatbot for querying the data.

---

## The Given Chatbot Code

The given chatbot code is functional but has a few limitations:

- Great Expectations runs validation and prints a pass/fail result, but the outcome is **not used to clean the data**. Cleaning is handled separately by disconnected pandas operations (`drop_duplicates`, `dropna`, `between`), which have to be maintained in parallel with the validation rules.
- There are **no audit files** — validation outcomes are only printed to the console, with no record of which rows failed or why.
- The chatbot retrieves only the **top 1** most similar document when answering a question, limiting the context available to the model.
- The chatbot prompt enforces a **rigid output format**, with a fixed `Best Match` section and bullet list.

---

## The Refactored Code

The refactored code addresses all of the above. The key changes are:

### Validation Drives Cleaning

Instead of running validation as a passive check, the refactored code uses the validation results directly to identify and remove invalid rows. Invalid row indices are extracted from `unexpected_index_list` and the clean DataFrame is produced by dropping those indices — no separate pandas filters needed.

### Audit Files

The refactored code saves two audit files to a `logs/` directory on every run:

- `logs/validation_report.json` — the full Great Expectations validation output
- `logs/invalid_generation_reports.csv` — all rows that failed validation

The `logs/` directory is created automatically if it does not exist.

### Handling -999.5 Sentinel Values

Some sensor and tag pipelines send a value of **-999.5** to indicate missing data. In the given chatbot code, this would cause those rows to fail the `power_output_kw` range check (valid range: 0–1000) and be removed from the dataset — which is incorrect, as **reports with missing readings must remain available for analysis**.

The refactored code adds a `row_condition` to the `power_output_kw` expectation to exclude `-999.5` from the range check:

```python
gx.expectations.ExpectColumnValuesToBeBetween(
    column="power_output_kw",
    min_value=0,
    max_value=1000,
    <TODO>
    condition_parser="pandas"
)
```

The valid range itself is unchanged — only rows with a genuinely out-of-range value (not `-999.5`) will fail validation.

### Updated Chatbot Prompt

The given chatbot prompt enforces a fixed output structure. The refactored prompt is more flexible and instructs the model to reason across time — treating the most recent report as the current state and using older reports for historical context:

**Given chatbot prompt:**
```
You are an energy operations assistant.
Use the following reports:
{context_text}
Answer the question:
{question}
Use this format for your answer:
    Best Match
    - Generation report
    - Inverter
    - Date
    - Power output
    - Battery temperature
    - Technician name
    - Report Comments
    - Status

    Questions Answer
```

**Refactored prompt:**
```
<TODO>
```

### Top 2 Embedding Similarity Selection

The given chatbot retrieves only the **top 1** most similar document from the vector store per query. The refactored code increases this to **top 2**, giving the model more context to work with when formulating its response:

**Given chatbot code:**
```python
results = collection.query(
    query_embeddings=[emb],
    n_results=1
)
```

**Refactored code:**
```python
<TODO>
```

---

## Other Changes

- `add_pandas` replaced with `add_or_update_pandas` for idempotent datasource creation
- Null check added for `report_date` in the expectation suite
- `site_name` added to the document string used for embedding generation
- Chatbot displays a startup banner before the input loop begins

---

## Success Criteria

Your refactored code should:

- [ ] Use validation results to drive data cleaning — no manual pandas filters
- [ ] Save a JSON validation report to `logs/validation_report.json`
- [ ] Save invalid rows to `logs/invalid_generation_reports.csv`
- [ ] Preserve rows where `power_output_kw == -999.5` — these must not be filtered out
- [ ] Use the updated flexible chatbot prompt
- [ ] Retrieve `n_results=2` from the vector store
- [ ] Use idempotent datasource creation (`add_or_update_pandas`)
- [ ] Include a null check for `report_date` in the expectation suite
- [ ] Include `site_name` in generated document strings
- [ ] Display a startup message in the chatbot
- [ ] Show how the new chatbot, when asked "Is there any issue with INV-200", responds with an answer that has this or similar semantic meaning:
Assistant: {.   
  "current_status": {.   
    "site": "Woodlands",    
    "inverter": "INV-200",    
    "date": "2026-02-18T12:30:00",    
    "power_output": -999.5,     
    "battery_temperature": 31.6,     
    "technician": "C. Volkov",     
    "comments": "Power output value not available, check tag mapping and sensor.",    
    "status": "ERROR"     
  },     
  "historical_status": {     
    "site": "Woodlands",     
    "inverter": "INV-200",     
    "date": "2025-02-18T12:25:00",     
    "power_output": 432.56,     
    "battery_temperature": 31.5,     
    "technician": "T. Kovac",      
    "comments": "All system working properly.",     
    "status": "OK"
  },      
  "analysis": "The most recent report (2026-02-18) indicates an error with the power output reading for inverter INV-200, showing an invalid value (-999.5 kW) and a status of ERROR. The technician's comments suggest checking the tag mapping and sensor. The previous report (2025-02-18) showed normal operation with no issues. Therefore, currently, there is an issue with INV-200 related to power output measurement or sensor/tag configuration."      
}
