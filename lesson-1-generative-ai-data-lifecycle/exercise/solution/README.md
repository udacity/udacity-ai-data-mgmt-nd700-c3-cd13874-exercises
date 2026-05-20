# Code Changelog: Generation Data Pipeline
> Changes from Code 1 (Original) to Code 2 (Updated)

---

## Overview

This document describes the changes made between Code 1 and Code 2 of the solar generation data pipeline. The pipeline loads generation reports, validates data using Great Expectations, stores embeddings in ChromaDB, and provides an AI-powered chatbot interface.

---

## 1. How Great Expectations Is Used Differently

In Code 1, Great Expectations is used **for alerting only** — it runs validation and reports whether the data passed or failed, but the results are not used to drive any downstream logic. The data cleaning is handled separately using manual pandas operations (`drop_duplicates`, `dropna`, `between`), which are disconnected from the validation rules. This means the cleaning logic has to be maintained in parallel with the expectations, and there is no guarantee they stay in sync.

Additionally, Code 1 produces **no audit files**. Validation outcomes are only printed to the console and are not persisted anywhere, and there is no record of which specific rows failed validation.

Code 2 addresses both of these issues: validation results are used directly to identify and remove invalid rows, and full audit output is saved to the `logs/` directory — a JSON validation report and a CSV of all invalid rows.

---

## 2. Great Expectations — Data Source Setup

The method for creating the pandas data source was updated to be idempotent.

| Category | Code 1 (Original) | Code 2 (Updated) |
|---|---|---|
| Datasource creation | `context.data_sources.add_pandas("generation_source")` | `context.data_sources.add_or_update_pandas(name="generation_source")` — uses named parameter and won't fail on re-run |

---

## 3. Expectation Suite — New & Modified Expectations

The expectation suite was expanded to cover more data quality rules.

| Category | Code 1 (Original) | Code 2 (Updated) |
|---|---|---|
| `report_date` null check | Not present | Added `ExpectColumnValuesToNotBeNull(column="report_date")` to catch missing dates |
| `power_output_kw` range | Simple between check (0–1000) | The valid range remains the same (0–1000), but a `row_condition='power_output_kw != -999.5'` is added with `condition_parser="pandas"` to exclude the value `-999.5` from the range check. Some sensor and tag pipelines send `-999.5` to denote missing data — this falls outside the valid range but should **not** be filtered out, as reports with missing readings must remain available for analysis. |
| `battery_temp_c` range | Checks -20 to 80 | Unchanged |

---

## 4. Validation Approach

Code 2 replaces the Checkpoint-based validation with a direct Validator approach and adds full audit output.

| Category | Code 1 (Original) | Code 2 (Updated) |
|---|---|---|
| Validation method | Creates `ValidationDefinition` and `Checkpoint`, then runs `checkpoint.run(batch_parameters={"dataframe": df})` | Uses `batch_definition.get_batch()` + `context.get_validator()` + `validator.validate(result_format="COMPLETE")` for detailed results |
| Validation report | Only prints `results.success` to console | Saves full JSON validation report to `logs/validation_report.json` |
| Invalid rows | Uses manual pandas filtering (`dropna`, `between`, `drop_duplicates`) to clean data | Extracts invalid row indices from `unexpected_index_list`; saves invalid rows to `logs/invalid_generation_reports.csv` |
| Clean DataFrame | Manually filtered using multiple pandas operations | Derived by dropping invalid indices: `df.drop(index=invalid_indices)` |

---

## 5. Document Formatting for Embeddings

The `row_to_document` function was updated to include additional context and cleaner field labels.

| Category | Code 1 (Original) | Code 2 (Updated) |
|---|---|---|
| Site name | Not included in document string | Added `f"Site {row.site_name}."` after report ID |
| Technician label | `Technician name:` | Shortened to `Technician` |
| Comments label | `Report Comments:` | Shortened to `Comments` |

---

## 6. Vector Retrieval

| Category | Code 1 (Original) | Code 2 (Updated) |
|---|---|---|
| `n_results` | `n_results=1` — only top 1 document retrieved per query | `n_results=2` — retrieves top 2 documents to provide richer context to the LLM |

---

## 7. Chatbot Interface

| Category | Code 1 (Original) | Code 2 (Updated) |
|---|---|---|
| Startup message | No introduction shown | Prints `Solar Generation AI Assistant` and `Type exit to quit` on startup |
| System prompt | `"You help analyze solar generation data."` | `"You analyze solar generation data."` (minor wording change) |
| User prompt | Rigid output format with fixed `Best Match` section and bullet list | Flexible prompt: instructs to use most current report as current state, older reports for historical context |
| Response printing | Stores `answer = response.choices[0].message.content` then prints | Prints `response.choices[0].message.content` inline without intermediate variable |

---

## Summary of Key Improvements

- **Idempotent data source creation** prevents re-run errors
- **Expanded validation**: `report_date` null check and sentinel value exclusion for `power_output_kw`
- **Full validation audit trail**: JSON report + CSV of invalid rows saved to `logs/`
- **Cleaner data cleaning**: driven by validation results rather than manual pandas filters
- **Richer embeddings context** by including `site_name` in document strings
- **Improved retrieval** with `n_results=2` for better LLM context
- **More flexible chatbot prompt** with temporal reasoning instructions
