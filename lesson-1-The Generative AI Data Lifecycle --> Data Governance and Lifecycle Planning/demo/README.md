# Solar Generation AI Assistant — Data Validation & Governance Demo

## Overview

This demo is for a solar energy company's Generative AI assistant. It demonstrates how to build a **data validation and governance pipeline** that sits between raw operational data and an AI system, ensuring only trusted, high-quality data enters the AI lifecycle.

The AI assistant answers engineering and operations questions such as:

- *"Which solar inverters failed last week?"*
- *"What maintenance actions were taken on inverter INV-203?"*
- *"Summarize battery overheating incidents."*

---

## Pipeline

```
Maintenance logs → Validation → Cleaning → Embeddings → Vector DB → AI Assistant
```

---

## What the Script Does

### 1. Secrets & Authentication
Connects to **Azure Key Vault** using browser-based interactive login to securely retrieve the Azure OpenAI endpoint and API key. No credentials are hardcoded.

### 2. Load Data
Reads solar generation maintenance reports from `data/Generation_Data.csv` into a Pandas DataFrame.

### 3. Data Validation with Great Expectations
Defines and runs a validation suite against the raw data with the following rules:

| Rule | Column | Constraint |
|---|---|---|
| Unique values | `report_id` | No duplicate reports |
| Not null | `inverter_id` | Every record must have an inverter ID |
| Value range | `power_output_kw` | Between 0 and 1000 kW |
| Value range | `battery_temp_c` | Between -20°C and 80°C |

The validation result is printed to the console. If validation fails, the pipeline surfaces which rules were violated before any data reaches the AI system.

### 4. Data Cleaning
Applies the same rules programmatically to produce a clean DataFrame — removing duplicates, dropping rows with null critical fields, and filtering out-of-range values.

### 5. Convert Records to Documents
Each clean row is converted into a human-readable text document that describes the generation report, ready for embedding.

### 6. Create Embeddings
Sends each document to the **Azure OpenAI** `text-embedding-ada-002` model to generate a vector embedding.

### 7. Store in Vector Database
Stores documents and their embeddings in a local **ChromaDB** collection for semantic retrieval.

### 8. RAG Chatbot
Runs an interactive command-line chatbot. For each user question, it:
1. Embeds the question using the same embedding model
2. Retrieves the most relevant report from ChromaDB
3. Passes the retrieved context and question to `gpt-4.1-mini` via Azure OpenAI
4. Returns a structured answer covering report ID, inverter, date, power output, temperature, technician, comments, status, and a direct answer

Type `exit` to quit the chatbot.

---

## Requirements

```
pandas
great_expectations
chromadb
openai
azure-identity
azure-keyvault-secrets
```

Install with:

```bash
pip install pandas great_expectations chromadb openai azure-identity azure-keyvault-secrets
```

---

## Setup

1. Ensure you have access to the `userAIsecrets` Azure Key Vault with the following secrets:
   - `structuredazureendpoint` — your Azure OpenAI endpoint URL
   - `structuredazureapikey` — your Azure OpenAI API key

2. Place your data file at `data/Generation_Data.csv`. The file should contain the following columns:

   | Column | Description |
   |---|---|
   | `report_id` | Unique report identifier |
   | `inverter_id` | Inverter unit identifier |
   | `report_date` | Date of the report |
   | `power_output_kw` | Power output in kilowatts |
   | `battery_temp_c` | Battery temperature in Celsius |
   | `technician` | Technician name |
   | `report_text` | Free-text maintenance comments |
   | `status` | Report status (e.g. resolved, open) |


---

## Key Learning Concepts

This demo covers the following course topics:

- **Data validation** — defining and running automated quality checks with Great Expectations
- **Data quality rules** — range checks, null checks, uniqueness constraints
- **Data governance** — enforcing a validation gate before data enters an AI pipeline
- **Monitoring** — surfacing validation results so failures are visible before they propagate
- **RAG (Retrieval-Augmented Generation)** — combining a vector database with a language model to answer domain-specific questions grounded in real data

---

## Notes

- The ChromaDB collection is created in-memory and will not persist between runs. For production use, configure a persistent ChromaDB client.
- The chatbot retrieves only the single most relevant document (`n_results=1`) per query. This can be increased for broader context.
- PII handling (e.g. technician names in report text) is flagged as a governance concern in the course but is not automatically redacted in this demo — this is intentional, as PII detection is discussed as an extension exercise.
