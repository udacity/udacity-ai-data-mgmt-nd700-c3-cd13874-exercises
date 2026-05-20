# Exercise: Improving a Synthetic Training Data Pipeline

## Overview

In this exercise you will modify an existing **Generative AI data pipeline** that creates **synthetic training data from documentation**.

The original system uses a **Large Language Model (LLM)** to generate **question–answer pairs** from a reference document. These pairs are then used to build a dataset for **fine-tuning or training a chatbot model**.

Your task is to **improve the pipeline design** by changing the order of operations to make the system **safer and more efficient**.

You are provided with:

* The **original pipeline code**
* A **reference document**

Your job is to **modify the code so the pipeline follows a new architecture**.

---

# Background

When building domain assistants, organizations need **training data that resembles real user interactions**.

Example training pair:

```
Question:
What responsibilities does a store manager have?

Answer:
A store manager oversees daily store operations, manages staff schedules,
monitors sales performance, ensures compliance with company policies,
and handles escalated customer issues.
```

Creating these datasets manually is often difficult because:

* SMEs must write many questions
* The process is slow
* It is expensive

To solve this, we can generate **synthetic datasets** using an LLM that reads documentation and creates **realistic question–answer pairs**.

---

# The Original Pipeline

The code you are given implements the following pipeline.

```
Reference Document
        │
        ▼
Document Chunking
        │
        ▼
LLM Generates Q&A Pairs
        │
        ▼
LLM Quality Review
        │
        ▼
PII Detection (Microsoft Presidio)
        │
        ▼
Clean Training Dataset
```

This approach works, but it has **two important problems**.

---

# Problem 1 — Sending PII to an External LLM

The reference document may contain **personally identifiable information (PII)** such as:

* Names
* Email addresses
* Phone numbers
* Credit card numbers

In the original pipeline, the **raw document is sent to the LLM** before checking for PII.

This means **sensitive information could be transmitted to an external AI service**.

For many organizations, this is **not acceptable**.

---

# Problem 2 — Generating Data That Will Be Discarded

In the original pipeline:

1. The LLM generates question–answer pairs.
2. Later, a **PII filtering step removes some of those pairs**.

This means the system may:

* Pay for LLM generation
* Then throw away the results

This **wastes compute and increases cost**.

---

# Your Task

You must modify the provided code so that the pipeline **cleans the reference document before using the LLM**.

Instead of filtering generated Q&A pairs, the pipeline should:

1. Detect and remove PII **from the document itself**
2. Save a **clean version of the document**
3. Generate question–answer pairs **only from the cleaned document**

---

# New Pipeline Architecture

After your changes, the pipeline should follow this structure:

```
Reference Document
        │
        ▼
PII Detection (Microsoft Presidio)
        │
        ▼
Clean Reference Document
        │
        ▼
Document Chunking
        │
        ▼
LLM Generates Q&A Pairs
        │
        ▼
LLM Quality Review
        │
        ▼
Training Dataset Creation
```

The key difference is that **PII detection happens before the LLM is used**.

---

# Expected Behavior of the Modified Pipeline

The new pipeline should perform the following steps.

---

## 1. Load the Reference Document

The system should accept:

* `.md`
* `.txt`
* `.pdf`

Example input:

```
reference.md
```

---

## 2. Detect PII in the Document

Use **Microsoft Presidio** to analyze the document and detect sensitive information.

Examples of detected entities:

* PERSON
* EMAIL_ADDRESS
* PHONE_NUMBER
* CREDIT_CARD
* US_SSN

---

## 3. Clean the Document

Replace detected PII with entity labels.

Example:

Original text:

```
Store manager John Smith can be contacted at john.smith@email.com
```

Cleaned text:

```
Store manager [PERSON] can be contacted at [EMAIL_ADDRESS]
```

The cleaned document should be saved as:

```
reference_clean.md
```

---

## 4. Chunk the Clean Document

The cleaned document should be split into chunks so the LLM can process it.

Example configuration:

```
Chunk size: 3000 characters
Overlap: 300 characters
```

---

## 5. Generate Question–Answer Pairs

For each chunk, the LLM generates synthetic Q&A pairs grounded in the document text.

Example instruction:

```
Generate 8 question answer pairs based only on the text below.
```

---

## 6. Review Generated Data

A second LLM pass reviews the generated pairs and removes examples that are:

* Weak
* Repetitive
* Not grounded in the document

---

## 7. Create the Training Dataset

The final dataset should be formatted for **chat model training**.

Example format:

```json
{
  "messages":[
    {"role":"system","content":"You are a helpful assistant"},
    {"role":"user","content":"What is inventory management?"},
    {"role":"assistant","content":"Inventory management is the process of..."}
  ]
}
```

The pipeline should generate:

```
train.jsonl
valid.jsonl
```

---

# Expected Output Files

After the pipeline runs, the project should produce:

```
reference_clean.md
train.jsonl
valid.jsonl
pii_report.json
```

Where:

* **reference_clean.md** → sanitized version of the document
* **train.jsonl** → training dataset
* **valid.jsonl** → validation dataset
* **pii_report.json** → summary of detected PII

---

# Requirements

Python libraries used in the pipeline:

```
openai
azure-identity
azure-keyvault-secrets
pypdf
presidio-analyzer
presidio-anonymizer
spacy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy language model:

```bash
python -m spacy download en_core_web_lg
```

---

# Running the Pipeline

Place the reference document in the project directory.

Example:

```
reference.md
```

Run the pipeline:

```bash
python app.py
```

---

# Key Learning Objectives

By completing this exercise, you will learn:

* How **synthetic training data pipelines** are built
* How to **modify AI workflows to improve safety**
* How to **integrate PII detection into data pipelines**
* Why **pipeline order matters in AI systems**
* How to **reduce LLM cost by avoiding unnecessary generation**

Most importantly, this exercise demonstrates a key enterprise AI principle:

**Sensitive data should be removed before sending information to external AI systems.**
