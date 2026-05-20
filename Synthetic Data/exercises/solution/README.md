# Synthetic Training Data Pipeline for Generative AI

## Overview

This exercise demonstrates how to build a **Generative AI data pipeline that creates synthetic training data from documentation**.

Instead of using **Retrieval Augmented Generation (RAG)** to answer questions from documents at runtime, another approach is to **train a model directly with domain knowledge**. When a model is trained with this information, it can respond using the trained knowledge rather than relying on general knowledge.

Training a model requires **high-quality training datasets**, typically composed of **question and answer pairs** that resemble the types of questions users ask in a chatbot.

However, these datasets are often **not available**.

Organizations typically try to obtain them by asking **subject matter experts (SMEs)** to manually write questions and answers. This process is often:

* Expensive
* Time consuming
* Difficult to scale

An alternative approach is to **generate synthetic training data**.

Large Language Models (LLMs) can analyze documentation and automatically produce **question–answer pairs that resemble real user queries**.

---

# Goal of This Exercise

The provided code already generates synthetic training data from a document.

Your task is to **modify the pipeline to improve safety and efficiency**.

Specifically, you will change the pipeline so that:

**PII detection happens before synthetic data generation.**

Instead of checking for PII after generating the dataset, the document will first be **cleaned using Microsoft Presidio**, and the **clean document will be used to generate the training dataset**.

This change improves the pipeline in two important ways.

### 1. Lower LLM Cost

In the original pipeline, the LLM generates question–answer pairs from the **raw document**, and later some of those pairs may be **discarded during PII filtering**.

This means the system may:

* Pay for LLM generation
* Then throw away the results

By cleaning the document first, we **avoid generating training examples that will never be used**, which reduces **LLM usage and cost**.

---

### 2. Prevent Sending PII to External LLMs

The original pipeline sends the **raw document text to an LLM**.

If the document contains sensitive information such as:

* names
* email addresses
* phone numbers
* credit card numbers

then that data may be sent to an **external AI service**.

By cleaning the document **before calling the LLM**, we ensure that:

* **Sensitive information never leaves the system**
* The LLM only sees **sanitized data**

This is an important **enterprise AI safety pattern**.

---

# Original Pipeline

The original code follows this architecture:

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

The problem is that **PII is only detected after the LLM has already processed the document**.

---

# New Pipeline (Your Task)

You will modify the code so the pipeline becomes:

```
Reference Document
        │
        ▼
PII Detection and Cleaning (Microsoft Presidio)
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

The key change is that **Presidio runs first**.

The cleaned document is saved as:

```
reference_clean.md
```

All synthetic data generation must use this **cleaned document**.

---

# Why This Matters

This pipeline demonstrates two important enterprise AI design principles.

### Data Safety

Sensitive information must be **detected and removed before AI systems process it**.

This reduces the risk of:

* Data leakage
* Compliance violations
* Accidental model memorization

---

### Cost Optimization

Synthetic data generation often requires **many LLM calls**.

By cleaning the data first, we avoid generating examples that will later be discarded.

This reduces:

* Token usage
* API costs
* Unnecessary compute

---

# Student Task

Modify the existing code so that the pipeline performs the following steps.

### Step 1 — Load the Reference Document

Supported formats:

* `.md`
* `.txt`
* `.pdf`

Example:

```
reference.md
```

---

### Step 2 — Run PII Detection on the Document

Use **Microsoft Presidio** to detect personally identifiable information in the document.

Examples of PII include:

* Names
* Email addresses
* Phone numbers
* Credit card numbers
* Social security numbers

---

### Step 3 — Clean the Document

Replace detected PII with labels such as:

```
[PERSON]
[EMAIL_ADDRESS]
[PHONE_NUMBER]
```

Save the cleaned document as:

```
reference_clean.md
```

---

### Step 4 — Chunk the Clean Document

Split the cleaned document into smaller pieces.

Example configuration:

```
Chunk size: 3000 characters
Overlap: 300 characters
```

---

### Step 5 — Generate Synthetic Q&A Pairs

For each chunk, use an LLM to generate question-answer pairs grounded in the document text.

Example instruction:

```
Generate 8 question answer pairs based only on the text below.
```

---

### Step 6 — Review Generated Data

A second LLM pass reviews the generated pairs and removes examples that are:

* Repetitive
* Weak
* Not grounded in the source text

---

### Step 7 — Create the Training Dataset

The final dataset must be formatted in **chat fine-tuning format**.

Example:

```json
{
  "messages": [
    {"role":"system","content":"You are a helpful assistant"},
    {"role":"user","content":"What is inventory management?"},
    {"role":"assistant","content":"Inventory management is the process of..."}
  ]
}
```

The pipeline should produce:

```
train.jsonl
valid.jsonl
```

---

# Demo Components

## `app.py`

This script implements the synthetic data pipeline.

Your modifications should:

1. Run **Presidio on the reference document**
2. Save a **cleaned document**
3. Generate training data **only from the cleaned document**

---

# Requirements

Required Python libraries:

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

Download the spaCy language model:

```bash
python -m spacy download en_core_web_lg
```

---

# Running the Demo

Place your reference document in the project directory.

Example:

```
reference.md
```

Run the pipeline:

```bash
python app.py
```

Output files:

```
reference_clean.md
train.jsonl
valid.jsonl
pii_report.json
```

---

# Key Takeaways

1. **Synthetic data generation can bootstrap training datasets.**
2. **RAG is not the only way to build domain assistants.**
3. **Data validation must occur before AI processing whenever possible.**
4. Moving safety checks earlier in the pipeline can **reduce risk and cost**.
5. **Preventing PII exposure to external LLMs is a critical enterprise AI practice.**
