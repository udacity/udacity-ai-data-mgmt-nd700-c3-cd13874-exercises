# Synthetic Training Data Pipeline for Generative AI

## Overview

This demo shows how to build a **Generative AI data pipeline that creates synthetic training data from documentation**.

Instead of using **Retrieval Augmented Generation (RAG)** to answer questions from documents at runtime, another approach is to **train a model directly with domain knowledge**. When a model is trained with this information, it can respond using the trained knowledge rather than relying on general knowledge.

Training a model requires **high-quality training datasets**, typically composed of **question and answer pairs** that resemble the types of questions users ask in a chatbot.

However, these datasets are often **not available**.

Organizations typically try to obtain them by asking **subject matter experts (SMEs)** to manually write questions and answers. This process is often:

* Expensive
* Time consuming
* Difficult to scale

An alternative approach is to **generate synthetic training data**.

Large Language Models (LLMs) can analyze documentation and automatically produce **question–answer pairs that resemble real user queries**. This demo illustrates that process.

---

# Architecture

The pipeline follows these steps:

1. **Input Documentation**
2. **Document Chunking**
3. **Synthetic Q&A Generation**
4. **Quality Review**
5. **PII Detection and Filtering**
6. **Training Dataset Creation**

The final result is a **clean dataset ready for model training**.

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
        │
        ├── train.jsonl
        └── valid.jsonl
```

---

# Why Synthetic Data?

Training a model requires **training examples that mimic real interactions**.

Example training pair:

```
Question:
What responsibilities does a store manager have?

Answer:
A store manager oversees daily store operations, manages staff schedules,
monitors sales performance, ensures compliance with company policies,
and handles escalated customer issues.
```

These examples allow the model to learn **how to answer user questions about the domain**.

If real data does not exist, synthetic data can be created by:

* Extracting knowledge from documentation
* Generating realistic user questions
* Producing grounded answers

This makes it possible to **bootstrap training datasets quickly**.

---

# Data Quality and Safety

Training models with **bad data can create harmful or unsafe behavior**.

Before training a model, training data must be validated for:

* Bias
* Fairness
* Harmful content
* Personally Identifiable Information (PII)

This demo adds a **data validation step** to ensure that sensitive data is not included in the dataset.

---

# PII Detection with Microsoft Presidio

The pipeline uses **Microsoft Presidio** to detect and remove training examples that contain PII.

Examples of PII include:

* Names
* Email addresses
* Phone numbers
* Social security numbers
* Credit card numbers

If a generated question or answer contains PII, the entire pair is **removed from the dataset**.

This produces a **clean training dataset suitable for model training**.

---

# Demo Components

## `app.py`

The main pipeline script.

Responsibilities:

1. Load the reference document
2. Split the document into chunks
3. Use an LLM to generate Q&A pairs
4. Review the generated pairs for quality
5. Format them for fine-tuning
6. Split them into training and validation sets

Output files:

```
train.jsonl
valid.jsonl
```

These files can be used directly for **model fine-tuning**.

---

# Pipeline Steps

### 1. Load Reference Documentation

The pipeline accepts:

* `.md`
* `.txt`
* `.pdf`

Example:

```
reference.md
```

---

### 2. Chunk the Document

Large documents are divided into smaller sections so the LLM can process them effectively.

```
Chunk size: 3000 characters
Overlap: 300 characters
```

---

### 3. Generate Q&A Pairs

For each chunk, the LLM generates several question-answer pairs.

Example prompt:

```
Generate 8 question answer pairs based only on the text below.
```

The model is instructed to:

* Avoid hallucinations
* Avoid duplicate questions
* Ground answers in the text

---

### 4. Review Generated Data

A second LLM pass reviews the generated pairs and removes examples that are:

* Weak
* Repetitive
* Not grounded in the source document

This improves dataset quality before training.

---

### 5. PII Filtering

The dataset is then scanned using **Microsoft Presidio**.

Pairs containing PII are removed.

This prevents **sensitive information from being embedded into the model**.

---

### 6. Dataset Creation

The dataset is saved in **chat fine-tuning format**:

```json
{
  "messages":[
    {"role":"system","content":"You are a helpful assistant"},
    {"role":"user","content":"What is inventory management?"},
    {"role":"assistant","content":"Inventory management is the process of..."}
  ]
}
```

The pipeline then creates:

```
train.jsonl
valid.jsonl
```

---

# Requirements

Python libraries used:

```
openai
azure-identity
azure-keyvault-secrets
pypdf
presidio-analyzer
presidio-anonymizer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Demo

Place your reference document in the project directory.

Example:

```
reference.md
```

Then run:

```bash
python app.py
```

Output:

```
train.jsonl
valid.jsonl
```

---

# When to Use This Approach

Synthetic training pipelines are useful when:

* Domain data exists but training examples do not
* SMEs cannot generate datasets quickly
* Organizations want to bootstrap model training

This approach is commonly used in:

* Enterprise copilots
* Domain-specific assistants
* Technical support chatbots
* Knowledge base automation

---

# Key Takeaways

1. **RAG is not the only approach** to building domain-specific assistants.
2. Models can also be **trained with curated domain datasets**.
3. When training data is unavailable, **synthetic data generation is a practical alternative**.
4. **Data validation is critical** before training.
5. **PII filtering protects sensitive information** from entering models.
