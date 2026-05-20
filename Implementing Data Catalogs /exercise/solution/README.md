# Exercise: Automating Document Classification and Using a Catalog to Support GenAI

## Overview

In this exercise, you will extend the **Data Catalog + Generative AI demo** by implementing **automated document classification** and then using the catalog to support a **Generative AI retrieval pipeline**.

In real organizations, thousands of documents exist across many domains. Manually classifying each document is not scalable. Instead, metadata classification must often be **automated using rules, keywords, or natural language techniques**.

This exercise will guide you through building a **simple automated classification system** and then using the catalog to identify documents relevant to a specific operational domain.

---

# Learning Objectives

By completing this exercise, you will learn how to:

* automate document classification using simple rules
* update catalog metadata programmatically
* query a catalog to discover documents by category
* understand how catalog metadata supports **RAG pipelines**
* connect **data management concepts with Generative AI architectures**

---

# Scenario

You are working on a **retail operations assistant** that helps employees answer questions about store procedures.

The system uses internal manuals covering topics such as:

* store safety
* shipping delays
* warehouse procedures
* inventory management
* HR policies

However, before these documents can be used by a GenAI system, they must be **properly classified** in the data catalog.

Your task is to implement an automated approach to classify documents and then use the catalog to identify documents relevant to **Safety procedures**.

---

# Starting Point

You will use the existing demo system:

```
catalog_rag_demo.py
catalog.json
manuals/
```

The catalog already contains metadata fields such as:

* `document_id`
* `title`
* `domain`
* `classification`
* `quality_score`
* `approved_for_genai`

You will extend this catalog with an additional field:

```
category
```

Example categories:

* Safety
* Inventory
* Shipping
* Store Operations
* HR
* Procurement

---

# Task 1 — Review the Catalog

Open `catalog.json` and examine the metadata for each document.

Questions to consider:

* Which metadata fields are already present?
* Which fields describe **technical metadata**?
* Which fields describe **business metadata**?
* Which fields are related to **governance**?

Write a short paragraph explaining how the catalog helps the AI system decide which documents to use.

---

# Task 2 — Implement Automated Classification

Instead of manually assigning categories, implement an **automated classification function**.

You may choose one of the following approaches:

### Option A — Keyword Matching

Use keywords found in document content or titles.

Example:

```python
if "spill" in text or "hazard" in text:
    category = "Safety"
```

### Option B — Regex Rules

Use pattern matching to detect categories.

Example:

```python
import re

if re.search(r"(spill|safety|incident)", text):
    category = "Safety"
```

### Option C — Simple NLP Heuristics

Use tokenization or keyword scoring.

Example:

```python
safety_terms = ["spill", "hazard", "safety", "incident"]
score = sum(term in text for term in safety_terms)

if score >= 2:
    category = "Safety"
```

For this exercise, **automation is recommended** so the catalog can scale to many documents.

Update the catalog entries with the new `category` field.

Example:

```json
{
  "document_id": "DOC004",
  "title": "Store Safety and Spill Response Manual",
  "category": "Safety"
}
```

---

# Task 3 — Update the Catalog

Write a script that:

1. reads the catalog
2. reads the associated manual text
3. assigns a category automatically
4. updates the catalog file

Your output catalog should now include:

```
category
```

for each document.

---

# Task 4 — Query the Catalog

Write a function that queries the catalog to find all **Safety-related documents**.

Example result:

```
DOC004  Store Safety and Spill Response Manual
DOC008  Warehouse Hazard Handling Guide
```

Export the metadata for these documents to a file:

```
safety_documents.json
```

The exported metadata should include:

* document_id
* title
* domain
* owner
* steward
* classification
* category
* last_updated

---

# Task 5 — Connect the Catalog to a RAG Pipeline

In a short written explanation (3–5 paragraphs), describe how the catalog results could feed into a Generative AI system.

Your explanation should address the following questions:

1. How could the catalog identify which documents should be embedded?
2. Why is metadata filtering important before building a vector index?
3. How would the exported **Safety documents** feed into a RAG pipeline?
4. What risks exist if the catalog filtering step is skipped?

You may reference the architecture used in the demo system.

---

# Expected Workflow

Your solution should follow this logical pipeline:

```
Manual Documents
      │
      ▼
Automated Classification
      │
      ▼
Updated Data Catalog
      │
      ▼
Catalog Query
      │
      ▼
Export Safety Metadata
      │
      ▼
Documents Selected for RAG Indexing
```

---

# Deliverables

Submit the following:

1. **classification script**
2. **updated catalog.json**
3. **exported safety_documents.json**
4. **short written explanation (3–5 paragraphs)**

---

# Optional Challenge

For an additional challenge, modify the RAG system so that it **only indexes documents belonging to a chosen category**.

Example:

```
category = "Safety"
```

The system should then build a vector index containing only Safety documents and answer questions such as:

```
What should employees do if there is a spill in the store?
```

---

# Key Insight

In enterprise AI systems, metadata is not just documentation.
It is a **control mechanism** that determines:

* which data enters AI pipelines
* which information models can retrieve
* how governance policies are enforced

By automating classification and catalog queries, organizations can scale these controls across thousands of documents.
