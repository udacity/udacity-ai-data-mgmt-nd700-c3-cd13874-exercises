# Data Catalog + Generative AI Demo

## Metadata-Driven Document Selection for a Retail Operations RAG Assistant

## Overview

This demo shows how a **data catalog can control which documents are allowed to enter a Generative AI retrieval pipeline**.

In many enterprise GenAI systems, the hardest problem is not building the model — it is **managing the data that the model is allowed to use**.

A data catalog provides the metadata necessary to make these decisions.

In this demo, we simulate a **retail and supply chain operations assistant** that answers questions using internal manuals such as:

* store returns policies
* warehouse picking procedures
* shipping delay playbooks
* inventory cycle count processes

However, **not all documents are appropriate for a GenAI system**. Some may be:

* confidential
* restricted
* low quality
* outdated
* unrelated to the operational domain

The **data catalog acts as a governance and selection layer**, ensuring that only approved and trusted documents are indexed into the RAG system.

---

# Learning Objectives

By completing this demo, students will understand:

* what a **data catalog** is
* how **metadata** describes data assets
* how **catalog metadata can control AI data access**
* how **document filtering occurs before indexing**
* how **Azure OpenAI embeddings can be used in a RAG pipeline**
* how **vector databases store embedded document chunks**
* how catalog governance improves **GenAI reliability and compliance**

---

# Architecture

This demo implements a simplified enterprise-style architecture.

```
Manual Documents
      │
      │
Data Catalog (metadata)
      │
      │  metadata filtering
      ▼
Approved Documents
      │
      │  chunking
      ▼
Azure OpenAI Embeddings
      │
      ▼
Chroma Vector Database
      │
      ▼
Azure OpenAI Chat Model
      │
      ▼
Retail Operations Assistant
```

The key idea is:

**The catalog determines which documents the AI system is allowed to use.**

---

# Technologies Used

This demo uses several common components in enterprise GenAI systems.

| Component    | Purpose                        |
| ------------ | ------------------------------ |
| Azure OpenAI | embeddings and chat model      |
| Chroma       | vector database                |
| Python       | orchestration                  |
| JSON catalog | metadata catalog for documents |
| LangChain    | vector store integration       |

---

# Repository Structure

```
repo/
│
├── catalog_rag_demo.py
│
├── catalog.json
│   metadata catalog for manuals
│
├── manuals/
│   store_returns_manual.txt
│   warehouse_picking_guide.txt
│   shipping_delay_playbook.txt
│   inventory_cycle_count.txt
│   employee_handbook.txt
│   supplier_pricing_terms.txt
│   legacy_store_sop.txt
│
└── catalog_demo_chroma/
    vector database created during indexing
```

The demo automatically creates the sample manuals and catalog the first time it runs.

---

# Data Catalog Structure

The catalog describes each document using metadata fields.

Example entry:

```json
{
  "document_id": "DOC001",
  "file_name": "store_returns_manual.txt",
  "title": "Store Returns and Exchanges Manual",
  "domain": "store_operations",
  "document_type": "manual",
  "owner": "Retail Operations",
  "steward": "Data Governance",
  "classification": "internal",
  "quality_score": 94,
  "last_updated": "2025-08-01",
  "approved_for_genai": true
}
```

Important metadata attributes include:

| Field              | Purpose                         |
| ------------------ | ------------------------------- |
| domain             | business domain of the document |
| classification     | security classification         |
| quality_score      | trustworthiness of data         |
| last_updated       | document freshness              |
| approved_for_genai | governance approval             |
| owner              | business owner                  |
| steward            | data steward responsible        |

---

# Catalog Filtering Rules

Before documents are indexed, the system evaluates metadata rules.

Documents are excluded if:

* `approved_for_genai = false`
* classification is **restricted** or **confidential**
* domain is not part of operations
* quality score is too low
* document is outdated

Example output during filtering:

```
EXCLUDE DOC005 Supplier Pricing Terms
 - blocked classification: confidential

EXCLUDE DOC006 Employee Handbook
 - not approved for GenAI
 - blocked classification: restricted

EXCLUDE DOC007 Legacy Store SOP
 - quality too low
 - too old
```

Only approved documents are indexed.

---

# Chunking and Embedding

After filtering:

1. documents are loaded
2. text is split into chunks
3. chunks are embedded using **Azure OpenAI embeddings**
4. embeddings are stored in **Chroma**

Chunk metadata is also stored in the vector database.

Example metadata stored with each chunk:

```
document_id
title
domain
owner
steward
quality_score
classification
indexed_at
chunk_index
```

This metadata can later support:

* governance auditing
* lineage
* access control
* document tracing

---

# Retrieval and Question Answering

When a user asks a question:

1. the question is embedded
2. the vector database retrieves relevant chunks
3. retrieved context is inserted into a prompt
4. Azure OpenAI generates an answer

Example query:

```
What should staff do when a shipping delay affects many orders?
```

The system retrieves the **Shipping Delay Response Playbook** and generates a grounded answer.

---

# Why Catalog Filtering Matters

If the catalog filtering step were removed, the system might retrieve:

* HR policies
* supplier pricing contracts
* confidential documents
* outdated procedures

This can lead to:

* inaccurate answers
* policy violations
* exposure of sensitive information

Using catalog metadata ensures that the AI assistant operates on **approved, relevant, and trusted data**.

---

# Setup

## 1 Install Dependencies

```
pip install langchain
pip install langchain-openai
pip install langchain-chroma
pip install chromadb
pip install openai
```

---

## 2 Configure Azure OpenAI

Set the following environment variables.

```
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_KEY="your-key"

export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="your-embedding-deployment"
export AZURE_OPENAI_CHAT_DEPLOYMENT="your-chat-deployment"

export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

---

# Running the Demo

Run:

```
python catalog_rag_demo.py
```

The script will:

1. create sample manuals
2. create a metadata catalog
3. filter documents using catalog metadata
4. build the vector index
5. run example questions

---

# Example Questions

```
What should staff do when a carrier delay affects many orders?

How should store employees process returns for damaged items?

What happens when inventory cycle count variance exceeds five units?

What are the employee benefits policies?
```

The final question should not return a meaningful answer because the **HR handbook is excluded by the catalog governance rules**.

---

# Key Takeaways

This demo illustrates an important principle in enterprise Generative AI:

**AI systems should not ingest all available data.**

Instead, they should rely on **metadata-driven governance** to determine which information is safe and appropriate for use.

The data catalog becomes a critical layer that:

* improves retrieval quality
* prevents sensitive data exposure
* enforces governance policies
* increases trust in AI outputs

In real production systems, the catalog would typically be implemented using enterprise tools such as:

* Microsoft Purview
* Collibra
* Alation
* Databricks Unity Catalog

However, the concepts demonstrated here remain the same.

---

# Possible Extensions

Students can extend this demo in several ways.

Examples include:

* connecting the catalog to a database instead of JSON
* implementing role-based document access
* adding document lineage tracking
* logging retrieval metadata for auditing
* integrating the catalog with a data governance platform

---

# Conclusion

This demo demonstrates how **data management concepts directly influence Generative AI systems**.

Metadata is not just documentation — it can actively control:

* which documents are indexed
* which knowledge the model can access
* how trustworthy the system's answers are

In modern AI systems, **data governance and metadata management are just as important as the models themselves.**
