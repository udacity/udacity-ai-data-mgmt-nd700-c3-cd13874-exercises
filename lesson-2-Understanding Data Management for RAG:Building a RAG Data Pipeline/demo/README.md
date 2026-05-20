# Enterprise RAG – Fleet Maintenance Demo

This project demonstrates a **production-style Enterprise RAG (Retrieval-Augmented Generation)** system built with:

* **Cosmos DB (Mongo API)** → System of record for structured and unstructured data
* **ChromaDB** → Vector store for semantic search
* **Azure OpenAI** → Embeddings + Chat model

The purpose of this demo is to teach **enterprise data management for Generative AI systems**, using **fleet maintenance reports** as the working dataset.

---

## 🎯 Purpose of the Demo

While most RAG tutorials focus on just retrieving documents or PDFs, **enterprise RAG is fundamentally about data architecture**.

This demo highlights:

* The different types of data required for enterprise RAG
* How they are stored
* Which data is user-owned and needs secure storage
* How to handle real-world data like truck maintenance reports

---

## 🧰 Data Types in Enterprise RAG for Fleet Maintenance

| Data Type                  | Purpose                               | Stored In                | Notes                                                                |
| -------------------------- | ------------------------------------- | ------------------------ | -------------------------------------------------------------------- |
| **Maintenance Reports**    | Source knowledge of fleet operations  | MongoDB (`documents`)    | Example: truck inspections, repairs, service logs                    |
| **Chunking Configuration** | Controls text splitting for retrieval | MongoDB (`rag_config`)   | Chunk size, overlap; ensures reproducible retrieval                  |
| **Vector Embeddings**      | Semantic search for relevant chunks   | Chroma                   | Embeds maintenance reports for fast similarity queries               |
| **Prompts**                | Defines model behavior                | MongoDB (`prompts`)      | System and user prompts; treated as user-owned intellectual property |
| **Memory**                 | Session summaries of prior questions  | MongoDB (`memory`)       | Stores user-specific conversation history; must be secure            |
| **Interactions**           | Logging for traceability              | MongoDB (`interactions`) | Audit trail for questions, answers, and retrieved documents          |

---

## 1️⃣ Maintenance Reports

* **Type:** User-owned enterprise data
* **Content:** Vehicle ID, maintenance type, service date, observations, repairs performed
* **Collection:** `documents`

```python
self.documents = self.db[document_collection]
```

These reports form the **source of truth** for the RAG system, providing context for all retrievals and responses.

---

## 2️⃣ RAG Configuration

* **Type:** Operational metadata
* **Collection:** `rag_config`
* **Purpose:** Determines chunk size and overlap when splitting long maintenance reports for embedding

Example:

```json
{
  "config_id": "default",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

This ensures that retrieval and embedding are **consistent and reproducible** across sessions.

---

## 3️⃣ Vector Embeddings

* Stored in **ChromaDB** at `./chroma_store`
* Contains:

  * Chunk text
  * Embedding vectors
  * Metadata linking back to the original maintenance report (`mongo_id`)

Embeddings enable **semantic search**, e.g., retrieving all maintenance reports mentioning "engine oil leak."

---

## 4️⃣ Prompt Store

* **Collection:** `prompts`
* Stores system prompts and templates defining RAG behavior

Example:

```json
{
  "prompt_id": "rag_v1",
  "system_prompt": "You are a maintenance assistant. Use only retrieved maintenance reports.",
  "template": "..."
}
```

* Prompts are **user-owned intellectual property** and define the AI's decision-making and response style.
* Must be **securely stored and versioned** in enterprise deployments.

---

## 5️⃣ Memory Store

* **Collection:** `memory`
* Stores conversation summaries per `session_id`
* Useful for **context-aware responses** in multi-turn queries

Memory is **user-owned**, so in enterprise setups it must be **encrypted and access-controlled**.

---

## 6️⃣ Interaction Logging

* **Collection:** `interactions`

* Stores:

  * Questions asked
  * Answers provided
  * Retrieved document IDs
  * Timestamps

* Critical for **traceability, auditing, and compliance**

---

## 🔐 Security Considerations

In this demo, **all data is stored in MongoDB for simplicity**, but in a real enterprise scenario:

* **User-owned data** (prompts, memory, interactions) must be stored in **secure, compliant environments**
* Could be:

  * Encrypted cloud storage
  * Private network databases
  * On-premise storage
* **Vector stores** should not be the sole source of knowledge; the **documents collection remains the system of record**

---

## 🏗️ Enterprise Data Layers

Conceptually, the architecture looks like this:

```
           ┌───────────────────────┐
           │   Azure OpenAI LLM    │
           └────────────┬──────────┘
                        │
                        ▼
           ┌───────────────────────┐
           │     Prompt Layer      │ (Mongo)
           └───────────────────────┘
                        │
                        ▼
           ┌───────────────────────┐
           │     Memory Layer      │ (Mongo)
           └───────────────────────┘
                        │
                        ▼
           ┌───────────────────────┐
           │   Retrieval Layer     │ (Chroma)
           └───────────────────────┘
                        │
                        ▼
           ┌───────────────────────┐
           │  Maintenance Reports  │ (Mongo)
           └───────────────────────┘
```

---

## 📚 Learning Objectives

By exploring this code, students will learn:

1. How **enterprise RAG differs from simple demos**
2. Why **prompt, memory, and interaction data are critical**
3. How to design AI systems as **data platforms**, not just models
4. How to handle **user-owned data securely**
5. How chunking and vectorization impact **retrieval quality and system performance**

---

## ⚡ Key Takeaways

* RAG is **a data system first**, not just a chat interface
* **Prompts and memory are valuable assets** that require secure storage
* **Maintenance reports** are the source of truth
* Vector stores **accelerate retrieval** but are not authoritative
* Interaction logging is **essential for compliance and responsible AI**
