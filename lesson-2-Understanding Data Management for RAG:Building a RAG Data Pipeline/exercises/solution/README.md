# Enterprise RAG – Privacy-Aware Fleet Maintenance Solution

This project demonstrates a **privacy-conscious Enterprise RAG (Retrieval-Augmented Generation)** system using:

* **MongoDB (Cosmos DB Mongo API)** → system of record for non-sensitive data
* **ChromaDB** → semantic vector store for document embeddings
* **Azure OpenAI** → embeddings and chat model

**Key change in this solution:** user-owned data (prompts, memory, interaction logs) is now **stored locally**, not in the cloud, to respect privacy and data ownership.

---

## 🎯 Assignment Solution Overview

As part of the assignment, the student was asked to:

* Identify which data in the original RAG implementation is **user-owned**

* Ensure that all user-owned data (e.g., prompts, session memory, interaction logs) is **never stored in the cloud**

* Refactor the code so that **MongoDB only contains non-sensitive data**, like:

  * RAG configuration
  * Knowledge-base documents (truck maintenance reports)

* Store all sensitive or user-derived data **on the local file system**, with session-level separation

This solution implements that design.

---

## 🧰 Data Storage Design

| Data Type                     | Storage                     | Reason / Privacy                                                  |
| ----------------------------- | --------------------------- | ----------------------------------------------------------------- |
| **RAG Configuration**         | MongoDB                     | Operational parameters (chunk size, overlap), no PII              |
| **Fleet Maintenance Reports** | MongoDB                     | Knowledge base, not user-specific                                 |
| **Prompts**                   | Local JSON (`prompts.json`) | Defines business logic, sensitive; owned by user                  |
| **Session Memory**            | Local JSON per session      | Summaries of conversation; user-owned                             |
| **Interaction Logs**          | Local JSONL per session     | Questions, answers, retrieved doc references; sensitive user data |

---

## 🔹 Local Data Structure

```
./local_store/
├─ prompts.json              # system & user prompts
├─ memory/                   # session memory JSON files
│   ├─ session_123.json
│   └─ session_456.json
└─ interactions/             # session interaction logs (JSONL)
    ├─ session_123.jsonl
    └─ session_456.jsonl
```

* Each session gets **separate memory and interaction files**
* Supports **right-to-erasure** by deleting session files

---

## 🔹 RAG Query Flow (Privacy-Aware)

1. Retrieve top-k relevant chunks from **ChromaDB**
2. Hydrate full documents from **MongoDB** (fleet maintenance reports)
3. Inject **context + local session memory** into the prompt
4. Call **Azure OpenAI** chat model
5. Log interaction and update memory **locally only**

---

## 🔹 Privacy & Ownership Benefits

* **Sensitive data never leaves the user machine**
* **MongoDB contains only non-sensitive enterprise data**
* **Prompts, session memory, and interaction logs** are fully under user control
* Supports **right-to-erasure** for GDPR/CCPA compliance
* Clear separation between **knowledge-base (shared)** and **user-owned data**

---

## 🧩 Key Features Implemented

* Local JSON prompt store with read/write/update capability
* Session-level memory and conversation summaries stored locally
* Interaction logs in JSONL, one per session, append-only for traceability
* Purge method to fully delete all user data for a session
* Vectorized retrieval with ChromaDB for fast semantic search of maintenance reports
* Chunking configuration remains in MongoDB (non-sensitive, versioned)

---

## 🚀 Learning Outcomes

Students who complete this assignment understand:

1. How to **differentiate user-owned data from system knowledge**
2. Why **local storage for sensitive data** may be required in enterprise AI
3. How to implement **right-to-erasure** and data privacy controls
4. How to refactor a RAG system to meet **privacy and compliance requirements**
5. How **enterprise RAG architecture** can split data across secure stores without losing functionality

---

## ⚡ Summary

This solution is a **privacy-aware, enterprise-grade RAG system** for fleet maintenance reporting:

* **MongoDB:** stores non-sensitive knowledge-base and configuration
* **Local file system:** stores all user-owned prompts, session memory, and interaction logs
* **ChromaDB & Azure OpenAI:** enable semantic search and LLM-powered responses

It demonstrates a **responsible AI design pattern** where **user data ownership and privacy are respected**.
