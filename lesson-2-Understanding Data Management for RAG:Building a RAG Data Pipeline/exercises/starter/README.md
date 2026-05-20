# Enterprise RAG – Starter Code for Fleet Maintenance Assignment

This project provides a **starter codebase for an Enterprise RAG (Retrieval-Augmented Generation)** system. The system is designed to answer questions about **fleet maintenance reports** using a combination of:

* **MongoDB (Cosmos DB Mongo API)** → system of record for knowledge-base documents and RAG configuration
* **ChromaDB** → vector store for semantic search
* **Azure OpenAI** → embeddings and chat model

---

## 🎯 Assignment Overview

The goal of this assignment is to **refactor and complete the RAG system** to implement **privacy-conscious user data storage**.

Students are asked to:

1. Identify which parts of the original RAG code contain **user-owned data** (prompts, session memory, interactions)
2. Ensure all **user-owned data is stored locally**, rather than in MongoDB or the cloud
3. Keep **MongoDB strictly for non-sensitive data**, such as:

   * RAG configuration (chunk size, overlap)
   * Fleet maintenance reports (knowledge base, no PII)
4. Implement **session-based local storage**:

   * Prompts in a JSON file
   * Memory as JSON per session
   * Interaction logs as JSONL per session
5. Support **right-to-erasure**: ability to delete all local session data on demand

---

## 🧰 Data Storage Overview (Starter Code)

| Data Type                 | Storage                       | Notes                                                     |
| ------------------------- | ----------------------------- | --------------------------------------------------------- |
| RAG Configuration         | MongoDB                       | Non-sensitive operational parameters                      |
| Fleet Maintenance Reports | MongoDB                       | Knowledge base (truck maintenance reports)                |
| Prompts                   | **To be implemented locally** | System/user prompts; sensitive user-owned data            |
| Session Memory            | **To be implemented locally** | Summaries of user conversations                           |
| Interaction Logs          | **To be implemented locally** | Raw questions, answers, and retrieved document references |

> In the starter code, **prompts, memory, and interactions are still partially in MongoDB**. Completing the assignment requires **moving them to local storage** and ensuring they are **fully controlled by the user**.

---

## 🔹 Starter Code Features

The starter code already includes:

* MongoDB setup for **RAG configuration and knowledge-base documents**
* Chroma vector store initialization for semantic search
* Azure OpenAI embeddings and chat LLM integration
* Initial RAG query method skeleton

The code is partially implemented; students need to **complete the following sections**:

* **Prompt Management:** load, save, and update prompts in local JSON
* **Memory Management:** store conversation summaries per session locally
* **Interaction Logging:** write session interactions to local JSONL files
* **Right-to-Erasure:** delete all local session data for a user

---

## 🔹 Assignment Requirements

When completing the starter code, ensure that:

1. **Local storage paths** are created automatically if missing (`./local_store/prompts.json`, `./local_store/memory/`, `./local_store/interactions/`)
2. **Session-based separation** is implemented for memory and interaction logs
3. **RAG queries** read memory and prompts locally and update them locally after each user interaction
4. **Right-to-erasure methods** can purge all local user data for a given session
5. MongoDB is **only used for non-sensitive enterprise knowledge and configuration**, never user-specific conversation data

---

## 🧩 Learning Objectives

By completing this assignment, students will learn:

* How to **separate user-owned data from system knowledge** in an enterprise RAG system
* How to **implement local storage** for privacy-sensitive data
* How to **manage session memory, prompts, and interaction logs** locally
* How to implement **right-to-erasure** in a privacy-aware application
* How **enterprise RAG architectures** split sensitive and non-sensitive data between secure stores

---

## ⚡ Key Takeaways

* RAG is a **data-driven system**, not just an AI model
* **Prompts, memory, and interactions** are **user-owned assets** that must be stored securely
* **MongoDB** contains only **non-sensitive knowledge** (fleet maintenance reports, configuration)
* **Local storage** ensures **user data privacy and ownership**
* Right-to-erasure supports **GDPR/CCPA compliance**

---

## 📝 Notes for Students

* Use the starter code as a scaffold; **do not overwrite MongoDB for user data**
* Ensure all local storage paths exist before writing files
* Test RAG queries to confirm that **local memory and interactions persist across sessions**
* Implement methods to **update prompts locally**
* Ensure **vector search and document retrieval** still function with ChromaDB + MongoDB
