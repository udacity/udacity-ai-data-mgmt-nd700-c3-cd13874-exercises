# GenAI Access Control Demo (SP5)

A reproducible **Generative AI access-controlled RAG chatbot** for supply-chain intelligence, demonstrating how **role-based access control must be enforced before retrieval in GenAI pipelines**.

Secrets are stored in **Azure Key Vault**, documents are stored in **Azure Blob Storage**, embeddings and chat are powered by **Azure OpenAI**, and the vector store is built with **Chroma**.

This demo illustrates how **AI systems must enforce data permissions before the model receives context**, preventing unauthorized information from being retrieved or generated.

---

## What This Demo Teaches

In a production GenAI system, **security must be applied across the entire AI data pipeline**, not just the application interface.

This demo shows how role-based access control interacts with key GenAI components.

| Component             | Role                                            |
| --------------------- | ----------------------------------------------- |
| Azure Blob Storage    | Stores supply-chain documents                   |
| Azure Key Vault       | Securely stores API keys and connection strings |
| Chroma Vector Store   | Stores embeddings used for semantic retrieval   |
| Access Control Policy | Determines which users can retrieve which data  |
| Audit Logging         | Records all access attempts                     |

The central security principle demonstrated is:

> **Access control must occur BEFORE retrieval and BEFORE the LLM receives context.**

If unauthorized documents are filtered out early, the model cannot generate responses from them.

---

## Architecture

```
Azure Blob Storage (documents)
        │
        │ load_documents_from_blob()
        ▼
Document metadata + access control filtering
        │
        ▼
Authorized document set
        │
        ▼
Chroma Vector Store (embeddings)
        │
        ▼
Retriever (top-k)
        │
        ▼
Azure OpenAI Chat Model
        │
        ▼
Access-controlled AI response
```

Unauthorized documents are **never embedded or retrieved for the user session**.

---

## Project Structure

```
genai-access-control-demo/
│
├── chat.py                     # Access-controlled RAG chatbot
├── audit_logger.py             # Logs access attempts
├── vector_store.py             # Embedding + retrieval logic
│
├── config/
│   └── policies.json           # Role-based access policies
│
├── automation/
│   ├── assign_roles.py         # Python IAM automation example
│   └── query_audit_logs.py     # Programmatic audit log analysis
│
├── logs/
│   └── access_log.json         # Access log output
│
└── requirements.txt
```

Documents themselves are **stored in Azure Blob Storage**, not in the repository.

---

## Supply Chain Scenario

The assistant answers questions about supply-chain operations across four operational stages.

| Stage         | Description                             |
| ------------- | --------------------------------------- |
| procurement   | Supplier and raw material information   |
| manufacturing | Production output and plant operations  |
| warehousing   | Inventory and storage capacity          |
| distribution  | Delivery and transportation performance |

Example documents stored in Azure Blob Storage:

```
supplier_delay_q1.txt
plant_output_week10.txt
houston_inventory_report.txt
carrier_performance_report.txt
```

These files are stored in a **single Azure Blob container**. The application determines the supply-chain stage from the filename.

---

## Roles and Access Policies

Access control is defined using role-based policies.

| Role                  | Access                     |
| --------------------- | -------------------------- |
| supply_planner        | procurement, manufacturing |
| logistics_coordinator | warehousing, distribution  |
| data_scientist        | all stages                 |

Example policy file — `config/policies.json`:

```json
{
  "supply_planner": {
    "read_stages": ["procurement", "manufacturing"],
    "write_embeddings": false
  },
  "logistics_coordinator": {
    "read_stages": ["warehousing", "distribution"],
    "write_embeddings": false
  },
  "data_scientist": {
    "read_stages": ["procurement", "manufacturing", "warehousing", "distribution"],
    "write_embeddings": true
  }
}
```

The `write_embeddings` permission demonstrates that **vector databases require their own access policies**.

---

## Azure Services Used

| Service            | Purpose                             |
| ------------------ | ----------------------------------- |
| Azure Blob Storage | Stores supply-chain documents       |
| Azure Key Vault    | Manages secrets securely            |
| Azure OpenAI       | Provides embeddings and chat models |

---

## Cloud Concept Mapping

Although this demo uses Azure, the architecture concepts transfer across clouds.

| Azure               | AWS             | GCP              |
| ------------------- | --------------- | ---------------- |
| Entra ID (Azure AD) | AWS IAM         | Cloud IAM        |
| Azure RBAC          | IAM Policies    | IAM Roles        |
| Azure Key Vault     | Secrets Manager | Secret Manager   |
| Azure Monitor Logs  | CloudTrail      | Cloud Audit Logs |

---

## Prerequisites

- Python **3.10+**
- An **Azure OpenAI resource**
- An **Azure Storage Account**
- An **Azure Key Vault**
- Git installed

---

## Azure Setup

### 1. Create Blob Storage Container

Create a container such as:

```
supply-chain-docs
```

Upload the example documents:

```
supplier_delay_q1.txt
plant_output_week10.txt
houston_inventory_report.txt
carrier_performance_report.txt
```

All files can exist **in the root of the container**.

### 2. Configure Azure Key Vault

The demo retrieves secrets from Azure Key Vault at runtime. Create the following secrets:

```
azure-openai-endpoint
azure-openai-key
azure-openai-api-version
azure-openai-chat-deployment
azure-openai-embedding-deployment
azure-storage-connection-string
```

The Key Vault URI is configured inside `chat.py`. Example:

```
https://your-vault-name.vault.azure.net/
```

---

## Setup

### Clone the repository

```bash
git clone <repo-url>
cd genai-access-control-demo
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Chatbot

Start the chatbot:

```bash
python chat.py
```

You will be prompted to select a role:

```
Enter role: supply_planner
```

---

## Example Queries

### Supply Planner

**Allowed stages:** `procurement`, `manufacturing`

```
Are there supplier delays?
What was plant output last week?
```

**Restricted example:**

```
How much inventory is in Houston?
```

**Expected result:**

```
No authorized information available
```

---

### Logistics Coordinator

**Allowed stages:** `warehousing`, `distribution`

```
How much inventory do we have?
Are deliveries on time?
```

**Restricted example:**

```
Which suppliers are delayed?
```

---

### Data Scientist

**Allowed stages:** all stages

```
Do we have product inventory?
What production issues occurred?
```

This role can also **build embeddings for the vector store**.

---

## Access Logging

Every request is recorded in `logs/access_log.json`.

Example log entry:

```json
{
  "timestamp": "2026-03-10T01:30:11",
  "user": "supply_planner",
  "action": "query",
  "resource": "inventory levels",
  "status": "denied"
}
```

Logs capture the user role, query text, accessed resource, and approval or denial status. This supports **security auditing and compliance**.

---

## Querying Audit Logs

Run the audit analysis script:

```bash
python automation/query_audit_logs.py
```

Example output:

```
Denied Access Attempts
logistics_coordinator -> supplier delay query
```

---

## IAM Automation Example

Add a new role using Python:

```bash
python automation/assign_roles.py
```

Example new role: `procurement_manager` with access to `procurement` only.

This demonstrates **automating access policies programmatically**.

---

## Security Lessons

| Principle                | Explanation                                  |
| ------------------------ | -------------------------------------------- |
| Retrieval filtering      | Unauthorized documents never reach the model |
| Least privilege          | Each role only accesses required data        |
| Secret management        | Credentials stored in Key Vault              |
| Vector store permissions | Embedding write access restricted            |
| Audit logging            | All access attempts tracked                  |

---

## Requirements

```
chromadb
langchain
langchain-openai
langchain-chroma
azure-identity
azure-keyvault-secrets
azure-storage-blob
openai
```
