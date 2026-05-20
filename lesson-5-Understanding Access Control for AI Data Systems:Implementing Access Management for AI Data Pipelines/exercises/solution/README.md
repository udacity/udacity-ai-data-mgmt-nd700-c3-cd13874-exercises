# SP5 — GenAI Access Control Exercise

## Your Task

**Add a 4th role to a working GenAI access-controlled RAG system.**

The system already has three roles with different levels of access to supply-chain data. You will implement a new role — `procurement_manager` — that is narrower than any existing role. This forces you to think carefully about how access policies are defined and enforced, and where in the pipeline those decisions are applied.

---

## The New Role: `procurement_manager`

| Property         | Value                                          |
| ---------------- | ---------------------------------------------- |
| Role name        | `procurement_manager`                          |
| Allowed stages   | `procurement` only                             |
| Blocked stages   | `manufacturing`, `warehousing`, `distribution` |
| Write embeddings | `false`                                        |

This role is intentionally narrower than `supply_planner`, which can read both `procurement` and `manufacturing`. The `procurement_manager` sees procurement data only — nothing else.

---

## What You Need to Change

You will edit **two files**:

### 1. `config/policies.json`

Add the new role entry following the same structure as the existing roles.

> **Do not modify the existing roles.** Only add the new entry.

After your change, the file should contain four roles: `supply_planner`, `logistics_coordinator`, `data_scientist`, and `procurement_manager`.

### 2. `chat.py`

The chatbot prompts the user to enter a role at startup. Make sure `procurement_manager` is a recognised and selectable role. Review how the existing roles are handled and apply the same pattern.

---

## Verify Your Implementation

Start the chatbot and log in as `procurement_manager`:

```
Enter role: procurement_manager
```

**Should be allowed:**

```
Are there any supplier delays?
Which suppliers are at risk?
```

**Expected result:** The chatbot returns relevant procurement information.

**Should be denied:**

```
What was plant output last week?
How much inventory is in Houston?
Are deliveries running on time?
```

**Expected result for each:**

```
No authorized information available
```

---

## Audit Log Check

After running your queries, inspect `logs/access_log.json` and confirm that:

- Allowed queries show `"status": "allowed"` with `"user": "procurement_manager"`
- Denied queries show `"status": "denied"` with `"user": "procurement_manager"`

---

## Reflection Questions

1. How is `procurement_manager` different from `supply_planner`? Why might a real organisation need both roles?
2. At what point in the pipeline is the access decision made — before or after the LLM receives context? Why does this matter?
3. What would happen if access filtering were applied *after* retrieval instead of before?
4. The `write_embeddings` flag is set to `false` for `procurement_manager`. What risk would exist if it were `true` for a role with restricted read access?

---

---

# Background: How the System Works

This exercise uses a **Generative AI access-controlled RAG chatbot** for supply-chain intelligence. Secrets are stored in **Azure Key Vault**, documents are stored in **Azure Blob Storage**, embeddings and chat are powered by **Azure OpenAI**, and the vector store is built with **Chroma**.

The central security principle the system demonstrates is:

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

Documents are **stored in Azure Blob Storage**, not in the repository.

---

## Supply Chain Scenario

The chatbot answers questions about supply-chain operations across four stages.

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

The application determines the supply-chain stage from the filename.

---

## Existing Roles

| Role                  | Allowed Stages             | Write Embeddings |
| --------------------- | -------------------------- | ---------------- |
| supply_planner        | procurement, manufacturing | No               |
| logistics_coordinator | warehousing, distribution  | No               |
| data_scientist        | all stages                 | Yes              |

Current `config/policies.json`:

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

---

## Azure Services

| Service            | Purpose                             |
| ------------------ | ----------------------------------- |
| Azure Blob Storage | Stores supply-chain documents       |
| Azure Key Vault    | Manages secrets securely            |
| Azure OpenAI       | Provides embeddings and chat models |

---

## Cloud Concept Mapping

| Azure               | AWS             | GCP              |
| ------------------- | --------------- | ---------------- |
| Entra ID (Azure AD) | AWS IAM         | Cloud IAM        |
| Azure RBAC          | IAM Policies    | IAM Roles        |
| Azure Key Vault     | Secrets Manager | Secret Manager   |
| Azure Monitor Logs  | CloudTrail      | Cloud Audit Logs |

---

## Prerequisites

- Python **3.10+**
- An Azure OpenAI resource
- An Azure Storage Account
- An Azure Key Vault
- Git installed

---

## Azure Setup

### 1. Create Blob Storage Container

Create a container named `supply-chain-docs` and upload the example documents:

```
supplier_delay_q1.txt
plant_output_week10.txt
houston_inventory_report.txt
carrier_performance_report.txt
```

### 2. Configure Azure Key Vault

Create the following secrets in Key Vault:

```
azure-openai-endpoint
azure-openai-key
azure-openai-api-version
azure-openai-chat-deployment
azure-openai-embedding-deployment
azure-storage-connection-string
```

The Key Vault URI is configured inside `chat.py`:

```
https://your-vault-name.vault.azure.net/
```

---

## Setup

```bash
git clone <repo-url>
cd genai-access-control-demo
pip install -r requirements.txt
```

---

## Access Logging

Every request is recorded in `logs/access_log.json`:

```json
{
  "timestamp": "2026-03-10T01:30:11",
  "user": "supply_planner",
  "action": "query",
  "resource": "inventory levels",
  "status": "denied"
}
```

Run the audit analysis script to review denied attempts:

```bash
python automation/query_audit_logs.py
```

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
