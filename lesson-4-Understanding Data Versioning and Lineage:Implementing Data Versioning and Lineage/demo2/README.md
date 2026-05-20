# Supermarket RAG + DVC Demo

A reproducible **Retrieval-Augmented Generation (RAG)** pipeline for supermarket supply-chain Q&A, demonstrating how to version-control not just code, but also **prompt artifacts** and **vector indexes** using [DVC](https://dvc.org/).

Secrets are stored in **Azure Key Vault**, embeddings and chat are powered by **Azure OpenAI**, and the vector store is built with **Chroma**.

---

## What this demo teaches

In a production GenAI system, three things need lineage and reproducibility — not just source code:

| Artifact | Role |
|---|---|
| `data/inventory.csv` | Source knowledge for retrieval |
| `artifacts/system_prompt.txt` | Operating instructions + current business context |
| `artifacts/chroma_db/` | Retrieval representation of the knowledge base |

DVC tracks all three through a dependency graph. When the inventory changes, `dvc repro` automatically rebuilds the prompt artifact and the vector index.

---

## Architecture

```
inventory.csv  ──►  build_prompt  ──►  system_prompt.txt
     │                                        │
     └──────────────────────────►  build_index  ──►  chroma_db/
                                                       │
                                              chat.py (RAG chatbot)
```

---

## Project Structure

```
supermarket-rag-dvc/
├── data/
│   └── inventory.csv          # Supermarket catalog + supply-chain snapshot
├── prompts/
│   └── base_rules.txt         # Business rules for the assistant
├── artifacts/
│   ├── system_prompt.txt      # Generated prompt artifact (DVC output)
│   ├── chunks.jsonl           # Chunked documents (DVC output)
│   ├── metrics.json           # Pipeline metrics (DVC metrics)
│   └── chroma_db/             # Persisted vector index (DVC output)
├── src/
│   ├── common.py              # Azure Key Vault + LangChain client helpers
│   ├── build_prompt.py        # Stage 1: builds the system prompt artifact
│   ├── build_index.py         # Stage 2: embeds chunks and builds Chroma index
│   └── chat.py                # RAG chatbot (retrieve + generate)
├── params.yaml                # Deployment names, RAG params, chunk size
├── dvc.yaml                   # DVC pipeline definition
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- An **Azure OpenAI** resource with two deployments:
  - A chat model (e.g. `gpt-4o`)
  - An embeddings model (e.g. `text-embedding-3-small`)
- An **Azure Key Vault** containing three secrets:
  - `azure-openai-endpoint`
  - `azure-openai-key`
  - `azure-openai-api-version`
- Git installed

---

## Setup

**1. Clone and install dependencies**

```bash
git clone <your-repo-url>
cd supermarket-rag-dvc
pip install -r requirements.txt
```

**2. Initialize Git and DVC**

```bash
git init
dvc init
```

**3. Configure deployment names**

Edit `params.yaml` to match your Azure OpenAI deployment names:

```yaml
azure:
  chat_deployment: "gpt-4o"
  embedding_deployment: "text-embedding-3-small"

rag:
  collection_name: "supermarket_inventory"
  k: 4

chunking:
  max_products_per_chunk: 2
```

---

## Running the Pipeline

**Track the inventory with DVC and run the first build:**

```bash
dvc add data/inventory.csv
git add .
git commit -m "Initial supermarket inventory v1"

dvc repro
git add .
git commit -m "Build prompt and vector index for inventory v1"
git tag inventory-v1
```

`dvc repro` runs `build_prompt` then `build_index` in dependency order, and caches the outputs.

**Start the chatbot:**

```bash
python src/chat.py
```

Try these example questions:

```
Which products are low stock?
Who supplies eggs?
Do we carry Greek Yogurt?
```

---

## Inventory Update (v2)

Add a new product to `data/inventory.csv` (Greek Yogurt, product ID 1006), then rebuild:

```bash
dvc repro
git add .
git commit -m "Inventory v2 adds Greek Yogurt and refreshes prompt and embeddings"
git tag inventory-v2
```

Because `build_prompt` depends on `inventory.csv`, and `build_index` depends on both `inventory.csv` and `system_prompt.txt`, DVC reruns both stages automatically.

Ask again:

```
Do we carry Greek Yogurt?
```

The assistant now retrieves and answers from the new product record.

---

## Comparing Versions

**Compare metrics between inventory snapshots:**

```bash
dvc metrics diff inventory-v1 inventory-v2
```

Expected output: `num_products` increases from 5 → 6, `num_chunks` may change depending on chunk size.

**Compare parameters:**

```bash
dvc params diff inventory-v1 inventory-v2
```

---

## Rolling Back

To restore the pipeline to an earlier state:

```bash
git checkout inventory-v1
dvc checkout
dvc repro
python src/chat.py
```

The assistant will no longer find Greek Yogurt, demonstrating full reproducibility across versions.

---

## How Secrets Are Managed

`src/common.py` authenticates to Azure Key Vault using `InteractiveBrowserCredential` and retrieves the endpoint, API key, and API version at runtime. No secrets are stored in code or config files, following Microsoft's recommended pattern for Azure OpenAI key management.

The Key Vault URI is set to `https://useraisecrets.vault.azure.net/` — update this to match your own vault name.

---

## DVC Pipeline Reference (`dvc.yaml`)

```yaml
stages:
  build_prompt:
    cmd: python src/build_prompt.py
    deps:
      - data/inventory.csv
      - prompts/base_rules.txt
      - src/build_prompt.py
    outs:
      - artifacts/system_prompt.txt

  build_index:
    cmd: python src/build_index.py
    deps:
      - data/inventory.csv
      - artifacts/system_prompt.txt
      - src/build_index.py
      - src/common.py
      - params.yaml
    metrics:
      - artifacts/metrics.json
    outs:
      - artifacts/chunks.jsonl
      - artifacts/chroma_db
```

---

## Key Concepts

| Concept | How it's used here |
|---|---|
| **DVC stages** | Declare deps/outputs so changes propagate automatically |
| **DVC metrics** | Track `num_products`, `num_chunks`, `prompt_chars` across runs |
| **DVC params** | Version deployment names and RAG settings in `params.yaml` |
| **Prompt artifact** | System prompt is a *generated file*, not hardcoded |
| **Chroma** | Persistent local vector store, tracked as a DVC output |
| **Azure Key Vault** | Secrets never touch the codebase or version control |

---

## Requirements

```
pandas
pyyaml
dvc
chromadb
langchain
langchain-openai
langchain-chroma
azure-identity
azure-keyvault-secrets
```
