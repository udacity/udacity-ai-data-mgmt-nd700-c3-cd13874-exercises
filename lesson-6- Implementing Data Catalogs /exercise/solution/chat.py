# ============================================================
# chat.py
# Data Catalog + Azure OpenAI RAG + Classification Exercise
# Azure Key Vault + Chroma + Azure OpenAI
# In-memory Chroma version
# ============================================================

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from azure.keyvault.secrets import SecretClient
from azure.identity import DeviceCodeCredential


# ============================================================
# LOAD SECRETS FROM AZURE KEY VAULT
# ============================================================

def load_secrets() -> Dict[str, str]:

    keyVaultName = "data-management-kv-pk"
    KVUri = f"https://{keyVaultName}.vault.azure.net/"

    print("Connecting to Azure for authentication.")

    credential = DeviceCodeCredential(additionally_allowed_tenants=["*"])
    client = SecretClient(vault_url=KVUri, credential=credential)

    secrets = {
        "azure_endpoint": client.get_secret("unstructuredazureendpoint").value,
        "azure_key": client.get_secret("unstructuredazurekey").value,
        "embedding_deployment": "text-embedding-ada-002",
        "chat_deployment": "gpt-4.1-mini",
    }

    return secrets


# ============================================================
# MAIN CLASS
# ============================================================

class CatalogRAGDemoWithClassification:
    def __init__(
        self,
        azure_endpoint: str,
        azure_key: str,
        embedding_deployment: str,
        chat_deployment: str,
        azure_api_version: str = "2024-02-15-preview",
        catalog_path: str = "./catalog.json",
        docs_path: str = "./manuals",
        safety_export_path: str = "./safety_documents.json",
    ):
        self.catalog_path = Path(catalog_path)
        self.docs_path = Path(docs_path)
        self.safety_export_path = Path(safety_export_path)

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=embedding_deployment,
            openai_api_version=azure_api_version,
            api_key=azure_key,
        )

        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=chat_deployment,
            openai_api_version=azure_api_version,
            openai_api_key=azure_key,
            temperature=0.2,
        )

        self.vector_db: Optional[Chroma] = None

        self.category_keywords = {
            "Safety": [
                "safety", "hazard", "spill", "incident", "injury",
                "danger", "wet floor", "cleanup"
            ],
            "Inventory": [
                "inventory", "cycle count", "stock", "variance", "shrinkage"
            ],
            "Shipping": [
                "shipping", "carrier", "delivery", "shipment", "delay", "urgent"
            ],
            "Warehouse": [
                "warehouse", "picking", "packing", "scanner", "staging", "tote"
            ],
            "Store Operations": [
                "store", "returns", "refund", "receipt", "exchange", "damaged items"
            ],
            "HR": [
                "employee", "benefits", "attendance", "leave", "conduct"
            ],
            "Procurement": [
                "supplier", "contract", "pricing", "vendor"
            ],
        }

    # ============================================================
    # SAMPLE DATA
    # ============================================================

    def create_sample_data(self) -> None:
        self.docs_path.mkdir(parents=True, exist_ok=True)

        manuals = [
            {
                "document_id": "DOC001",
                "file_name": "store_returns_manual.txt",
                "title": "Store Returns Manual",
                "domain": "store_operations",
                "classification": "internal",
                "quality_score": 94,
                "last_updated": "2025-08-01",
                "approved_for_genai": True,
                "content": """
Customers may return unopened items within 30 days with a receipt.
Damaged items must be logged before refund approval.
Managers approve exceptions for missing receipts.
Exchanges require verification of the product and reason for return.
""",
            },
            {
                "document_id": "DOC002",
                "file_name": "warehouse_picking_guide.txt",
                "title": "Warehouse Picking Guide",
                "domain": "warehouse",
                "classification": "internal",
                "quality_score": 91,
                "last_updated": "2025-07-15",
                "approved_for_genai": True,
                "content": """
Pickers scan each tote before selecting items.
Fragile items require protective packing material.
Packages move to outbound staging lanes.
Warehouse staff should confirm item count before sealing the package.
""",
            },
            {
                "document_id": "DOC003",
                "file_name": "shipping_delay_playbook.txt",
                "title": "Shipping Delay Playbook",
                "domain": "shipping",
                "classification": "internal",
                "quality_score": 89,
                "last_updated": "2025-06-20",
                "approved_for_genai": True,
                "content": """
When a carrier delay exceeds 24 hours customer support must be notified.
Urgent orders escalate to logistics coordinator.
If many orders are delayed in one region, an incident should be opened.
Customers should receive a delay notification when service levels are impacted.
""",
            },
            {
                "document_id": "DOC004",
                "file_name": "store_safety_manual.txt",
                "title": "Store Safety Spill Response",
                "domain": "store_operations",
                "classification": "internal",
                "quality_score": 93,
                "last_updated": "2025-07-28",
                "approved_for_genai": True,
                "content": """
If a spill occurs employees must secure the area immediately.
Wet floor signs should be placed and a cleanup kit used.
If a customer is affected, staff should notify the manager and log an incident.
Safety procedures require quick isolation of the hazard area.
""",
            },
            {
                "document_id": "DOC005",
                "file_name": "inventory_cycle_count_manual.txt",
                "title": "Inventory Cycle Count Manual",
                "domain": "inventory",
                "classification": "internal",
                "quality_score": 95,
                "last_updated": "2025-05-03",
                "approved_for_genai": True,
                "content": """
Cycle counts are performed weekly by zone.
If the variance exceeds five units, a recount is required.
Stock adjustments must be approved by the inventory supervisor.
Shrinkage trends should be reviewed by leadership.
""",
            },
            {
                "document_id": "DOC006",
                "file_name": "employee_handbook.txt",
                "title": "Employee Handbook",
                "domain": "hr",
                "classification": "restricted",
                "quality_score": 90,
                "last_updated": "2025-03-01",
                "approved_for_genai": False,
                "content": """
Employee attendance benefits leave and workplace conduct policies.
""",
            }
        ]

        catalog = []

        for manual in manuals:
            file_path = self.docs_path / manual["file_name"]
            file_path.write_text(manual["content"].strip(), encoding="utf-8")
            record = {k: v for k, v in manual.items() if k != "content"}
            catalog.append(record)

        with self.catalog_path.open("w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

        print("Sample manuals and catalog created.")

    # ============================================================
    # LOAD / SAVE CATALOG
    # ============================================================

    def load_catalog(self) -> List[Dict]:
        with self.catalog_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_catalog(self, catalog: List[Dict]) -> None:
        with self.catalog_path.open("w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    # ============================================================
    # AUTOMATED CLASSIFICATION
    # ============================================================

    def classify_documents(self) -> None:
        catalog = self.load_catalog()

        print("\nRunning automated classification...")

        for doc in catalog:
            text_path = self.docs_path / doc["file_name"]
            text = text_path.read_text(encoding="utf-8").lower()

            scores = {}
            for category, keywords in self.category_keywords.items():
                score = 0
                for kw in keywords:
                    if kw in text:
                        score += 1
                scores[category] = score

            best = max(scores, key=scores.get)
            doc["category"] = best if scores[best] > 0 else "Other"

            print(f"{doc['title']} -> {doc['category']}")

        self.save_catalog(catalog)

    # ============================================================
    # EXPORT SAFETY DOCUMENTS
    # ============================================================

    def export_safety_docs(self) -> None:
        catalog = self.load_catalog()
        safety_docs = [doc for doc in catalog if doc.get("category") == "Safety"]

        with self.safety_export_path.open("w", encoding="utf-8") as f:
            json.dump(safety_docs, f, indent=2)

        print("Safety docs exported.")

    # ============================================================
    # VECTOR STORE
    # ============================================================

    def reset_vector_store(self) -> None:
        self.vector_db = Chroma(
            collection_name=f"catalog_rag_manuals_{uuid.uuid4().hex[:8]}",
            embedding_function=self.embeddings,
        )
        print("Vector store reset in memory.")

    def build_index(self) -> None:
        if self.vector_db is None:
            self.reset_vector_store()

        catalog = self.load_catalog()

        texts = []
        metadatas = []
        ids = []

        for doc in catalog:
            if not doc.get("approved_for_genai"):
                continue

            if doc.get("classification") in ["restricted", "confidential"]:
                continue

            text_path = self.docs_path / doc["file_name"]
            text = text_path.read_text(encoding="utf-8").strip()

            if not text:
                continue

            texts.append(text)
            ids.append(str(uuid.uuid4()))
            metadatas.append({
                "document_id": str(doc.get("document_id", "")),
                "file_name": str(doc.get("file_name", "")),
                "title": str(doc.get("title", "")),
                "domain": str(doc.get("domain", "")),
                "classification": str(doc.get("classification", "")),
                "quality_score": int(doc.get("quality_score", 0)),
                "last_updated": str(doc.get("last_updated", "")),
                "approved_for_genai": str(doc.get("approved_for_genai", "")),
                "category": str(doc.get("category", "")),
            })

        if not texts:
            print("No documents available for indexing.")
            return

        self.vector_db.add_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
        )

        print("Vector index built.")

    # ============================================================
    # ASK
    # ============================================================

    def ask(self, question: str, k: int = 3) -> str:
        if self.vector_db is None:
            return "Vector index has not been built."

        results = self.vector_db.similarity_search(question, k=k)

        if not results:
            return "No relevant documents found."

        print("\nDocuments used:")
        for i, doc in enumerate(results, start=1):
            print(
                f"{i}. {doc.metadata.get('document_id')} | "
                f"{doc.metadata.get('title')} | "
                f"{doc.metadata.get('category')} | "
                f"{doc.metadata.get('domain')}"
            )

        context = "\n\n".join([doc.page_content for doc in results])

        prompt = f"""
Use the context to answer the question.
If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question:
{question}
"""

        response = self.llm.invoke(prompt)
        return response.content


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    secrets = load_secrets()

    demo = CatalogRAGDemoWithClassification(
        azure_endpoint=secrets["azure_endpoint"],
        azure_key=secrets["azure_key"],
        embedding_deployment=secrets["embedding_deployment"],
        chat_deployment=secrets["chat_deployment"],
    )

    demo.create_sample_data()
    demo.classify_documents()
    demo.export_safety_docs()
    demo.reset_vector_store()
    demo.build_index()

    questions = [
        "What should employees do if there is a spill in the store?",
        "How should damaged returns be handled?",
        "What happens when a carrier delay exceeds 24 hours?",
        "What should warehouse staff do before sealing a package?",
        "What happens if inventory variance is more than five units?",
        "How should customers be informed about shipping delays?",
        "What should staff do if a customer is affected by a spill?",
        "What are the employee benefits policies?"
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("Question:", question)
        print("=" * 80)
        answer = demo.ask(question, k=3)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()