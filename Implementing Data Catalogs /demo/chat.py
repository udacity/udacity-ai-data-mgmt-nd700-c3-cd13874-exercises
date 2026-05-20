# ============================================================
# catalog_rag_demo.py
# Data Catalog + Azure OpenAI Embeddings + Chroma
# Retail / Supply Chain Manuals Demo
# ============================================================

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from azure.keyvault.secrets import SecretClient
from azure.identity import DeviceCodeCredential

# ============================================================
# LOAD SECRETS FROM AZURE KEY VAULT
# ============================================================

def load_secrets():

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


class CatalogRAGDemo:
    """
    Demo RAG system showing how a data catalog controls document selection
    before indexing into a vector store.

    Use case:
    Retail / supply chain manuals

    Core lesson:
    The catalog is not just documentation.
    It acts as a selection and governance layer for GenAI.
    """

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        azure_endpoint: str,
        azure_key: str,
        embedding_deployment: str,
        chat_deployment: str,
        azure_api_version: str = "2024-02-15-preview",
        chroma_path: str = "./catalog_demo_chroma",
        catalog_path: str = "./catalog.json",
        docs_path: str = "./manuals",
    ):
        self.catalog_path = Path(catalog_path)
        self.docs_path = Path(docs_path)

        # ---------------------------
        # Azure Embeddings
        # ---------------------------
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=embedding_deployment,
            openai_api_version=azure_api_version,
            api_key=azure_key,
        )

        # ---------------------------
        # Azure Chat Model
        # ---------------------------
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=chat_deployment,
            openai_api_version=azure_api_version,
            openai_api_key=azure_key,
            temperature=0.2,
        )

        # ---------------------------
        # Chroma Vector Store
        # ---------------------------
        self.vector_db = Chroma(
            collection_name="catalog_rag_manuals",
            embedding_function=self.embeddings,
            persist_directory=chroma_path,
        )

    # ============================================================
    # SAMPLE DATA SETUP
    # ============================================================

    def create_sample_data(self):
        """
        Create a small set of manuals plus a catalog file.
        This avoids needing 100 PDFs and keeps the demo fast.
        """

        self.docs_path.mkdir(parents=True, exist_ok=True)

        manuals = [
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
                "approved_for_genai": True,
                "keywords": ["returns", "refunds", "receipts", "damaged items"],
                "content": """
Customers may return unopened items within 30 days with a receipt.
Damaged items must be logged before refund approval.
Exchanges require SKU verification and inventory confirmation.
Managers approve exceptions for missing receipts.
"""
            },
            {
                "document_id": "DOC002",
                "file_name": "warehouse_picking_guide.txt",
                "title": "Warehouse Picking and Packing Guide",
                "domain": "warehouse",
                "document_type": "manual",
                "owner": "Distribution Center Operations",
                "steward": "Supply Chain Governance",
                "classification": "internal",
                "quality_score": 91,
                "last_updated": "2025-07-15",
                "approved_for_genai": True,
                "keywords": ["picking", "packing", "scanner", "labels", "staging"],
                "content": """
Pickers scan each tote before selecting items from the assigned aisle.
Fragile items require protective packing material.
Packing associates verify item count, print shipping labels,
and place finished packages in the outbound staging lane.
"""
            },
            {
                "document_id": "DOC003",
                "file_name": "shipping_delay_playbook.txt",
                "title": "Shipping Delay Response Playbook",
                "domain": "shipping",
                "document_type": "playbook",
                "owner": "Logistics",
                "steward": "Supply Chain Governance",
                "classification": "internal",
                "quality_score": 89,
                "last_updated": "2025-06-20",
                "approved_for_genai": True,
                "keywords": ["carrier delay", "late delivery", "escalation", "incident"],
                "content": """
When a carrier delay exceeds 24 hours, customer support must be notified.
Urgent orders should be escalated to the logistics coordinator.
If more than 50 orders in one region are affected, an incident ticket must be opened.
"""
            },
            {
                "document_id": "DOC004",
                "file_name": "inventory_cycle_count.txt",
                "title": "Inventory Cycle Count Procedure",
                "domain": "inventory",
                "document_type": "manual",
                "owner": "Inventory Control",
                "steward": "Master Data Team",
                "classification": "internal",
                "quality_score": 96,
                "last_updated": "2025-05-12",
                "approved_for_genai": True,
                "keywords": ["cycle count", "variance", "stock adjustment", "shrinkage"],
                "content": """
Cycle counts are performed weekly by zone.
If variance exceeds 5 units, a recount is required.
Stock adjustments must be approved by the inventory supervisor.
Shrinkage trends should be escalated to store leadership.
"""
            },
            {
                "document_id": "DOC005",
                "file_name": "employee_handbook.txt",
                "title": "Employee Handbook",
                "domain": "hr",
                "document_type": "policy",
                "owner": "Human Resources",
                "steward": "HR Governance",
                "classification": "internal",
                "quality_score": 90,
                "last_updated": "2025-03-01",
                "approved_for_genai": True,
                "keywords": ["leave", "attendance", "benefits", "conduct"],
                "content": """
This handbook includes attendance, benefits, leave, and employee conduct policies.
You may take up to 3 weeks of paid leave in a year.
"""
            },
            {
                "document_id": "DOC006",
                "file_name": "supplier_pricing_terms.txt",
                "title": "Supplier Pricing Terms",
                "domain": "procurement",
                "document_type": "reference",
                "owner": "Procurement",
                "steward": "Legal Data Steward",
                "classification": "confidential",
                "quality_score": 92,
                "last_updated": "2025-04-10",
                "approved_for_genai": False,
                "keywords": ["supplier", "pricing", "contract", "renewal"],
                "content": """
This file contains negotiated supplier pricing, commercial terms, and contract clauses.
"""
            },
            {
                "document_id": "DOC007",
                "file_name": "legacy_store_sop.txt",
                "title": "Legacy Store SOP",
                "domain": "store_operations",
                "document_type": "manual",
                "owner": "Retail Operations",
                "steward": "Data Governance",
                "classification": "internal",
                "quality_score": 62,
                "last_updated": "2021-09-15",
                "approved_for_genai": True,
                "keywords": ["legacy", "register", "nightly close"],
                "content": """
This is an outdated operating procedure for older store systems no longer in production.
"""
            }
        ]

        catalog_records = []

        for manual in manuals:
            file_path = self.docs_path / manual["file_name"]
            file_path.write_text(manual["content"].strip(), encoding="utf-8")

            catalog_record = {k: v for k, v in manual.items() if k != "content"}
            catalog_records.append(catalog_record)

        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog_records, f, indent=2)

        print(f"Created sample manuals in: {self.docs_path}")
        print(f"Created catalog file: {self.catalog_path}")

    # ============================================================
    # CATALOG LOADING
    # ============================================================

    def load_catalog(self) -> List[Dict[str, Any]]:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ============================================================
    # CATALOG FILTERING
    # ============================================================

    def filter_catalog_for_genai(self) -> List[Dict[str, Any]]:
        """
        Use metadata to select which documents may enter the RAG pipeline.
        """

        catalog = self.load_catalog()
        included = []

        print("\n==============================")
        print("CATALOG FILTER DECISIONS")
        print("==============================")

        for doc in catalog:
            reasons = []

            if not doc.get("approved_for_genai", False):
                reasons.append("not approved for GenAI")

            if doc.get("classification") in ["restricted", "confidential"]:
                reasons.append(f"blocked classification: {doc.get('classification')}")

            if doc.get("domain") not in [
                "store_operations",
                "warehouse",
                "shipping",
                "inventory",
                "hr"
            ]:
                reasons.append(f"domain not allowed: {doc.get('domain')}")

            if doc.get("quality_score", 0) < 80:
                reasons.append(f"quality too low: {doc.get('quality_score')}")

            updated_year = int(doc["last_updated"].split("-")[0])
            if updated_year < 2024:
                reasons.append(f"too old: {doc.get('last_updated')}")

            if reasons:
                print(f"EXCLUDE  {doc['document_id']}  {doc['title']}")
                for r in reasons:
                    print(f"   - {r}")
            else:
                included.append(doc)
                print(f"INCLUDE  {doc['document_id']}  {doc['title']}")

        return included

    # ============================================================
    # CHUNKING
    # ============================================================

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index_from_catalog(self):
        """
        Build the Chroma index only from documents selected by the catalog.
        """

        selected_docs = self.filter_catalog_for_genai()

        texts = []
        ids = []
        metadatas = []

        print("\n==============================")
        print("BUILDING VECTOR INDEX")
        print("==============================")

        existing = self.vector_db._collection.get()

        if existing and existing.get("ids"):
            print(f"Clearing {len(existing['ids'])} existing chunk(s) from Chroma...")
            self.vector_db._collection.delete(ids=existing["ids"])
        else:
            print("Chroma collection is already empty.")

        for doc in selected_docs:
            file_path = self.docs_path / doc["file_name"]

            if not file_path.exists():
                print(f"Skipping missing file: {file_path}")
                continue

            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            chunks = self.chunk_text(content)

            print(f"Processing {doc['document_id']} | {doc['title']}")
            print(f" - {len(chunks)} chunk(s) created")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['document_id']}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                texts.append(chunk)
                ids.append(chunk_id)
                metadatas.append({
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "domain": doc["domain"],
                    "owner": doc["owner"],
                    "steward": doc["steward"],
                    "classification": doc["classification"],
                    "quality_score": doc["quality_score"],
                    "last_updated": doc["last_updated"],
                    "chunk_index": i,
                    "indexed_at": str(datetime.datetime.utcnow()),
                })

        if not texts:
            print("No documents selected for indexing.")
            return

        print(f"\nEmbedding and indexing {len(texts)} chunk(s)...")

        self.vector_db.add_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
        )

        print("Index build complete.")
        print("Vector DB size:", self.vector_db._collection.count())

    # ============================================================
    # SEARCH
    # ============================================================

    def retrieve(self, question: str, k: int = 3):
        return self.vector_db.similarity_search(question, k=k)

    # ============================================================
    # ASK
    # ============================================================

    def ask(self, question: str, k: int = 3) -> str:
        """
        Retrieve relevant chunks and answer using Azure OpenAI chat.
        """

        results = self.retrieve(question, k=k)

        if not results:
            return "No relevant manuals were found."

        context_parts = []
        source_info = []

        for doc in results:
            context_parts.append(doc.page_content)
            source_info.append(
                f"{doc.metadata.get('document_id')} | "
                f"{doc.metadata.get('title')} | "
                f"{doc.metadata.get('domain')}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a retail and supply chain operations assistant.
Answer using only the retrieved manual content.
If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

        response = self.llm.invoke(prompt)

        final_answer = response.content

        print("\n==============================")
        print("RETRIEVED SOURCES")
        print("==============================")
        for s in source_info:
            print("-", s)

        return final_answer


# ============================================================
# MAIN
# ============================================================

def main():

    secrets = load_secrets()

    AZURE_ENDPOINT = secrets["azure_endpoint"]
    AZURE_KEY = secrets["azure_key"]
    EMBEDDING_DEPLOYMENT = secrets["embedding_deployment"]
    CHAT_DEPLOYMENT = secrets["chat_deployment"]
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([AZURE_ENDPOINT, AZURE_KEY, EMBEDDING_DEPLOYMENT, CHAT_DEPLOYMENT]):
        raise ValueError(
            "Missing Azure OpenAI environment variables.\n"
            "Set:\n"
            "AZURE_OPENAI_ENDPOINT\n"
            "AZURE_OPENAI_KEY\n"
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT\n"
            "AZURE_OPENAI_CHAT_DEPLOYMENT\n"
            "AZURE_OPENAI_API_VERSION (optional)"
        )

    demo = CatalogRAGDemo(
        azure_endpoint=AZURE_ENDPOINT,
        azure_key=AZURE_KEY,
        embedding_deployment=EMBEDDING_DEPLOYMENT,
        chat_deployment=CHAT_DEPLOYMENT,
        azure_api_version=API_VERSION,
        chroma_path="./catalog_demo_chroma",
        catalog_path="./catalog.json",
        docs_path="./manuals",
    )

    # Create sample manuals + catalog
    demo.create_sample_data()

    # Build governed index
    demo.build_index_from_catalog()

    # Ask a few sample questions
    questions = [
        "What should staff do when a carrier delay affects many orders?",
        "How should store employees handle returns for damaged items?",
        "What happens when cycle count variance exceeds five units?",
        "What are the employee leave policies?"
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print("QUESTION:", q)
        print("=" * 80)
        answer = demo.ask(q, k=3)
        print("\nANSWER:")
        print(answer)


if __name__ == "__main__":
    main()