# ============================================================
# starter_chat.py
# Starter Code: Data Catalog + Azure OpenAI RAG Exercise
# Azure Key Vault + Chroma + Azure OpenAI
# ============================================================
#
# Instructions:
# Complete the TODOs in this file to build a simple catalog-driven
# RAG demo for retail / supply chain manuals.
#
# Goals:
# 1. Load secrets from Azure Key Vault
# 2. Create sample manuals and a catalog
# 3. Classify documents automatically using keyword matching
# 4. Export Safety documents from the catalog
# 5. Build an in-memory Chroma index using Azure embeddings
# 6. Ask questions and show which documents were retrieved
# ============================================================

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from azure.keyvault.secrets import SecretClient
from azure.identity import InteractiveBrowserCredential


# ============================================================
# LOAD SECRETS FROM AZURE KEY VAULT
# ============================================================

def load_secrets() -> Dict[str, str]:
    """
    Authenticate with Azure Key Vault and return the secrets
    needed to connect to Azure OpenAI.
    """

    print("Authenticating with Azure...")

    # TODO 1:
    # Create an InteractiveBrowserCredential
    credential = None

    # TODO 2:
    # Set the Key Vault URL
    kv_uri = "https://useraisecrets.vault.azure.net/"

    # TODO 3:
    # Create a SecretClient using the vault URL and credential
    client = None

    # TODO 4:
    # Read the Azure OpenAI endpoint and key from Key Vault.
    # Use these secret names:
    # - unstructuredazureendpoint
    # - unstructuredazurekey
    secrets = {
        "azure_endpoint": "",
        "azure_key": "",
        "embedding_deployment": "text-embedding-ada-002",
        "chat_deployment": "gpt-4.1-mini",
    }

    return secrets


# ============================================================
# MAIN CLASS
# ============================================================

class CatalogRAGStarter:
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

        # Keyword dictionary used for automatic classification
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
                "store", "returns", "refund", "receipt", "exchange", "damaged"
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
        """
        Create sample manuals and the initial catalog file.
        """

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
            },
        ]

        catalog = []

        # TODO 5:
        # Write each manual's content to a text file inside self.docs_path.
        # Then create a catalog record that excludes the "content" field.
        # Append each record to the catalog list.

        # TODO 6:
        # Save the catalog list to self.catalog_path as JSON.

        print("Sample manuals and catalog created.")

    # ============================================================
    # LOAD / SAVE CATALOG
    # ============================================================

    def load_catalog(self) -> List[Dict]:
        # TODO 7:
        # Open self.catalog_path and return the parsed JSON.
        return []

    def save_catalog(self, catalog: List[Dict]) -> None:
        # TODO 8:
        # Save the updated catalog back to self.catalog_path.
        pass

    # ============================================================
    # AUTOMATED CLASSIFICATION
    # ============================================================

    def classify_documents(self) -> None:
        """
        Add a category field to each catalog record using keyword matching.
        """

        catalog = self.load_catalog()

        print("\nRunning automated classification...")

        for doc in catalog:
            text_path = self.docs_path / doc["file_name"]
            text = text_path.read_text(encoding="utf-8").lower()

            scores = {}

            # TODO 9:
            # For each category in self.category_keywords,
            # count how many keywords appear in the document text.
            # Store the result in the scores dictionary.

            # TODO 10:
            # Pick the category with the highest score.
            # If the best score is 0, assign "Other".
            doc["category"] = "Other"

            print(f"{doc['title']} -> {doc['category']}")

        # TODO 11:
        # Save the updated catalog with the new category field.
        pass

    # ============================================================
    # EXPORT SAFETY DOCUMENTS
    # ============================================================

    def export_safety_docs(self) -> None:
        """
        Query the catalog for Safety documents and export them.
        """

        catalog = self.load_catalog()

        # TODO 12:
        # Filter the catalog to keep only documents whose category is "Safety".
        safety_docs = []

        # TODO 13:
        # Save safety_docs to self.safety_export_path as JSON.

        print("Safety docs exported.")

    # ============================================================
    # VECTOR STORE
    # ============================================================

    def reset_vector_store(self) -> None:
        """
        Create a fresh in-memory Chroma vector store.
        """

        # TODO 14:
        # Create a new Chroma collection in memory.
        # Use a unique collection name with uuid.uuid4().
        self.vector_db = None

        print("Vector store reset in memory.")

    def build_index(self) -> None:
        """
        Build the vector index from approved catalog documents.
        """

        if self.vector_db is None:
            self.reset_vector_store()

        catalog = self.load_catalog()

        texts = []
        metadatas = []
        ids = []

        for doc in catalog:
            # TODO 15:
            # Skip documents that are not approved for GenAI.
            # Skip documents whose classification is restricted or confidential.

            text_path = self.docs_path / doc["file_name"]
            text = text_path.read_text(encoding="utf-8").strip()

            if not text:
                continue

            # TODO 16:
            # Append the document text to texts.
            # Create a UUID string and append it to ids.
            # Append useful metadata to metadatas:
            # document_id, file_name, title, domain,
            # classification, quality_score, last_updated,
            # approved_for_genai, category

        if not texts:
            print("No documents available for indexing.")
            return

        # TODO 17:
        # Add texts, ids, and metadatas to the Chroma vector store.
        print("Vector index built.")

    # ============================================================
    # ASK
    # ============================================================

    def ask(self, question: str, k: int = 3) -> str:
        """
        Retrieve relevant documents and ask the chat model to answer.
        """

        if self.vector_db is None:
            return "Vector index has not been built."

        # TODO 18:
        # Run similarity_search on the vector store.
        results = []

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

        # TODO 19:
        # Combine the retrieved page_content values into one context string.
        context = ""

        prompt = f"""
Use the context to answer the question.
If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question:
{question}
"""

        # TODO 20:
        # Invoke the Azure chat model and return the response text.
        return "TODO"


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # TODO 21:
    # Load secrets from Azure Key Vault.
    secrets = {}

    # TODO 22:
    # Create an instance of CatalogRAGStarter using the loaded secrets.
    demo = None

    # TODO 23:
    # Run the main pipeline steps in order:
    # - create_sample_data()
    # - classify_documents()
    # - export_safety_docs()
    # - reset_vector_store()
    # - build_index()

    questions = [
        "What should employees do if there is a spill in the store?",
        "How should damaged returns be handled?",
        "What happens when a carrier delay exceeds 24 hours?",
        "What should warehouse staff do before sealing a package?",
        "What happens if inventory variance is more than five units?",
        "How should customers be informed about shipping delays?",
        "What should staff do if a customer is affected by a spill?",
        "What are the employee benefits policies?",
    ]

    # TODO 24:
    # Loop through the questions, call demo.ask(question, k=3),
    # and print the question and answer nicely.
    pass


if __name__ == "__main__":
    main()