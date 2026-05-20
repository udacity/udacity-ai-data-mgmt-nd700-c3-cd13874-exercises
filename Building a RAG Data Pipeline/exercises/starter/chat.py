# ============================================================
# app.py
# Main Application
# ============================================================

import warnings
warnings.filterwarnings("ignore", message="Field.*conflict with protected namespace")
warnings.filterwarnings("ignore", category=UserWarning)

from azure.keyvault.secrets import SecretClient
from azure.identity import InteractiveBrowserCredential
import uuid

from agents.enterprise_rag import EnterpriseRAG


# ============================================================
# LOAD SECRETS FROM AZURE KEY VAULT
# ============================================================

def load_secrets():

    print("Authenticating with Azure...")

    credential = InteractiveBrowserCredential(additionally_allowed_tenants=["*"])
    KVUri = "https://aiusersecrets.vault.azure.net/"

    client = SecretClient(vault_url=KVUri, credential=credential)

    secrets = {
        "mongo_uri": client.get_secret("unstructuredmongourl").value,
        "mongo_db": client.get_secret("unstructureddbname").value,
        "mongo_collection": client.get_secret("reportscollectionname").value,
        "azure_endpoint": client.get_secret("unstructuredazureendpoint").value,
        "azure_key": client.get_secret("unstructuredazurekey").value,
        "embedding_deployment": "text-embedding-ada-002",
        "chat_deployment": "gpt-4.1-mini",
    }

    return secrets


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    secrets = load_secrets()

    rag = EnterpriseRAG(
        mongo_uri=secrets["mongo_uri"],
        db_name=secrets["mongo_db"],
        document_collection=secrets["mongo_collection"],
        azure_endpoint=secrets["azure_endpoint"],
        azure_key=secrets["azure_key"],
        embedding_deployment=secrets["embedding_deployment"],
        chat_deployment=secrets["chat_deployment"],
    )

    # Build vector index
    rag.build_index()

    # New session
    session_id = str(uuid.uuid4())

    print("\nEnterprise RAG System Ready\n")

    while True:
        question = input("Ask a question (or type exit): ")

        if question.lower() == "exit":
            break

        answer = rag.ask(session_id, question)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")
        