import json
from pathlib import Path
from typing import Dict, List, Any

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma

from azure.keyvault.secrets import SecretClient
from azure.identity import DeviceCodeCredential
from azure.storage.blob import BlobServiceClient

from src.audit_logger import log_event


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
WORK_DIR = BASE_DIR / "work"
CHROMA_DIR = WORK_DIR / "chroma_db"
POLICY_FILE = CONFIG_DIR / "policies.json"

COLLECTION_NAME = "supply_chain_access_control"
TOP_K = 3

# Azure Blob container that stores the supply chain docs
BLOB_CONTAINER_NAME = "supply-chain-docs"


# ============================================================
# LOAD SECRETS FROM AZURE KEY VAULT
# ============================================================

def load_secrets() -> Dict[str, str]:
    """
    Load Azure OpenAI and Azure Storage connection details from Key Vault.

    Expected secrets:
    - azure-openai-endpoint
    - azure-openai-key
    - azure-openai-api-version
    - azure-openai-chat-deployment
    - azure-openai-embedding-deployment
    - azure-storage-connection-string
    """

    keyVaultName = "data-management-kv-pk"
    KVUri = f"https://{keyVaultName}.vault.azure.net/"

    print("Connecting to Azure for authentication.")

    credential = DeviceCodeCredential(additionally_allowed_tenants=["*"])
    client = SecretClient(vault_url=KVUri, credential=credential)

    azure_endpoint = client.get_secret("azure-openai-endpoint").value
    azure_api_key = client.get_secret("azure-openai-key").value
    azure_api_version = client.get_secret("azure-openai-api-version").value
    azure_storage_connection_string = client.get_secret(
        "azure-storage-connection-string"
    ).value

    try:
        chat_deployment = client.get_secret("azure-openai-chat-deployment").value
    except Exception:
        chat_deployment = "gpt-4o-mini"

    try:
        embedding_deployment = client.get_secret("azure-openai-embedding-deployment").value
    except Exception:
        embedding_deployment = "text-embedding-3-small"

    return {
        "azure_endpoint": azure_endpoint,
        "azure_api_key": azure_api_key,
        "azure_api_version": azure_api_version,
        "azure_storage_connection_string": azure_storage_connection_string,
        "chat_deployment": chat_deployment,
        "embedding_deployment": embedding_deployment,
    }


SECRETS = load_secrets()

AZURE_ENDPOINT = SECRETS["azure_endpoint"]
AZURE_API_KEY = SECRETS["azure_api_key"]
AZURE_API_VERSION = SECRETS["azure_api_version"]
AZURE_STORAGE_CONNECTION_STRING = SECRETS["azure_storage_connection_string"]
CHAT_DEPLOYMENT = SECRETS["chat_deployment"]
EMBEDDING_DEPLOYMENT = SECRETS["embedding_deployment"]


# ============================================================
# CLIENT FACTORIES
# ============================================================

def get_chat_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_deployment=CHAT_DEPLOYMENT,
        temperature=0,
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_deployment=EMBEDDING_DEPLOYMENT,
    )


# ============================================================
# SETUP
# ============================================================

def ensure_dirs() -> None:
    WORK_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)


def load_policies() -> Dict[str, Any]:
    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# AZURE BLOB DOCUMENT LOADING
# ============================================================

def load_documents_from_blob(
    connection_string: str,
    container_name: str,
) -> List[Document]:
    """
    Load .txt documents from Azure Blob Storage.

    Expected blob structure:
    procurement/file1.txt
    manufacturing/file2.txt
    warehousing/file3.txt
    distribution/file4.txt

    The first path segment is treated as the supply chain stage.
    """
    print("Loading documents from Azure Blob Storage...")

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)

    documents: List[Document] = []

    for blob in container_client.list_blobs():
        print(f"Found blob: {blob.name}")
        if not blob.name.endswith(".txt"):
            continue

        # determine stage from filename instead of directory
        name = blob.name.lower()

        if "supplier" in name or "procurement" in name:
            stage = "procurement"
        elif "plant" in name or "production" in name or "manufacturing" in name:
            stage = "manufacturing"
        elif "inventory" in name or "warehouse" in name:
            stage = "warehousing"
        elif "carrier" in name or "delivery" in name or "distribution" in name:
            stage = "distribution"
        else:
            stage = "unknown"

        blob_client = container_client.get_blob_client(blob.name)
        text = blob_client.download_blob().readall().decode("utf-8", errors="ignore")

        doc = Document(
            page_content=text,
            metadata={
                "source": blob.name,
                "stage": stage,
            },
        )

        documents.append(doc)

    print(f"Loaded {len(documents)} documents from Azure Blob Storage.")
    return documents


# ============================================================
# ACCESS CONTROL HELPERS
# ============================================================

def get_role_policy(role: str, policies: Dict[str, Any]) -> Dict[str, Any]:
    if role not in policies:
        raise ValueError(
            f"Unknown role '{role}'. Available roles: {', '.join(policies.keys())}"
        )
    return policies[role]


def filter_documents_by_stage(
    documents: List[Document],
    allowed_stages: List[str],
) -> List[Document]:
    """
    Critical access control point:
    only authorized documents are allowed into the vector store for this user session.
    """
    return [
        doc for doc in documents
        if doc.metadata.get("stage", "").lower() in allowed_stages
    ]


# ============================================================
# VECTOR STORE
# ============================================================

def get_role_collection_name(role: str) -> str:
    """
    Separate collection per role for teaching clarity.
    In a production system, you may use metadata filtering instead.
    """
    return f"{COLLECTION_NAME}_{role}"


def reset_role_vectorstore(role: str) -> None:
    """
    Delete the role collection so it can be rebuilt for the current session.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=get_role_collection_name(role),
        embedding_function=embeddings,
    )
    try:
        vectorstore.delete_collection()
    except Exception:
        pass


def build_role_vectorstore(
    role: str,
    documents: List[Document],
    allow_write_embeddings: bool,
) -> Chroma:
    """
    Only roles with embedding write permission may build/update the vector store.
    For demo purposes, all roles can query, but only some roles can rebuild.
    """
    if not allow_write_embeddings:
        raise PermissionError(
            f"Role '{role}' is not allowed to write embeddings to the vector store."
        )

    reset_role_vectorstore(role)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=get_role_collection_name(role),
    )

    print(
        f"Vector store built for role '{role}' "
        f"with {len(documents)} authorized documents."
    )
    return vectorstore


def load_role_vectorstore(role: str) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=get_role_collection_name(role),
        embedding_function=embeddings,
    )


def ensure_role_vectorstore(
    role: str,
    authorized_docs: List[Document],
    allow_write_embeddings: bool,
) -> Chroma:
    """
    If the role can write embeddings, build or rebuild its collection.
    If not, try to load an existing collection. If missing, fall back to in-memory
    retrieval using the LLM context only.
    """
    role_collection = get_role_collection_name(role)

    try:
        vectorstore = load_role_vectorstore(role)
        # Trigger a small operation to verify the collection exists
        _ = vectorstore._collection.count()
        if vectorstore._collection.count() > 0:
            return vectorstore
    except Exception:
        pass

    if allow_write_embeddings:
        return build_role_vectorstore(
            role=role,
            documents=authorized_docs,
            allow_write_embeddings=allow_write_embeddings,
        )

    return None


# ============================================================
# RAG HELPERS
# ============================================================

def format_retrieved_docs(docs: List[Document]) -> str:
    if not docs:
        return "No authorized documents were retrieved."

    blocks = []
    for doc in docs:
        blocks.append(
            f"""
SOURCE: {doc.metadata.get("source", "unknown")}
STAGE: {doc.metadata.get("stage", "unknown")}

CONTENT:
{doc.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def retrieve_authorized_docs(
    question: str,
    role: str,
    authorized_docs: List[Document],
    allow_write_embeddings: bool,
    top_k: int = TOP_K,
) -> List[Document]:
    """
    If a role-specific vector store exists, use semantic retrieval.
    Otherwise, use a simple keyword fallback over already-authorized docs.
    """
    vectorstore = ensure_role_vectorstore(
        role=role,
        authorized_docs=authorized_docs,
        allow_write_embeddings=allow_write_embeddings,
    )

    if vectorstore is not None:
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(question)

    # Fallback for read-only roles with no writable embedding permission
    question_terms = set(question.lower().split())
    scored_docs = []

    for doc in authorized_docs:
        doc_terms = set(doc.page_content.lower().split())
        overlap = len(question_terms.intersection(doc_terms))
        if overlap > 0:
            scored_docs.append((overlap, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


def ask_supply_chain_chatbot(
    role: str,
    question: str,
    allowed_stages: List[str],
    allow_write_embeddings: bool,
    all_documents: List[Document],
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    authorized_docs = filter_documents_by_stage(all_documents, allowed_stages)

    retrieved_docs = retrieve_authorized_docs(
        question=question,
        role=role,
        authorized_docs=authorized_docs,
        allow_write_embeddings=allow_write_embeddings,
        top_k=top_k,
    )

    if not retrieved_docs:
        log_event(
            user=role,
            action="query",
            status="denied",
            resource=question,
        )
        return {
            "answer": (
                "No authorized information is available for that question. "
                "The requested information may be outside your allowed supply chain stages."
            ),
            "docs": [],
        }

    context = format_retrieved_docs(retrieved_docs)
    llm = get_chat_llm()

    prompt = f"""
You are a supply chain assistant.

Answer only from the retrieved authorized context.
Do not use outside knowledge.
If the answer is not fully supported by the context, say so clearly.

User role:
{role}

Allowed stages:
{", ".join(allowed_stages)}

User question:
{question}

Retrieved authorized context:
{context}

Instructions:
- keep the answer concise
- mention the stage when useful
- do not mention any stage outside the allowed stages
"""

    response = llm.invoke(prompt)

    log_event(
        user=role,
        action="query",
        status="approved",
        resource=question,
    )

    return {
        "answer": response.content,
        "docs": retrieved_docs,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dirs()

    policies = load_policies()
    all_documents = load_documents_from_blob(
        connection_string=AZURE_STORAGE_CONNECTION_STRING,
        container_name=BLOB_CONTAINER_NAME,
    )

    print("\nSupply chain access-control chatbot ready.")
    print("Available roles:", ", ".join(sorted(policies.keys())))
    print("Type 'exit' to quit.\n")

    role = input("Enter role: ").strip()
    if role.lower() == "exit":
        return

    try:
        policy = get_role_policy(role, policies)
    except ValueError as exc:
        print(exc)
        return

    allowed_stages = policy["read_stages"]
    allow_write_embeddings = policy["write_embeddings"]

    print(f"\nLogged in as: {role}")
    print(f"Allowed stages: {allowed_stages}")
    print(f"Can write embeddings: {allow_write_embeddings}\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = ask_supply_chain_chatbot(
            role=role,
            question=question,
            allowed_stages=allowed_stages,
            allow_write_embeddings=allow_write_embeddings,
            all_documents=all_documents,
            top_k=TOP_K,
        )

        print("\nANSWER")
        print(result["answer"])

        print("\nRETRIEVED DOCUMENTS")
        if result["docs"]:
            for doc in result["docs"]:
                print(
                    f"- {doc.metadata.get('source', 'unknown')} "
                    f"[stage={doc.metadata.get('stage', 'unknown')}]"
                )
        else:
            print("- None")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()