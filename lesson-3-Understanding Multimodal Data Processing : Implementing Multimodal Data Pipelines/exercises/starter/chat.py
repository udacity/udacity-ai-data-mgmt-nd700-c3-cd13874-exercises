import io
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma

from azure.keyvault.secrets import SecretClient
from azure.identity import InteractiveBrowserCredential

# ============================================================
# LOAD SECRETS FROM AZURE KEY VAULT
# ============================================================

def load_secrets() -> Dict[str, str]:
    """
    Load Azure AI Foundry / Azure OpenAI connection details from Key Vault.

    Expected secrets in Key Vault:
    - azure-openai-endpoint
    - azure-openai-key
    - azure-openai-api-version

    Optional secrets if you want deployments in KV too:
    - azure-openai-chat-deployment
    - azure-openai-vision-deployment
    - azure-openai-embedding-deployment
    """
    print("Authenticating with Azure Key Vault...")

    credential = InteractiveBrowserCredential(additionally_allowed_tenants=["*"])
    kv_uri = "https://useraisecrets.vault.azure.net/"
    client = SecretClient(vault_url=kv_uri, credential=credential)

    # Required connection secrets
    azure_endpoint = client.get_secret("azure-openai-endpoint").value
    azure_api_key = client.get_secret("azure-openai-key").value
    azure_api_version = client.get_secret("azure-openai-api-version").value

    # Deployment names can also live in Key Vault.
    # If you don't store them there, replace these with string literals.
    try:
        chat_deployment = client.get_secret("azure-openai-chat-deployment").value
    except Exception:
        chat_deployment = "gpt-5"

    try:
        vision_deployment = client.get_secret("azure-openai-vision-deployment").value
    except Exception:
        vision_deployment = "gpt-5"

    try:
        embedding_deployment = client.get_secret("azure-openai-embedding-deployment").value
    except Exception:
        embedding_deployment = "text-embedding-3-small"

    return {
        "azure_endpoint": azure_endpoint,
        "azure_api_key": azure_api_key,
        "azure_api_version": azure_api_version,
        "embedding_deployment": embedding_deployment,
        "chat_deployment": chat_deployment,
        "vision_deployment": vision_deployment,
    }


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "data/shoes.pdf"
WORK_DIR = Path("catalog_langchain_rag")
IMAGE_DIR = WORK_DIR / "images"
ARTIFACT_DIR = WORK_DIR / "artifacts"
CHROMA_DIR = WORK_DIR / "chroma_db"

COLLECTION_NAME = "apparel_catalog"

secrets = load_secrets()

AZURE_ENDPOINT = secrets["azure_endpoint"]
AZURE_API_KEY = secrets["azure_api_key"]
AZURE_API_VERSION = secrets["azure_api_version"]

EMBEDDING_DEPLOYMENT = secrets["embedding_deployment"]
CHAT_DEPLOYMENT = secrets["chat_deployment"]
VISION_DEPLOYMENT = secrets["vision_deployment"]

TARGET_SIZE = (224, 224)
TOP_K = 3


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


def get_vision_llm() -> AzureChatOpenAI:
    """
    Vision-capable deployment.
    This still uses AzureChatOpenAI, but points to a deployment that supports image input.
    """
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_deployment=VISION_DEPLOYMENT,
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
    IMAGE_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image,
    target_size: tuple[int, int] = TARGET_SIZE,
    normalize: bool = True,
    return_tensor_layout: bool = True,
) -> Dict[str, Any]:
    """
    Explicit image preprocessing.

    We document:
    - dimensions
    - normalization
    - format

    We intentionally do NOT tokenize the image here because that would
    couple storage to a specific downstream model.
    """
    image = image.convert("RGB")
    resized = image.resize(target_size)

    arr = np.asarray(resized).astype("float32")  # HWC

    if normalize:
        arr = arr / 255.0

    result = {
        "array_shape_hwc": list(arr.shape),
        "dtype": str(arr.dtype),
        "normalized_to_0_1": normalize,
        "target_size": list(target_size),
        "storage_decision": (
            "Store raw image and preprocessing metadata; defer model-specific "
            "tokenization until retrieval or model-execution time."
        ),
    }

    if return_tensor_layout:
        tensor = np.transpose(arr, (2, 0, 1))  # CHW
        result["tensor_shape_chw"] = list(tensor.shape)

    return result


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_largest_image_from_page(
    pdf_doc: fitz.Document,
    page: fitz.Page,
) -> Optional[Image.Image]:
    image_list = page.get_images(full=True)
    if not image_list:
        return None

    largest_img = None
    largest_area = -1

    for img_info in image_list:
        xref = img_info[0]
        try:
            extracted = pdf_doc.extract_image(xref)
            image_bytes = extracted["image"]
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            area = pil_img.width * pil_img.height

            if area > largest_area:
                largest_area = area
                largest_img = pil_img
        except Exception as exc:
            print(f"Warning: could not extract image: {exc}")

    return largest_img


# ============================================================
# VISION DESCRIPTION
# ============================================================

def describe_image_with_vision(
    image: Image.Image,
    page_number: int,
    vision_client: AzureChatOpenAI,
) -> str:
    """
    Turn image into reusable text for retrieval.
    """
    image_b64 = image_to_base64(image)

    message = [
        {
            "type": "text",
            "text": (
                f"You are analyzing page {page_number} from an apparel catalog.\n"
                "Describe the apparel in a structured, concise way.\n\n"
                "Include:\n"
                "1. main clothing items\n"
                "2. colors\n"
                "3. accessories\n"
                "4. style category\n"
                "5. likely occasion/use case\n"
                "6. team/brand references if visible\n"
                "7. a short natural-language summary\n\n"
                "Keep it concise but specific."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            },
        },
    ]

    response = vision_client.invoke([("human", message)])

    if isinstance(response.content, str):
        return response.content

    return str(response.content)


# ============================================================
# DOCUMENT BUILDING
# ============================================================

def build_documents_from_pdf(pdf_path: str) -> List[Document]:
    ensure_dirs()

    vision_llm = get_vision_llm()
    pdf_doc = fitz.open(pdf_path)

    documents: List[Document] = []

    for page_idx in range(len(pdf_doc)):
        page = pdf_doc[page_idx]
        page_number = page_idx + 1

        print(f"Processing page {page_number}...")

        page_text = page.get_text("text").strip()
        page_text = page_text if page_text else "No extractable text on this page."

        pil_img = extract_largest_image_from_page(pdf_doc, page)

        image_path = None
        image_description = "No image extracted from this page."
        preprocessing_meta = None

        if pil_img is not None:
            image_path = IMAGE_DIR / f"page_{page_number}.png"
            pil_img.save(image_path)

            preprocessing_meta = preprocess_image(
                pil_img,
                target_size=TARGET_SIZE,
                normalize=True,
                return_tensor_layout=True,
            )

            image_description = describe_image_with_vision(
                pil_img,
                page_number,
                vision_llm,
            )

            with open(
                ARTIFACT_DIR / f"page_{page_number}_preprocessing.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(preprocessing_meta, f, indent=2)

        combined_content = f"""
Page number: {page_number}

Page text:
{page_text}

Image description:
{image_description}
""".strip()

        metadata = {
            "page_number": page_number,
            "image_path": str(image_path) if image_path else "",
            "has_image": pil_img is not None,
            "preprocessing_path": (
                str(ARTIFACT_DIR / f"page_{page_number}_preprocessing.json")
                if preprocessing_meta
                else ""
            ),
        }

        documents.append(
            Document(
                page_content=combined_content,
                metadata=metadata,
            )
        )

    pdf_doc.close()
    return documents


# ============================================================
# CHROMA INDEX
# ============================================================

def build_vectorstore(pdf_path: str) -> Chroma:
    documents = build_documents_from_pdf(pdf_path)
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print(f"\nChroma collection '{COLLECTION_NAME}' built successfully.")
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


# ============================================================
# RAG
# ============================================================

def format_retrieved_docs(docs: List[Document]) -> str:
    blocks = []

    for doc in docs:
        blocks.append(
            f"""
PAGE {doc.metadata.get("page_number", "unknown")}
Image path: {doc.metadata.get("image_path", "N/A")}

{doc.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def ask_catalog(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    retrieved_docs = retriever.invoke(question)
    context = format_retrieved_docs(retrieved_docs)

    llm = get_chat_llm()

    prompt = f"""
You are a helpful assistant for an apparel catalog.

Answer only from the retrieved context.
If the answer is not supported by the catalog, say that you are not sure based on the catalog.

User question:
{question}

Retrieved context:
{context}

Instructions:
- mention page numbers when useful
- keep the answer practical
- if comparing looks, explain briefly why
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "docs": retrieved_docs,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dirs()

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print("No Chroma index found. Building it now...")
        build_vectorstore(PDF_PATH)

    print("\nCatalog chatbot ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask about the catalog: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        result = ask_catalog(question, top_k=TOP_K)

        print("\nANSWER")
        print(result["answer"])

        print("\nRETRIEVED PAGES")
        for doc in result["docs"]:
            print(
                f"- Page {doc.metadata.get('page_number')} | "
                f"Image: {doc.metadata.get('image_path', 'N/A')}"
            )

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()