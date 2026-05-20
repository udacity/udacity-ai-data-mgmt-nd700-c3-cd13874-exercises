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
from azure.identity import DeviceCodeCredential

# ============================================================
# LOAD SECRETS FROM AZURE KEY VAULT
# ============================================================

def load_secrets() -> Dict[str, str]:
    """
    Load Azure AI Foundry / Azure OpenAI connection details from Key Vault.
    """
    keyVaultName = "data-management-kv-pk"
    KVUri = f"https://{keyVaultName}.vault.azure.net/"

    print("Connecting to Azure for authentication.")

    credential = DeviceCodeCredential(additionally_allowed_tenants=["*"])
    client = SecretClient(vault_url=KVUri, credential=credential)

    azure_endpoint = client.get_secret("azure-openai-endpoint").value
    azure_api_key = client.get_secret("azure-openai-key").value
    azure_api_version = client.get_secret("azure-openai-api-version").value

    try:
        chat_deployment = client.get_secret("azure-openai-chat-deployment").value
    except Exception:
        chat_deployment = "gpt-4o-mini"

    try:
        vision_deployment = client.get_secret("azure-openai-vision-deployment").value
    except Exception:
        vision_deployment = "gpt-4o-mini"

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

def extract_all_images_from_page(
    pdf_doc: fitz.Document,
    page: fitz.Page,
) -> List[Image.Image]:
    """
    Extract all images from a page.

    Returns a list of PIL images in the order they appear in page.get_images().
    Duplicate xrefs are skipped.
    """
    image_list = page.get_images(full=True)
    if not image_list:
        return []

    extracted_images: List[Image.Image] = []
    seen_xrefs = set()

    for img_idx, img_info in enumerate(image_list, start=1):
        xref = img_info[0]

        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            extracted = pdf_doc.extract_image(xref)
            image_bytes = extracted["image"]
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            extracted_images.append(pil_img)
        except Exception as exc:
            print(f"Warning: could not extract image {img_idx} on page: {exc}")

    return extracted_images


# ============================================================
# VISION DESCRIPTION
# ============================================================

def describe_image_with_vision(
    image: Image.Image,
    page_number: int,
    image_number: int,
    vision_client: AzureChatOpenAI,
) -> str:
    image_b64 = image_to_base64(image)

    message = [
        {
            "type": "text",
            "text": (
                f"You are analyzing image {image_number} from page {page_number} "
                "of an apparel catalog.\n"
                "Describe the apparel in a structured, concise way.\n\n"
                "Include:\n"
                "1. main clothing or footwear item\n"
                "2. colors\n"
                "3. visible materials or finish if likely\n"
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

        page_images = extract_all_images_from_page(pdf_doc, page)

        if not page_images:
            combined_content = f"""
Page number: {page_number}

Page text:
{page_text}

Image description:
No image extracted from this page.
""".strip()

            documents.append(
                Document(
                    page_content=combined_content,
                    metadata={
                        "page_number": page_number,
                        "image_number": None,
                        "image_path": "",
                        "has_image": False,
                        "preprocessing_path": "",
                    },
                )
            )
            continue

        for image_idx, pil_img in enumerate(page_images, start=1):
            print(f"  Describing image {image_idx} on page {page_number}...")

            image_path = IMAGE_DIR / f"page_{page_number}_image_{image_idx}.png"
            pil_img.save(image_path)

            preprocessing_meta = preprocess_image(
                pil_img,
                target_size=TARGET_SIZE,
                normalize=True,
                return_tensor_layout=True,
            )

            preprocessing_path = (
                ARTIFACT_DIR / f"page_{page_number}_image_{image_idx}_preprocessing.json"
            )
            with open(preprocessing_path, "w", encoding="utf-8") as f:
                json.dump(preprocessing_meta, f, indent=2)

            image_description = describe_image_with_vision(
                pil_img,
                page_number,
                image_idx,
                vision_llm,
            )

            combined_content = f"""
Page number: {page_number}
Image number: {image_idx}

Page text:
{page_text}

Image description:
{image_description}
""".strip()

            metadata = {
                "page_number": page_number,
                "image_number": image_idx,
                "image_path": str(image_path),
                "has_image": True,
                "preprocessing_path": str(preprocessing_path),
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
IMAGE {doc.metadata.get("image_number", "N/A")}
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
- mention page numbers and image numbers when useful
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

        print("\nRETRIEVED RESULTS")
        for doc in result["docs"]:
            print(
                f"- Page {doc.metadata.get('page_number')} | "
                f"Image {doc.metadata.get('image_number')} | "
                f"Path: {doc.metadata.get('image_path', 'N/A')}"
            )

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()