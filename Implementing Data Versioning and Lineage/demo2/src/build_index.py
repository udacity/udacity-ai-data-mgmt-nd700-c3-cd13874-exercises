import json
import shutil
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma

from common import load_params, get_embeddings

DATA_PATH = Path("data/inventory.csv")
PROMPT_PATH = Path("artifacts/system_prompt.txt")
CHUNKS_PATH = Path("artifacts/chunks.jsonl")
METRICS_PATH = Path("artifacts/metrics.json")
CHROMA_DIR = Path("artifacts/chroma_db")


def build_documents(df: pd.DataFrame, max_products_per_chunk: int) -> List[Document]:
    docs = []

    rows = df.to_dict(orient="records")
    for i in range(0, len(rows), max_products_per_chunk):
        batch = rows[i:i + max_products_per_chunk]

        lines = []
        product_ids = []

        for row in batch:
            product_ids.append(str(row["product_id"]))
            lines.append(
                f"product_id: {row['product_id']}\n"
                f"product_name: {row['product_name']}\n"
                f"category: {row['category']}\n"
                f"brand: {row['brand']}\n"
                f"store_qty: {row['store_qty']}\n"
                f"warehouse_qty: {row['warehouse_qty']}\n"
                f"reorder_point: {row['reorder_point']}\n"
                f"unit_price: {row['unit_price']}\n"
                f"supplier: {row['supplier']}\n"
                f"lead_time_days: {row['lead_time_days']}\n"
                f"aisle: {row['aisle']}\n"
                f"last_updated: {row['last_updated']}"
            )

        content = "\n\n---\n\n".join(lines)

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "inventory.csv",
                    "product_ids": ",".join(product_ids),
                    "chunk_index": len(docs),
                },
            )
        )

    return docs


def main() -> None:
    params = load_params()
    embeddings = get_embeddings()

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")

    max_products_per_chunk = params["chunking"]["max_products_per_chunk"]
    collection_name = params["rag"]["collection_name"]

    docs = build_documents(df, max_products_per_chunk=max_products_per_chunk)

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(
                json.dumps(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "num_products": int(len(df)),
        "num_chunks": int(len(docs)),
        "prompt_chars": int(len(prompt_text)),
        "collection_name": collection_name,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Touch the retriever once so the code path is validated
    retriever = vectorstore.as_retriever(search_kwargs={"k": params["rag"]["k"]})
    sample_docs = retriever.invoke("Which items are low stock?")
    print(f"Index built. Retrieved {len(sample_docs)} sample docs during validation.")
    print(f"Wrote {CHUNKS_PATH}")
    print(f"Wrote {METRICS_PATH}")
    print(f"Persisted Chroma DB to {CHROMA_DIR}")


if __name__ == "__main__":
    main()