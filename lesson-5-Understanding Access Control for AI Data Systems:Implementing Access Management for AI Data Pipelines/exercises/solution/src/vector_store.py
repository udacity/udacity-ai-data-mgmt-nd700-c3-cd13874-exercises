import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection("supply_chain_docs")

def write_embeddings(role, documents, allow_write):

    if not allow_write:
        raise PermissionError("Role not allowed to write embeddings")

    for i, doc in enumerate(documents):

        collection.add(
            documents=[doc["text"]],
            ids=[str(i)],
            metadatas=[{"stage": doc["stage"]}]
        )


def query_embeddings(query):

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    return results