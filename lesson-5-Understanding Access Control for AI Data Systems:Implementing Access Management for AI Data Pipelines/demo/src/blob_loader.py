from azure.storage.blob import BlobServiceClient


def infer_stage_from_filename(filename: str) -> str:
    """
    Infer supply chain stage from file name.
    """

    name = filename.lower()

    if "supplier" in name or "procurement" in name:
        return "procurement"

    if "plant" in name or "production" in name or "manufacturing" in name:
        return "manufacturing"

    if "inventory" in name or "warehouse" in name:
        return "warehousing"

    if "carrier" in name or "delivery" in name or "distribution" in name:
        return "distribution"

    return "unknown"


def load_documents(connection_string, container):

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container)

    documents = []

    for blob in container_client.list_blobs():

        if not blob.name.endswith(".txt"):
            continue

        # determine stage from filename
        stage = infer_stage_from_filename(blob.name)

        blob_client = container_client.get_blob_client(blob.name)
        text = blob_client.download_blob().readall().decode()

        documents.append({
            "stage": stage,
            "text": text,
            "source": blob.name
        })

    return documents