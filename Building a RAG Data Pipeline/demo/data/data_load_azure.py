import os
from pymongo import MongoClient
import PyPDF2

def load_pdfs_to_mongo(folder_path="maintenance_reports"):
    # Connect to MongoDB
    client = MongoClient("mongodb+srv://unstructureddataadmin:Landmark1@unstructureddatacluster.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
    # Create or access the database and collection
    db = client["documents"]
    collection_name = "maintenance_reports"

    # Explicitly create collection if it doesn't exist
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

    collection = db[collection_name]

    # Ensure folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Loop through PDFs
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")

            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""

                for page in reader.pages:
                    text += page.extract_text() or ""

            # Insert into MongoDB
            doc = {
                "filename": filename,
                "content": text
            }

            collection.insert_one(doc)

    print("All PDFs loaded into MongoDB (documents.maintenance_reports).")


if __name__ == "__main__":
    load_pdfs_to_mongo()

