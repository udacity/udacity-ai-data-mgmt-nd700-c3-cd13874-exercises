# ============================================================
# enterprise_rag.py
# Enterprise RAG System
# Cosmos DB (Mongo API) + Chroma + Azure OpenAI
# ============================================================

import datetime
import json
import os
import os
import uuid
from bson import ObjectId
from pymongo import MongoClient

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI


class EnterpriseRAG:
    """
    Production-style RAG system
    - Cosmos DB (Mongo API) → system of record
    - ChromaDB → vector index
    - Azure OpenAI → embeddings + chat
    - Prompt store
    - Memory store
    - Interaction logging
    """

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        document_collection: str,
        azure_endpoint: str,
        azure_key: str,
        embedding_deployment: str,
        chat_deployment: str,
        azure_api_version: str = "2024-02-15-preview",
        chroma_path: str = "./chroma_store",
    ):

        # ---------------------------
        # Cosmos DB (Mongo API)
        # ---------------------------
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]

        self.rag_config = self.db["rag_config"]
        self.documents = self.db[document_collection]
        self.prompts = self.db["prompts"]
        self.memory = self.db["memory"]
        self.interactions = self.db["interactions"]

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
        # Chroma Vector Store
        # ---------------------------
        self.vector_db = Chroma(
            collection_name="rag_chunks",
            embedding_function=self.embeddings,
            persist_directory=chroma_path,
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

    # ============================================================
    # RAG CONFIGURATION
    # ============================================================

    def create_rag_config_if_missing(self):

        if self.rag_config.count_documents({"config_id": "default"}) == 0:

            self.rag_config.insert_one({
                "config_id": "default",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            })

    def get_rag_config(self, config_id: str = "default"):

        config = self.rag_config.find_one({"config_id": config_id})
        
        if not config:
            print("RAG config not found. Creating default config.")

            default_config = {
                "config_id": config_id,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }

            self.rag_config.insert_one(default_config)

            return default_config

        return config

    # ============================================================
    # BUILD VECTOR INDEX
    # ============================================================

    def build_index(self):

        rag_config = self.get_rag_config()
        CHUNK_SIZE = rag_config["chunk_size"]
        CHUNK_OVERLAP = rag_config["chunk_overlap"]

        def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap
            return chunks

        print("Loading documents from MongoDB")

        docs = list(self.documents.find({}, {"_id": 1, "content": 1}))

        texts = []
        ids = []
        metadatas = []

        print("Chunking and Indexing documents")

        for d in docs:
            content = d.get("content", "").strip()
            if not content:
                continue

            mongo_id = str(d["_id"])
            print(f"Processing document {mongo_id} with length {len(content)} characters.")
            chunks = chunk_text(content)
            print(f" - {len(chunks)} chunks created.")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{mongo_id}_chunk_{i}"
                texts.append(chunk)
                ids.append(chunk_id)
                metadatas.append({
                    "mongo_id": mongo_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "indexed_at": str(datetime.datetime.utcnow())
                })

        if not texts:
            print("No documents found.")
            return

        print(f"Embedding {len(texts)} chunks from documents")

        self.vector_db.add_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
        )

        print("Vector DB size:", self.vector_db._collection.count())

    # ============================================================
    # PROMPT MANAGEMENT
    # ============================================================

    def create_prompt_if_missing(self):

        if self.prompts.count_documents({"prompt_id": "rag_v1"}) == 0:

            self.prompts.insert_one({
                "prompt_id": "rag_v1",
                "system_prompt": "You are a helpful assistant. Use only retrieved context.",
                "template": """
Context:
{context}

Memory:
{memory}

Question:
{question}

Answer clearly and concisely:
""",
                "model_version": "v1",
                "created_at": datetime.datetime.utcnow()
            })

    def get_prompt(self):
        return self.prompts.find_one({"prompt_id": "rag_v1"})

    # ============================================================
    # MEMORY MANAGEMENT
    # ============================================================

    def get_memory(self, session_id):

        record = self.memory.find_one({"session_id": session_id})
        return record["summary"] if record else ""

    def update_memory(self, session_id, conversation_text):

        summary_prompt = f"""
Summarize this conversation briefly:

{conversation_text}
"""

        response = self.llm.invoke(summary_prompt)

        self.memory.update_one(
            {"session_id": session_id},
            {"$set": {
                "summary": response.content,
                "updated_at": datetime.datetime.utcnow()
            }},
            upsert=True
        )

    # ============================================================
    # RAG QUERY
    # ============================================================

    def ask(self, session_id: str, question: str, k: int = 3):

        results = self.vector_db.similarity_search(question, k=k)

        context_chunks = []
        source_ids = []

        for doc in results:
            mongo_id = doc.metadata.get("mongo_id")

            try:
                mongo_doc = self.documents.find_one(
                    {"_id": ObjectId(mongo_id)}
                )
            except Exception:
                mongo_doc = None

            if mongo_doc and mongo_doc.get("content"):
                context_chunks.append(mongo_doc["content"])
                source_ids.append(str(mongo_doc["_id"]))

        if not context_chunks:
            return "No relevant documents found."

        context_text = "\n\n".join(context_chunks)
        memory_text = self.get_memory(session_id)

        prompt_record = self.get_prompt()

        final_prompt = prompt_record["template"].format(
            context=context_text,
            memory=memory_text,
            question=question
        )

        response = self.llm.invoke([
            {"role": "system", "content": prompt_record["system_prompt"]},
            {"role": "user", "content": final_prompt}
        ])

        answer = response.content

        # Log interaction
        self.interactions.insert_one({
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "retrieved_doc_ids": source_ids,
            "timestamp": datetime.datetime.utcnow()
        })

        # Update memory
        self.update_memory(session_id, f"Q:{question}\nA:{answer}")

        return answer