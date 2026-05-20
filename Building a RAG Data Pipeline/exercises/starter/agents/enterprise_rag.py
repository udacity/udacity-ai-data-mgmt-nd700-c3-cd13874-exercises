# ============================================================
# enterprise_rag.py
# Enterprise RAG System
# Cosmos DB (Mongo API) + Chroma + Azure OpenAI
#
# Privacy split:
#   MongoDB  → <TODO: specify what goes here, e.g. prompts, session memory, interaction logs> (PII and sensitive data),RAG config, document metadata> (non-PII operational data)
#   Local FS → <TODO: specify what goes here, e.g. prompts, session memory, interaction logs> (PII and sensitive data), RAG config, document metadata> (non-PII operational data)
# ============================================================

import datetime
import json
import os
import uuid
from bson import ObjectId
from pymongo import MongoClient

from langchain_chroma import Chroma
from langchain_openai import <TODO 1: import the Azure OpenAI classes needed for embeddings and chat>


class EnterpriseRAG:
    """
    Production-style RAG system with privacy-conscious data handling.

    Data storage split:
    <TODO 2: Complete the table below describing where each type of data is stored and why>
    ┌─────────────────────┬──────────────┬───────────────────────────────────┐
    │ Data                │ Store        │ Reason                            │
    ├─────────────────────┼──────────────┼───────────────────────────────────┤
    │ RAG config          │ <TODO>       │ Operational config, no PII        │
    │ Knowledge-base docs │ <TODO>       │ Your content, not user data       │
    │ Prompts             │ <TODO>       │ Business logic, sensitive         │
    │ Session memory      │ <TODO>       │ Derived from user conversations   │
    │ Interaction logs    │ <TODO>       │ Raw user questions & answers      │
    └─────────────────────┴──────────────┴───────────────────────────────────┘
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
        local_store_path: str = "./local_store",
    ):
        # ---------------------------
        # Local storage paths
        # (stay on-device)
        # ---------------------------
        self.local_store_path = <TODO 3: Set the local_store_path variable to the provided argument>
        <TODO 4: Set the paths for what needs to be stored locally>

        # Ensure local directories exist
        for directory in [self.memory_dir, self.interactions_dir]:
            os.makedirs(directory, exist_ok=True)

        # ---------------------------
        # Cosmos DB (Mongo API)
        # Only non-PII collections
        # ---------------------------
        self.mongo_client = MongoClient(mongo_uri)
        <TODO 5: Set up the MongoDB collections>

        # ---------------------------
        # Azure Embeddings
        # ---------------------------
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=<TODO 6: Use the embedding_deployment variable here>,
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
            azure_deployment=<TODO 7: Use the chat_deployment variable here>,
            openai_api_version=azure_api_version,
            openai_api_key=azure_key,
            temperature=0.2,
        )

        # Seed default prompt locally if not present
        self._init_default_prompt()

    # ============================================================
    # RAG CONFIGURATION 
    # ============================================================

    def create_rag_config_if_missing(self):
        """Insert default RAG config into MongoDB if absent."""
        if self.rag_config.count_documents({"config_id": "default"}) == 0:
            <TODO 8: Insert a default RAG config document into the rag_config collection in MongoDB>
            })

    def get_rag_config(self, config_id: str = "default") -> dict:
        """Fetch RAG config from MongoDB, creating defaults if missing."""
        config = self.rag_config.find_one({"config_id": config_id})

        if not config:
            print("RAG config not found. Creating default config.")
            default_config = {
                <TODO 9: Define the default RAG config fields such as chunk_size and chunk_overlap, and insert it into MongoDB>
            }
            self.rag_config.insert_one(default_config)
            return default_config

        return config

    def update_rag_config(self, config_id: str = "default", **kwargs) -> None:
        """Update RAG config fields in MongoDB."""
        kwargs["updated_at"] = datetime.datetime.utcnow()
        self.rag_config.update_one(
            {"config_id": config_id},
            {"$set": kwargs},
            upsert=True,
        )

    # ============================================================
    # BUILD VECTOR INDEX
    # ============================================================

    def build_index(self) -> None:
        """Chunk all knowledge-base documents and embed into ChromaDB."""
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

        print("Loading documents from MongoDB …")
        docs = list(self.documents.find({}, {"_id": 1, "content": 1}))

        texts, ids, metadatas = [], [], []

        print("Chunking and indexing documents …")
        for d in docs:
            content = d.get("content", "").strip()
            if not content:
                continue

            mongo_id = str(d["_id"])
            chunks = chunk_text(content)
            print(f"  Document {mongo_id}: {len(content)} chars → {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                ids.append(f"{mongo_id}_chunk_{i}")
                metadatas.append({
                    "mongo_id": mongo_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "indexed_at": str(datetime.datetime.utcnow()),
                })

        if not texts:
            print("No documents found — index is empty.")
            return

        print(f"Embedding {len(texts)} chunks …")
        self.vector_db.add_texts(texts=texts, ids=ids, metadatas=metadatas)
        print("Vector DB size:", self.vector_db._collection.count())

    # ============================================================
    # PROMPT MANAGEMENT 
    # ============================================================

    def _init_default_prompt(self) -> None:
        <TODO 10: Check if the prompts exists, and if not, create it with a default prompt template, make sure you are reading/writing from where prompts are stored: locally or cloud>

        default_prompts = {
            "rag_v1": {
                "prompt_id": "rag_v1",
                "system_prompt": (
                    "You are a helpful assistant. "
                    "Use only the retrieved context to answer questions."
                ),
                "template": (
                    "\nContext:\n{context}"
                    "\n\nMemory:\n{memory}"
                    "\n\nQuestion:\n{question}"
                    "\n\nAnswer clearly and concisely:\n"
                ),
                "model_version": "v1",
                "created_at": str(datetime.datetime.utcnow()),
            }
        }

        <TODO 11: Write the default_prompts if it doesn't already exist, or if the specific prompt_id is missing>

        print(f"Default prompt written to <TODO 12: specify the location where prompts are stored>")

    def _load_prompts(self) -> dict:
        <TODO 13: Load all prompts>

    def get_prompt(self, prompt_id: str = "rag_v1") -> dict:
        <TODO 14: Fetch a specific prompt by ID from the loaded prompts, and handle the case where it might not exist>

    def save_prompt(self, prompt_id: str, system_prompt: str, template: str) -> None:
        """Create or update a prompt in the local JSON file."""
        prompts = self._load_prompts()
        prompts[prompt_id] = {
            "prompt_id": prompt_id,
            "system_prompt": system_prompt,
            "template": template,
            "model_version": "v1",
            "updated_at": str(datetime.datetime.utcnow()),
        }
        <TODO 15: Save the updated prompts>>

    # ============================================================
    # MEMORY MANAGEMENT
    # ============================================================

    def _memory_path(self, session_id: str) -> str:
        return <TODO 16: Return the memory corresponding to the given session_id>

    def get_memory(self, session_id: str) -> str:
        """Load the conversation summary for a session"""
        <TODO 17: Implement the logic to read the memory for the given session_id and return the summary text, handle the case where the memory might not exist>

    def update_memory(self, session_id: str, conversation_text: str) -> None:
        """
        Summarise the latest conversation turn and persist.
        """
        summary_prompt = (
            "Summarize the following conversation exchange briefly and neutrally:\n\n"
            f"{conversation_text}"
        )
        response = self.<TODO 18: Call the Azure Chat model with the summary_prompt to get a summary of the conversation turn>  

        record = {
            "session_id": session_id,
            "summary": response.content,
            "updated_at": str(datetime.datetime.utcnow()),
        }

        <TODO 19: Write the summary record to the memory store for the given session_id>

    def clear_memory(self, session_id: str) -> None:
        """Delete the local memory file for a session (right-to-erasure)."""
        <TODO 20: Implement the logic to delete the memory file corresponding to the given session_id>

    # ============================================================
    # INTERACTION LOGGING
    # ============================================================

    def _interactions_path(self, session_id: str) -> str:
        return <TODO 21: Return the interactions log path corresponding to the given session_id>

    def _log_interaction(
        self,
        session_id: str,
        question: str,
        answer: str,
        source_ids: list,
    ) -> None:
        """
        Append one interaction record to a session-scoped JSONL file locally.
        """
        record = {
            "interaction_id": str(uuid.uuid4()),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "retrieved_doc_ids": source_ids,
            "timestamp": str(datetime.datetime.utcnow()),
        }
        <TODO 22: Append the interaction record to the interactions log for the given session_id>

    def get_interactions(self, session_id: str) -> list:
        """Read all logged interactions for a session from local storage."""
        <TODO 23: Implement the logic to read all interaction records for the given session_id and return them as a list of dictionaries, handle the case where the interactions log might not exist>

    def clear_interactions(self, session_id: str) -> None:
        """Delete the local interaction log for a session (right-to-erasure)."""
        <TODO 24: Implement the logic to delete the interactions log file corresponding to the given session_id>

    # ============================================================
    # RIGHT-TO-ERASURE HELPER
    # ============================================================

    def purge_session(self, session_id: str) -> None:
        """
        Remove ALL locally stored data for a given session.
        Call this to honour a user deletion / right-to-erasure request.
        """
        self.<TODO 25: Call the methods to clear both memory and interactions for the given session_id>
        self.<TODO 26: Print a confirmation message that the session has been purged>
        print(f"Session {session_id} fully purged from local store.")

    # ============================================================
    # RAG QUERY
    # ============================================================

    def ask(self, session_id: str, question: str, k: int = 3) -> str:
        """
        Run a RAG query:
          1. Retrieve relevant chunks from ChromaDB
          2. Hydrate full documents from MongoDB
          3. Inject context + local memory into prompt
          4. Call Azure OpenAI
          5. Log interaction and update memory — both locally only
        """
        # --- 1. Vector search ---
        results = self.vector_db.similarity_search(question, k=k)

        context_chunks = []
        source_ids = []

        # --- 2. Hydrate from MongoDB ---
        for doc in results:
            mongo_id = doc.metadata.get("mongo_id")
            try:
                mongo_doc = self.documents.find_one({"_id": ObjectId(mongo_id)})
            except Exception:
                mongo_doc = None

            if mongo_doc and mongo_doc.get("content"):
                context_chunks.append(mongo_doc["content"])
                source_ids.append(str(mongo_doc["_id"]))

        if not context_chunks:
            return "No relevant documents found."

        # --- 3. Build prompt ---
        context_text = "\n\n".join(context_chunks)
        memory_text = self.get_memory(session_id)   # local read
        prompt_record = self.get_prompt()            # local read

        final_prompt = prompt_record["template"].format(
            context=context_text,
            memory=memory_text,
            question=question,
        )

        # --- 4. Call LLM ---
        response = self.llm.invoke([
            {"role": "system", "content": prompt_record["system_prompt"]},
            {"role": "user", "content": final_prompt},
        ])
        answer = response.content

        # --- 5. Persist ---
        self.<TODO 27: Append the interaction record to the interactions log for the given session_id>
        self.<TODO 28: Update the memory for the session_id with the latest conversation turn (question + answer)>

        return answer