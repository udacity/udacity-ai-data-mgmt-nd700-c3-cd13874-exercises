# ============================================================
# enterprise_rag.py
# Enterprise RAG System
# Cosmos DB (Mongo API) + Chroma + Azure OpenAI
#
# Privacy split:
#   MongoDB  → RAG config, knowledge-base documents (no PII)
#   Local FS → prompts, session memory, interaction logs
# ============================================================

import datetime
import json
import os
import uuid
from bson import ObjectId
from pymongo import MongoClient

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI


class EnterpriseRAG:
    """
    Production-style RAG system with privacy-conscious data handling.

    Data storage split:
    ┌─────────────────────┬──────────────┬───────────────────────────────────┐
    │ Data                │ Store        │ Reason                            │
    ├─────────────────────┼──────────────┼───────────────────────────────────┤
    │ RAG config          │ MongoDB      │ Operational config, no PII        │
    │ Knowledge-base docs │ MongoDB      │ Your content, not user data       │
    │ Prompts             │ Local JSON   │ Business logic, sensitive         │
    │ Session memory      │ Local JSON   │ Derived from user conversations   │
    │ Interaction logs    │ Local JSONL  │ Raw user questions & answers      │
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
        # (prompts, memory, interactions stay on-device)
        # ---------------------------
        self.local_store_path = local_store_path
        self.prompts_path = os.path.join(local_store_path, "prompts.json")
        self.memory_dir = os.path.join(local_store_path, "memory")
        self.interactions_dir = os.path.join(local_store_path, "interactions")

        # Ensure local directories exist
        for directory in [self.memory_dir, self.interactions_dir]:
            os.makedirs(directory, exist_ok=True)

        # ---------------------------
        # Cosmos DB (Mongo API)
        # Only non-PII collections: rag_config + knowledge-base documents
        # ---------------------------
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.rag_config = self.db["rag_config"]
        self.documents = self.db[document_collection]

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

        # Seed default prompt locally if not present
        self._init_default_prompt()

    # ============================================================
    # RAG CONFIGURATION  (MongoDB — no PII)
    # ============================================================

    def create_rag_config_if_missing(self):
        """Insert default RAG config into MongoDB if absent."""
        if self.rag_config.count_documents({"config_id": "default"}) == 0:
            self.rag_config.insert_one({
                "config_id": "default",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow(),
            })

    def get_rag_config(self, config_id: str = "default") -> dict:
        """Fetch RAG config from MongoDB, creating defaults if missing."""
        config = self.rag_config.find_one({"config_id": config_id})

        if not config:
            print("RAG config not found. Creating default config.")
            default_config = {
                "config_id": config_id,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow(),
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
    # PROMPT MANAGEMENT  (local JSON — privacy-sensitive)
    # ============================================================

    def _init_default_prompt(self) -> None:
        """Write the default prompt file locally if it does not exist."""
        if os.path.exists(self.prompts_path):
            return

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

        with open(self.prompts_path, "w") as f:
            json.dump(default_prompts, f, indent=2)

        print(f"Default prompt written to {self.prompts_path}")

    def _load_prompts(self) -> dict:
        """Read all prompts from the local JSON file."""
        with open(self.prompts_path, "r") as f:
            return json.load(f)

    def get_prompt(self, prompt_id: str = "rag_v1") -> dict:
        """Return a single prompt record by ID."""
        prompts = self._load_prompts()
        prompt = prompts.get(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_id}' not found in {self.prompts_path}")
        return prompt

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
        with open(self.prompts_path, "w") as f:
            json.dump(prompts, f, indent=2)

    # ============================================================
    # MEMORY MANAGEMENT  (local JSON per session — privacy-sensitive)
    # ============================================================

    def _memory_path(self, session_id: str) -> str:
        return os.path.join(self.memory_dir, f"{session_id}.json")

    def get_memory(self, session_id: str) -> str:
        """Load the conversation summary for a session from local storage."""
        path = self._memory_path(session_id)
        if not os.path.exists(path):
            return ""
        with open(path, "r") as f:
            return json.load(f).get("summary", "")

    def update_memory(self, session_id: str, conversation_text: str) -> None:
        """
        Summarise the running memory plus the latest conversation turn and persist.
        """
        existing_summary = self.get_memory(session_id)
        summary_prompt = (
            "You maintain a concise running memory for one RAG chat session.\n\n"
            "Existing session summary:\n"
            f"{existing_summary or '(none yet)'}\n\n"
            "Latest exchange:\n"
            f"{conversation_text}\n\n"
            "Write an updated cumulative summary. Preserve durable facts, user intent, "
            "important constraints, and unresolved follow-ups. Do not include unnecessary "
            "verbatim transcript."
        )
        response = self.llm.invoke(summary_prompt)

        record = {
            "session_id": session_id,
            "summary": response.content,
            "updated_at": str(datetime.datetime.utcnow()),
        }

        with open(self._memory_path(session_id), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    def clear_memory(self, session_id: str) -> None:
        """Delete the local memory file for a session (right-to-erasure)."""
        path = self._memory_path(session_id)
        if os.path.exists(path):
            os.remove(path)
            print(f"Memory cleared for session {session_id}")

    # ============================================================
    # INTERACTION LOGGING  (local JSONL per session — privacy-sensitive)
    # ============================================================

    def _interactions_path(self, session_id: str) -> str:
        return os.path.join(self.interactions_dir, f"{session_id}.jsonl")

    def _log_interaction(
        self,
        session_id: str,
        question: str,
        answer: str,
        source_ids: list,
    ) -> None:
        """
        Append one interaction record to a session-scoped JSONL file locally.
        Questions, answers, and source references never leave the local machine.
        """
        record = {
            "interaction_id": str(uuid.uuid4()),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "retrieved_doc_ids": source_ids,
            "timestamp": str(datetime.datetime.utcnow()),
        }
        with open(self._interactions_path(session_id), "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_interactions(self, session_id: str) -> list:
        """Read all logged interactions for a session from local storage."""
        path = self._interactions_path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def clear_interactions(self, session_id: str) -> None:
        """Delete the local interaction log for a session (right-to-erasure)."""
        path = self._interactions_path(session_id)
        if os.path.exists(path):
            os.remove(path)
            print(f"Interaction log cleared for session {session_id}")

    # ============================================================
    # RIGHT-TO-ERASURE HELPER
    # ============================================================

    def purge_session(self, session_id: str) -> None:
        """
        Remove ALL locally stored data for a given session.
        Call this to honour a user deletion / right-to-erasure request.
        """
        self.clear_memory(session_id)
        self.clear_interactions(session_id)
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

        # --- 5. Persist locally only ---
        self._log_interaction(session_id, question, answer, source_ids)
        self.update_memory(session_id, f"Q: {question}\nA: {answer}")

        return answer