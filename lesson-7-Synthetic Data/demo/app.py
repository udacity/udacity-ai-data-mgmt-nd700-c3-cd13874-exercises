from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any

from openai import AzureOpenAI

from azure.identity import DeviceCodeCredential
from azure.keyvault.secrets import SecretClient

from pypdf import PdfReader

from presidio_analyzer import AnalyzerEngine


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

REFERENCE_FILE = "reference.md"
OUTPUT_TRAIN_FILE = "train.jsonl"
OUTPUT_VALID_FILE = "valid.jsonl"
OUTPUT_TRAIN_FILE_CLEAN = "train_clean.jsonl"
OUTPUT_VALID_FILE_CLEAN = "valid_clean.jsonl"
PII_REPORT_FILE = "pii_report.json"

QUESTION_STYLE = "long"
PAIRS_PER_CHUNK = 8
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300

VALIDATION_SPLIT = 0.2
MIN_REVIEW_SCORE = 4.0

SYSTEM_PROMPT_FOR_FINETUNE = (
    "You are a helpful assistant. "
    "You will be presented with a question, please provide a clear and accurate answer."
)

MAX_OUTPUT_TOKENS = 4000

# Presidio config
PII_LANG = "en"
PII_SCORE_THRESHOLD = 0.5
PII_ACTION = "drop"   # supported: "drop", "report_only"

# Set to None to use Presidio defaults, or restrict to a subset like:
# ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN"]
PII_ENTITIES = None


# ============================================================
# LOAD SECRETS FROM AZURE KEY VAULT
# ============================================================

def load_secrets():

    print("Authenticating with Azure...")

    keyVaultName = "data-management-kv-pk"
    KVUri = f"https://{keyVaultName}.vault.azure.net/"

    print("Connecting to Azure for authentication.")

    credential = DeviceCodeCredential(additionally_allowed_tenants=["*"])
    client = SecretClient(vault_url=KVUri, credential=credential)

    secrets = {
        "azure_endpoint": client.get_secret("azure-openai-endpoint").value,
        "azure_api_key": client.get_secret("azure-openai-key").value,
        "azure_api_version": client.get_secret("azure-openai-api-version").value,
        "embedding_deployment": "text-embedding-ada-002",
        "chat_deployment": "gpt-4.1-mini",
        "azure_storage_connection_string": client.get_secret("azure-storage-connection-string").value,
    }

    return secrets


# ---------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------

def get_openai_client(secrets: Dict[str, str]) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=secrets["azure_endpoint"],
        api_key=secrets["azure_api_key"],
        api_version=secrets["azure_api_version"],
    )


# ---------------------------------------------------
# LOAD DOCUMENT
# ---------------------------------------------------

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("Install pypdf")
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n\n".join(text)


def load_reference_document(path_str: str) -> str:
    path = Path(path_str)
    if path.suffix.lower() in [".txt", ".md"]:
        return load_text_file(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf_file(path)
    raise ValueError("Supported input types: txt, md, pdf")


# ---------------------------------------------------
# TEXT CLEANUP
# ---------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------
# CHUNKING
# ---------------------------------------------------

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


# ---------------------------------------------------
# GENERATION
# ---------------------------------------------------

def build_generation_prompt(chunk: str) -> str:
    return f"""
Generate {PAIRS_PER_CHUNK} question answer pairs based only on the text below.

Rules:
- Do not invent information.
- Questions must be diverse.
- Avoid duplicates.

Return JSON:

{{
 "pairs":[
   {{"question":"...","answer":"..."}}
 ]
}}

Text:
\"\"\"
{chunk}
\"\"\"
"""


def generate_pairs(client, deployment, chunk):
    prompt = build_generation_prompt(chunk)
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You generate training datasets."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("pairs", [])


# ---------------------------------------------------
# REVIEW STEP
# ---------------------------------------------------

def review_pairs(client, deployment, chunk, pairs):
    prompt = f"""
Evaluate these Q&A pairs for fine tuning quality.

Reject pairs that:
- are repetitive
- not grounded in the text
- weak answers

Return JSON:

{{
 "approved":[
   {{"question":"...","answer":"..."}}
 ]
}}

Text:
{chunk}

Pairs:
{json.dumps(pairs, indent=2)}
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You review datasets."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("approved", [])


# ---------------------------------------------------
# FORMAT DATASET
# ---------------------------------------------------

def to_chat_format(question, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_FINETUNE},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# ---------------------------------------------------
# SAVE JSONL
# ---------------------------------------------------

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------
# PII CHECK WITH MICROSOFT PRESIDIO
# ---------------------------------------------------

def get_pii_analyzer() -> AnalyzerEngine:
    if AnalyzerEngine is None:
        raise ImportError(
            "Presidio is not installed. Install with: "
            "pip install presidio-analyzer spacy && "
            "python -m spacy download en_core_web_lg"
        )
    return AnalyzerEngine()


def record_to_text(record: Dict[str, Any]) -> str:
    parts = []
    for msg in record.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def detect_pii_in_text(
    analyzer: AnalyzerEngine,
    text: str,
    entities: List[str] | None = None,
    language: str = "en",
    score_threshold: float = 0.5,
):
    return analyzer.analyze(
        text=text,
        entities=entities,
        language=language,
        score_threshold=score_threshold,
    )


def filter_pii_records(
    records: List[Dict[str, Any]],
    analyzer: AnalyzerEngine,
    entities: List[str] | None = None,
    language: str = "en",
    score_threshold: float = 0.5,
    action: str = "drop",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean_records = []
    flagged_records = []

    for idx, record in enumerate(records):
        text = record_to_text(record)
        findings = detect_pii_in_text(
            analyzer=analyzer,
            text=text,
            entities=entities,
            language=language,
            score_threshold=score_threshold,
        )

        if findings:
            flagged_records.append(
                {
                    "index": idx,
                    "findings": [
                        {
                            "entity_type": f.entity_type,
                            "score": f.score,
                            "start": f.start,
                            "end": f.end,
                            "text": text[f.start:f.end],
                        }
                        for f in findings
                    ],
                    "record": record,
                }
            )

            if action == "report_only":
                clean_records.append(record)
        else:
            clean_records.append(record)

    return clean_records, flagged_records


def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    secrets = load_secrets()
    client = get_openai_client(secrets)
    deployment = secrets["chat_deployment"]

    print("Loading document...")
    text = normalize_text(load_reference_document(REFERENCE_FILE))
    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    examples = []

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}/{len(chunks)}")
        pairs = generate_pairs(client, deployment, chunk)
        pairs = review_pairs(client, deployment, chunk, pairs)

        for p in pairs:
            examples.append(
                to_chat_format(p["question"], p["answer"])
            )

    random.shuffle(examples)

    split = int(len(examples) * (1 - VALIDATION_SPLIT))
    train = examples[:split]
    valid = examples[split:]

    # -----------------------------
    # PII CHECK
    # -----------------------------
    print("Running PII check with Microsoft Presidio...")
    pii_analyzer = get_pii_analyzer()

    train_clean, train_flagged = filter_pii_records(
        records=train,
        analyzer=pii_analyzer,
        entities=PII_ENTITIES,
        language=PII_LANG,
        score_threshold=PII_SCORE_THRESHOLD,
        action=PII_ACTION,
    )

    valid_clean, valid_flagged = filter_pii_records(
        records=valid,
        analyzer=pii_analyzer,
        entities=PII_ENTITIES,
        language=PII_LANG,
        score_threshold=PII_SCORE_THRESHOLD,
        action=PII_ACTION,
    )

    pii_report = {
        "config": {
            "language": PII_LANG,
            "score_threshold": PII_SCORE_THRESHOLD,
            "action": PII_ACTION,
            "entities": PII_ENTITIES,
        },
        "summary": {
            "train_before": len(train),
            "train_after": len(train_clean),
            "train_flagged": len(train_flagged),
            "valid_before": len(valid),
            "valid_after": len(valid_clean),
            "valid_flagged": len(valid_flagged),
        },
        "train_flagged": train_flagged,
        "valid_flagged": valid_flagged,
    }

    write_json(PII_REPORT_FILE, pii_report)

    write_jsonl(OUTPUT_TRAIN_FILE, train)
    write_jsonl(OUTPUT_VALID_FILE, valid)

    write_jsonl(OUTPUT_TRAIN_FILE_CLEAN, train_clean)
    write_jsonl(OUTPUT_VALID_FILE_CLEAN, valid_clean)

    print("Done")
    print("Train:", len(train_clean))
    print("Valid:", len(valid_clean))
    print("PII report:", PII_REPORT_FILE)


if __name__ == "__main__":
    main()