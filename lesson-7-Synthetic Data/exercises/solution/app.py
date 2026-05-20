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
from presidio_anonymizer import AnonymizerEngine


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

REFERENCE_FILE = "reference.md"
CLEAN_REFERENCE_FILE = "reference_clean.md"

OUTPUT_TRAIN_FILE = "train.jsonl"
OUTPUT_VALID_FILE = "valid.jsonl"
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

# supported: "redact", "replace", "report_only"
PII_ACTION = "redact"

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
        "azure_endpoint": client.get_secret("azure-end-point").value,
        "azure_api_key": client.get_secret("azure-api-key").value,
        "azure_api_version": client.get_secret("azure-api-version").value,
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
# SAVE JSON / JSONL / TEXT
# ---------------------------------------------------

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------
# PII CLEANING WITH MICROSOFT PRESIDIO
# ---------------------------------------------------

def get_pii_analyzer() -> AnalyzerEngine:
    if AnalyzerEngine is None:
        raise ImportError(
            "Presidio is not installed. Install with: "
            "pip install presidio-analyzer presidio-anonymizer spacy && "
            "python -m spacy download en_core_web_lg"
        )
    return AnalyzerEngine()


def get_pii_anonymizer() -> AnonymizerEngine:
    if AnonymizerEngine is None:
        raise ImportError(
            "Presidio anonymizer is not installed. Install with: "
            "pip install presidio-anonymizer"
        )
    return AnonymizerEngine()


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


def redact_text_spans(text: str, findings) -> str:
    """
    Replace detected PII spans with entity labels, preserving the rest of the text.
    Example: John Smith -> [PERSON]
    """
    if not findings:
        return text

    findings = sorted(findings, key=lambda x: x.start)
    result = []
    last_idx = 0

    for f in findings:
        if f.start < last_idx:
            continue
        result.append(text[last_idx:f.start])
        result.append(f"[{f.entity_type}]")
        last_idx = f.end

    result.append(text[last_idx:])
    return "".join(result)


def clean_reference_document(
    text: str,
    analyzer: AnalyzerEngine,
    entities: List[str] | None = None,
    language: str = "en",
    score_threshold: float = 0.5,
    action: str = "redact",
) -> Tuple[str, List[Dict[str, Any]]]:
    findings = detect_pii_in_text(
        analyzer=analyzer,
        text=text,
        entities=entities,
        language=language,
        score_threshold=score_threshold,
    )

    finding_report = [
        {
            "entity_type": f.entity_type,
            "score": f.score,
            "start": f.start,
            "end": f.end,
            "text": text[f.start:f.end],
        }
        for f in findings
    ]

    if not findings or action == "report_only":
        return text, finding_report

    if action == "redact":
        cleaned = redact_text_spans(text, findings)
        return cleaned, finding_report

    if action == "replace":
        anonymizer = get_pii_anonymizer()
        anonymized_result = anonymizer.anonymize(text=text, analyzer_results=findings)
        return anonymized_result.text, finding_report

    raise ValueError("PII_ACTION must be one of: redact, replace, report_only")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    secrets = load_secrets()
    client = get_openai_client(secrets)
    deployment = secrets["chat_deployment"]

    print("Loading document...")
    raw_text = load_reference_document(REFERENCE_FILE)
    raw_text = normalize_text(raw_text)

    # -----------------------------
    # STEP 1: CLEAN REFERENCE DOC
    # -----------------------------
    print("Running PII cleaning on reference document...")
    pii_analyzer = get_pii_analyzer()

    clean_text, reference_pii_findings = clean_reference_document(
        text=raw_text,
        analyzer=pii_analyzer,
        entities=PII_ENTITIES,
        language=PII_LANG,
        score_threshold=PII_SCORE_THRESHOLD,
        action=PII_ACTION,
    )

    clean_text = normalize_text(clean_text)
    write_text(CLEAN_REFERENCE_FILE, clean_text)

    # -----------------------------
    # STEP 2: GENERATE DATASET FROM CLEAN DOC
    # -----------------------------
    chunks = split_into_chunks(clean_text, CHUNK_SIZE, CHUNK_OVERLAP)

    examples = []

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}/{len(chunks)}")
        pairs = generate_pairs(client, deployment, chunk)
        pairs = review_pairs(client, deployment, chunk, pairs)

        for p in pairs:
            if "question" in p and "answer" in p:
                examples.append(to_chat_format(p["question"], p["answer"]))

    random.shuffle(examples)

    split = int(len(examples) * (1 - VALIDATION_SPLIT))
    train = examples[:split]
    valid = examples[split:]

    pii_report = {
        "config": {
            "language": PII_LANG,
            "score_threshold": PII_SCORE_THRESHOLD,
            "action": PII_ACTION,
            "entities": PII_ENTITIES,
        },
        "reference_document": {
            "source_file": REFERENCE_FILE,
            "clean_file": CLEAN_REFERENCE_FILE,
            "pii_findings_count": len(reference_pii_findings),
            "findings": reference_pii_findings,
        },
        "dataset_summary": {
            "total_examples": len(examples),
            "train_examples": len(train),
            "valid_examples": len(valid),
        },
    }

    write_json(PII_REPORT_FILE, pii_report)

    write_jsonl(OUTPUT_TRAIN_FILE, train)
    write_jsonl(OUTPUT_VALID_FILE, valid)

    print("Done")
    print("Clean reference file:", CLEAN_REFERENCE_FILE)
    print("Train:", len(train))
    print("Valid:", len(valid))
    print("PII report:", PII_REPORT_FILE)


if __name__ == "__main__":
    main()