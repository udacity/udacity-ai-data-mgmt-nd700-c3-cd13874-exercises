"""
============================================================
PRODUCTION PATTERN: GREAT EXPECTATIONS VALIDATION PIPELINE
WITH AUDIT TRAIL AND RAG CHATBOT
============================================================

FEATURES:
• Expectation Suite = single source of truth
• Automatic filtering using GE results
• Audit dataset export (invalid rows)
• Validation report export
• Clean data only enters vector database
"""

import os
import json
import pandas as pd
import great_expectations as gx
import chromadb

from openai import AzureOpenAI
from azure.identity import DeviceCodeCredential
from azure.keyvault.secrets import SecretClient


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "data/Generation_Data.csv"

AUDIT_DIR = "logs"
INVALID_ROWS_FILE = f"logs/invalid_generation_reports.csv"
VALIDATION_REPORT_FILE = f"logs/validation_report.json"

os.makedirs(AUDIT_DIR, exist_ok=True)


# ============================================================
#  Secrets
# ============================================================

keyVaultName = "data-management-kv-pk"
KVUri = f"https://{keyVaultName}.vault.azure.net/"

print("Connecting to Azure for authentication.")

credential = DeviceCodeCredential(additionally_allowed_tenants=["*"])
client = SecretClient(vault_url=KVUri, credential=credential)

endpoint = retrieved_secret = client.get_secret("structuredazureendpoint").value
api_key = retrieved_secret = client.get_secret("structuredazureapikey").value
api_version = "2024-12-01-preview"
CHAT_DEPLOYMENT = "gpt-4.1-mini"
EMBED_DEPLOYMENT = "text-embedding-ada-002"

# ============================================================
#  Connect to Microsoft AI Foundry model
# ============================================================

ai_client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("Loaded rows:", len(df))


# ============================================================
# GREAT EXPECTATIONS SETUP
# ============================================================

context = gx.get_context()

datasource = context.data_sources.add_or_update_pandas(
    name="generation_source"
)

asset = datasource.add_dataframe_asset(
    name="generation_reports"
)

batch_definition = asset.add_batch_definition_whole_dataframe(
    name="full_dataframe"
)

suite = context.suites.add_or_update(
    gx.ExpectationSuite(
        name="generation_suite"
    )
)


# ============================================================
# DEFINE EXPECTATIONS
# ============================================================

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(
        column="report_id"
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="inverter_id"
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="report_date"
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="power_output_kw",
        min_value=0,
        max_value=1000,
        row_condition='power_output_kw != -999.5',
        condition_parser="pandas"
    )
)

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="battery_temp_c",
        min_value=-20,
        max_value=80
    )
)

context.suites.add_or_update(suite)


# ============================================================
# VALIDATE USING VALIDATOR
# ============================================================

batch = batch_definition.get_batch(
    batch_parameters={"dataframe": df}
)

validator = context.get_validator(
    batch=batch,
    expectation_suite=suite
)

results = validator.validate(
    result_format={
        "result_format": "COMPLETE"
    }
)

print("Validation success:", results["success"])


# ============================================================
# SAVE FULL VALIDATION REPORT
# ============================================================

validation_json = results.to_json_dict()

with open(VALIDATION_REPORT_FILE, "w") as f:

    json.dump(
        validation_json,
        f,
        indent=2
    )

print("Validation report saved:",
      VALIDATION_REPORT_FILE)


# ============================================================
# EXTRACT INVALID ROW INDICES
# ============================================================

invalid_indices = set()

for result in results["results"]:

    idx_list = result["result"].get(
        "unexpected_index_list"
    )

    if idx_list:

        invalid_indices.update(idx_list)


print("Invalid row count:", len(invalid_indices))


# ============================================================
# CREATE CLEAN AND AUDIT DATASETS
# ============================================================

invalid_df = df.loc[list(invalid_indices)]

clean_df = df.drop(
    index=invalid_indices
)


print("Clean rows:", len(clean_df))
print("Invalid rows:", len(invalid_df))


# ============================================================
# SAVE AUDIT DATASET
# ============================================================

invalid_df.to_csv(
    INVALID_ROWS_FILE,
    index=False
)

print("Invalid rows saved:",
      INVALID_ROWS_FILE)


# ============================================================
# CONVERT CLEAN DATA TO DOCUMENTS
# ============================================================

def row_to_document(row):

    return (
        f"Generation report {row.report_id}. "
        f"Site {row.site_name}. "
        f"Inverter {row.inverter_id}. "
        f"Date {row.report_date}. "
        f"Power output {row.power_output_kw} kW. "
        f"Battery temperature {row.battery_temp_c} C. "
        f"Technician {row.technician}. "
        f"Comments {row.report_text}. "
        f"Status {row.status}."
    )


documents = clean_df.apply(
    row_to_document,
    axis=1
).tolist()

ids = clean_df["report_id"].astype(str).tolist()


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("Creating embeddings...")

embeddings = []

for doc in documents:

    response = ai_client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=doc
    )

    embeddings.append(
        response.data[0].embedding
    )


# ============================================================
# STORE IN VECTOR DATABASE
# ============================================================

print("Storing in ChromaDB...")

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="generation_reports"
)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Stored clean documents:", len(ids))


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_context(question):

    emb = ai_client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=question
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[emb],
        n_results=2
    )

    return "\n".join(
        results["documents"][0]
    )


# ============================================================
# CHATBOT
# ============================================================

def chatbot():

    print("\nSolar Generation AI Assistant")
    print("Type exit to quit\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":

            break

        context_text = retrieve_context(
            question
        )

        prompt = f"""
You are an energy operations assistant.
Always use the most current reports as the current state, and use older reports for historical context.

Reports:
{context_text}

Question:
{question}

Provide structured answer.
"""

        response = ai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content":
                    "You analyze solar generation data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        print(
            "\nAssistant:",
            response.choices[0].message.content
        )


# ============================================================
# RUN CHATBOT
# ============================================================

chatbot()
