import pandas as pd
import great_expectations as gx
import chromadb

from openai import AzureOpenAI

from azure.identity import DeviceCodeCredential
from azure.keyvault.secrets import SecretClient

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

df = pd.read_csv("data/Generation_Data.csv")

print("Loaded rows:", len(df))


# ============================================================
# VALIDATE DATA WITH GREAT EXPECTATIONS
# ============================================================

context = gx.get_context()

datasource = context.data_sources.add_pandas(
    "generation_source"
)

asset = datasource.add_dataframe_asset(
    "generation_reports"
)

batch_definition = asset.add_batch_definition_whole_dataframe(
    "full_dataframe"
)

suite = context.suites.add_or_update(
    gx.ExpectationSuite(
        name="generation_suite"
    )
)

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
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="power_output_kw",
        min_value=0,
        max_value=1000
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

validation_definition = context.validation_definitions.add_or_update(
    gx.ValidationDefinition(
        name="generation_validation",
        data=batch_definition,
        suite=suite
    )
)

checkpoint = context.checkpoints.add_or_update(
    gx.Checkpoint(
        name="generation_checkpoint",
        validation_definitions=[
            validation_definition
        ]
    )
)

results = checkpoint.run(
    batch_parameters={
        "dataframe": df
    }
)

print("Validation success:", results.success)

# ============================================================
# CLEAN DATA
# ============================================================

clean_df = df.copy()

clean_df = clean_df.drop_duplicates(
    subset=["report_id"]
)

clean_df = clean_df.dropna(
    subset=[
        "inverter_id",
        "report_date"
    ]
)

clean_df = clean_df[
    clean_df["power_output_kw"].between(
        0,
        1000
    )
]

clean_df = clean_df[
    clean_df["battery_temp_c"].between(
        -20,
        80
    )
]

print("Clean rows:", len(clean_df))


# ============================================================
# CONVERT TO DOCUMENTS
# ============================================================

def row_to_document(row):

    return (
        f"Generation report {row.report_id}. "
        f"Inverter {row.inverter_id}. "
        f"Date {row.report_date}. "
        f"Power output {row.power_output_kw} kW. "
        f"Battery temperature {row.battery_temp_c} C. "
        f"Technician name: {row.technician}. "
        f"Report Comments: {row.report_text}. "
        f"Status {row.status}."
    )

documents = clean_df.apply(
    row_to_document,
    axis=1
).tolist()

ids = clean_df["report_id"].astype(str).tolist()


# ============================================================
# CREATE EMBEDDINGS USING AI FOUNDRY
# ============================================================

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

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="generation_reports"
)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Stored documents in vector DB")


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
        n_results=1
    )

    return "\n".join(
        results["documents"][0]
    )


# ============================================================
# RAG CHATBOT USING AI FOUNDRY MODEL
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
Use the following reports:
{context_text}
Answer the question:
{question}
Use this format for your answer:
        Best Match
        - Generation report
        - Inverter
        - Date
        - Power output
        - Battery temperature
        - Technician name
        - Report Comments
        - Status 

        Questions Answer

"""

        response = ai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content":
                    "You help analyze solar generation data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        print("\nAssistant:", answer)
        print()


# ============================================================
# CHATBOT
# ============================================================

chatbot()