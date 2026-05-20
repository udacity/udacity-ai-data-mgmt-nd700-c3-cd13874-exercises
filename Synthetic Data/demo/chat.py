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

endpoint = client.get_secret("azure-end-point").value
api_key = client.get_secret("azure-api-key").value
api_version = client.get_secret("azure-api-version").value
CHAT_DEPLOYMENT_1 = "gpt-4-04-14"
CHAT_DEPLOYMENT_2 = "gpt-4.1-mini"

# ============================================================
#  Connect to Microsoft AI Foundry model
# ============================================================

ai_client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

# ============================================================
# CHATBOT USING AI FOUNDRY MODEL
# ============================================================

def chatbot():

    print("\nRetail Store Operations Guide Assistant")
    print("Type exit to quit\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        

        prompt = f"""
You are a Retail Store Operations Guide Assistant.

Answer the question:
{question}

"""
        messages=[
                {
                    "role": "system",
                    "content":
                    "You help with Retail Store Operations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        response = ai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT_1,
            messages=messages,
            temperature=0
        )
        answer = response.choices[0].message.content
        print("\nAssistant with trained model:", answer)

        response = ai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT_2,
            messages=messages,
            temperature=0
        )
        answer = response.choices[0].message.content
        print("\nAssistant with un-trained model:", answer)

        print()


# ============================================================
# CHATBOT
# ============================================================

chatbot()