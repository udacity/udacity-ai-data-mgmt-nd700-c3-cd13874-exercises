from pathlib import Path
from typing import Dict, Any

import yaml
from azure.identity import InteractiveBrowserCredential
from azure.keyvault.secrets import SecretClient
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

PARAMS_PATH = Path("params.yaml")


def load_params() -> Dict[str, Any]:
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_secrets() -> Dict[str, str]:
    """
    Loads Azure OpenAI connection details from Azure Key Vault.

    Required Key Vault secrets:
    - azure-openai-endpoint
    - azure-openai-key
    - azure-openai-api-version
    """
    credential = InteractiveBrowserCredential(additionally_allowed_tenants=["*"])
    kv_uri = "https://useraisecrets.vault.azure.net/"
    client = SecretClient(vault_url=kv_uri, credential=credential)

    azure_endpoint = client.get_secret("azure-openai-endpoint").value
    azure_api_key = client.get_secret("azure-openai-key").value
    azure_api_version = client.get_secret("azure-openai-api-version").value

    return {
        "azure_endpoint": azure_endpoint,
        "azure_api_key": azure_api_key,
        "azure_api_version": azure_api_version,
    }


def get_embeddings() -> AzureOpenAIEmbeddings:
    params = load_params()
    secrets = load_secrets()

    return AzureOpenAIEmbeddings(
        azure_endpoint=secrets["azure_endpoint"],
        api_key=secrets["azure_api_key"],
        api_version=secrets["azure_api_version"],
        model=params["azure"]["embedding_deployment"],
    )


def get_llm() -> AzureChatOpenAI:
    params = load_params()
    secrets = load_secrets()

    return AzureChatOpenAI(
        azure_endpoint=secrets["azure_endpoint"],
        api_key=secrets["azure_api_key"],
        api_version=secrets["azure_api_version"],
        deployment_name=params["azure"]["chat_deployment"],
        temperature=0,
    )