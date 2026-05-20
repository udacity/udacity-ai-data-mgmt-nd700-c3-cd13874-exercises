import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

def load_secrets():

    vault_name = os.getenv("KEY_VAULT_NAME")
    url = f"https://{vault_name}.vault.azure.net/"

    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=url, credential=credential)

    secrets = {
        "endpoint": client.get_secret("azure-openai-endpoint").value,
        "key": client.get_secret("azure-openai-key").value,
        "version": client.get_secret("azure-openai-version").value,
        "deployment": client.get_secret("azure-openai-deployment").value
    }

    return secrets