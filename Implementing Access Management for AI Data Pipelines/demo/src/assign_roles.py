import json

POLICY_FILE = "../config/policies.json"

with open(POLICY_FILE) as f:
    policies = json.load(f)

new_role = "procurement_manager"

policies[new_role] = {
    "read_stages": ["procurement"],
    "write_embeddings": False
}

with open(POLICY_FILE,"w") as f:
    json.dump(policies,f,indent=2)

print("Role added:", new_role)