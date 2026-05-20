import json

LOG_FILE = "../logs/access_log.json"

denied = []

with open(LOG_FILE) as f:

    for line in f:
        event = json.loads(line)

        if event["status"] == "denied":
            denied.append(event)

print("Denied Access Attempts")

for d in denied:
    print(d)