import json
import datetime

LOG_FILE = "logs/access_log.json"

def log_event(user, action, status, resource):

    event = {
        "timestamp": str(datetime.datetime.utcnow()),
        "user": user,
        "action": action,
        "resource": resource,
        "status": status
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")