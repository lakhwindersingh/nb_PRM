# utils/logger.py
import os
import json
from datetime import datetime

class Logger:
    def __init__(self, log_file="logs/reasoning_log.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_step(self, context, output):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "context": context,
            "output": output
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
