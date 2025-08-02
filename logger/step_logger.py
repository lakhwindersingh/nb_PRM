# logger/step_logger.py
import json
import uuid
from datetime import datetime

class StepLogger:
    def __init__(self, filepath):
        self.filepath = filepath

    def log_step(self, context, output, metadata=None):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "output": output,
            "metadata": metadata or {}
        }
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(entry) + '\n')

