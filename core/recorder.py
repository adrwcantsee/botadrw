import json
import os
from datetime import datetime

LOG_DIR = "data/training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def save_turn_data(data):
    """Appends one training turn record to a daily JSON log file."""
    filename = datetime.now().strftime("%Y-%m-%d") + ".json"
    path = os.path.join(LOG_DIR, filename)

    # Read existing data
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    # Append new record
    existing.append(data)

    # Write updated log
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
