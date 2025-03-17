import json
import logging
from config import OUTPUT_JSON

def save_json(data, batch_num):
    batch_output = f"{OUTPUT_JSON}_batch_{batch_num}.json"
    try:
        with open(batch_output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Exported {len(data)} tasks to {batch_output}")
    except Exception as e:
        logging.error(f"Error writing JSON file: {e}")
