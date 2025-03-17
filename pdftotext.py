import json

LABEL_STUDIO_JSON = r"C:\Users\harsh\Desktop\tf-idf\project-6-at-2025-02-15-18-48-ea88d8cf.json"
TEXT_FILE = "training_texts.txt"

def extract_text(json_path, output_txt_file):
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_txt_file, "w", encoding="utf-8") as out_f:
        for item in data:
            text = item.get("data", {}).get("text", "")
            if text:
                out_f.write(text + "\n")

extract_text(LABEL_STUDIO_JSON, TEXT_FILE)
print(f"Extracted text saved to {TEXT_FILE}")
