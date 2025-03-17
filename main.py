   
    
import os
import logging
import torch
from extract_json import extract_text_from_pdf
from preprocess import preprocess_texts
from save_json import save_json
from utils import extract_roll_number
from config import PDF_FOLDER, OUTPUT_JSON, BATCH_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def check_pytorch():
    if not torch.cuda.is_available():
        raise RuntimeError("This script is restricted to local PyTorch execution and cannot run on Google Colab.")

check_pytorch()

def process_pdfs():
    if not os.path.exists(PDF_FOLDER):
        logging.error(f"Folder '{PDF_FOLDER}' does not exist.")
        return

    all_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    tasks = []
    texts = []

    for i, filename in enumerate(all_files):
        file_path = os.path.join(PDF_FOLDER, filename)

        logging.info(f"Processing ({i+1}/{len(all_files)}): {filename}")
        extracted_text = extract_text_from_pdf(file_path)

        if extracted_text:
            texts.append(extracted_text)
            roll_no = extract_roll_number(filename)

            tasks.append({
                "roll_no": roll_no,
                "filename": filename,
                "text": extracted_text
            })

    if not texts:
        logging.warning("No text extracted from PDFs. Exiting.")
        return

    logging.info("Applying TF-IDF filtering and text cleaning...")
    cleaned_texts = preprocess_texts(texts)

    for idx, task in enumerate(tasks):
        task["text"] = cleaned_texts[idx]

    save_json(tasks, 1)

    logging.info("Processing complete. Cleaned JSON saved.")

if __name__ == "__main__":
    process_pdfs()

