


import logging
import os
import fitz
import pdfplumber
import re
import json
from ocr import extract_text_from_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_text_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc]).strip()
        return text
    except Exception as e:
        logging.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        return ""

def extract_text_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()
        return text
    except Exception as e:
        logging.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
        return ""

def clean_text(text):
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_pdf(file_path):
    text = extract_text_pymupdf(file_path)

    if not text.strip():
        logging.warning(f"PyMuPDF failed for {file_path}, using pdfplumber.")
        text = extract_text_pdfplumber(file_path)

    if not text.strip():
        logging.warning(f"No selectable text found in {file_path}, using OCR.")
        text = extract_text_from_images(file_path)

    return clean_text(text)
