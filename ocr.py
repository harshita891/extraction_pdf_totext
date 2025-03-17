r"""import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from config import OCR_LANG, OCR_CONFIG


from spellchecker import SpellChecker




def preprocess_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  
    enhanced = cv2.equalizeHist(blurred)
    return enhanced



def correct_ocr_text(ocr_text):
    spell = SpellChecker()
    words = ocr_text.split()
    corrected_text = []

    for word in words:
       
        corrected_word = spell.correction(word)
        corrected_text.append(corrected_word)
    
    return " ".join(corrected_text)



def extract_text_from_images(pdf_path):
    
    images = convert_from_path(pdf_path, dpi=300)
    extracted_texts = []

    for img in images:
        
        preprocessed_img = preprocess_image(img)
        
        
        ocr_text = pytesseract.image_to_string(preprocessed_img, lang=OCR_LANG, config=OCR_CONFIG)
        
       
        corrected_text = correct_ocr_text(ocr_text)
        
        extracted_texts.append(corrected_text)

    return "\n\n".join(extracted_texts).strip()"""
    
    
import cv2
import numpy as np
import pytesseract
import torch
from pdf2image import convert_from_path
from config import OCR_LANG, OCR_CONFIG
from spellchecker import SpellChecker

def preprocess_image(img):
    """Preprocess image for OCR by converting to grayscale, blurring, and enhancing contrast."""
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.equalizeHist(blurred)
    return enhanced

def correct_ocr_text(ocr_text):
    """Correct OCR-extracted text using spell checking."""
    spell = SpellChecker()
    words = ocr_text.split()
    corrected_text = []

    for word in words:
        corrected_word = spell.correction(word)
        corrected_text.append(corrected_word if corrected_word else word)  # Keep original if None

    return " ".join(corrected_text)

def extract_text_from_images(pdf_path):
    """Extract text from images generated from PDF using OCR and spell correction."""
    images = convert_from_path(pdf_path, dpi=300)
    extracted_texts = []

    for img in images:
        preprocessed_img = preprocess_image(img)
        ocr_text = pytesseract.image_to_string(preprocessed_img, lang=OCR_LANG, config=OCR_CONFIG)
        
        if not ocr_text.strip():
            continue  # Skip empty OCR results
        
        corrected_text = correct_ocr_text(ocr_text)
        extracted_texts.append(corrected_text)

    return "\n\n".join(extracted_texts).strip()
