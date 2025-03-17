import os


PDF_FOLDER = r"C:\Users\harsh\Desktop\tf-idf\Data\BE-CSBS"
OUTPUT_JSON = "pdf_data_extracted_CSBS"
FAILED_FILES_LOG = "failed_files.log"


BATCH_SIZE = 50

OCR_LANG = "eng"
OCR_CONFIG = "--oem 3 --psm 6"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
