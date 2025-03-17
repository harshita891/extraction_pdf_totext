import spacy
import logging
import re
import torch
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from config import OUTPUT_JSON

# Ensure PyTorch runs only on local, not Colab
def check_pytorch():
    if not torch.cuda.is_available() and "COLAB_GPU" in os.environ:
        raise RuntimeError("This script is restricted to local PyTorch execution and cannot run on Google Colab.")

check_pytorch()

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

def load_text_data(json_files):
   
    corpus = []
    file_map = {}

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                text = item["text"]
                corpus.append(text)
                file_map[len(corpus) - 1] = item  # Map index to original JSON item

    return corpus, file_map

def compute_tfidf(corpus, top_n=100):
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    
    X_torch = torch.tensor(X.toarray(), dtype=torch.float32, device="cuda")
    mean_tfidf = torch.mean(X_torch, dim=0)
    word_scores = list(zip(feature_names, mean_tfidf.tolist()))

    
    word_scores.sort(key=lambda x: x[1], reverse=True)
    frequent_words = word_scores[:top_n]
    custom_stop_words = [word for word, _ in frequent_words]

    return custom_stop_words, frequent_words

def get_tfidf_stop_words(json_folder, top_n=100):
    
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith(".json")]

    if not json_files:
        logging.error("No JSON files found for TF-IDF processing.")
        return []

    corpus, _ = load_text_data(json_files)
    stop_words, _ = compute_tfidf(corpus, top_n=top_n)

    return stop_words

def remove_tfidf_stop_words(file_map, stop_words):
    
    cleaned_data = []
    stop_words_set = set(stop_words)

    for idx, item in file_map.items():
        words = item["text"].split()
        tensor_words = torch.tensor([hash(word) for word in words], dtype=torch.int64, device="cuda")
        stop_words_tensor = torch.tensor([hash(word) for word in stop_words_set], dtype=torch.int64, device="cuda")
        mask = torch.isin(tensor_words, stop_words_tensor, invert=True)
        filtered_text = " ".join([words[i] for i in range(len(words)) if mask[i].item()])
        item["text"] = filtered_text
        cleaned_data.append(item)

    return cleaned_data

def save_cleaned_json(cleaned_data):
    
    output_file = f"{OUTPUT_JSON}_cleaned.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Cleaned JSON saved as {output_file}")

def save_frequent_words(frequent_words):
    
    output_file = f"{OUTPUT_JSON}_frequent_words.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for word, score in frequent_words:
            f.write(f"{word}: {score}\n")
    logging.info(f"Frequent words saved as {output_file}")

def process_tfidf(json_folder):
    
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith(".json")]

    if not json_files:
        logging.error("No JSON files found for TF-IDF processing.")
        return

    corpus, file_map = load_text_data(json_files)
    stop_words, frequent_words = compute_tfidf(corpus, top_n=100)
    cleaned_data = remove_tfidf_stop_words(file_map, stop_words)

    save_cleaned_json(cleaned_data)
    save_frequent_words(frequent_words)

    return stop_words
