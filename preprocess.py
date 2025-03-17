
import spacy
import logging
import re
import torch
import os
from tfidf import get_tfidf_stop_words



def check_pytorch():
    if not torch.cuda.is_available() and "COLAB_GPU" in os.environ:
        raise RuntimeError("This script is restricted to local PyTorch execution and cannot run on Google Colab.")

check_pytorch()

nlp = spacy.load("en_core_web_sm")

def clean_text(text, stop_words=None):
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    doc = nlp(text)

    tokens = [token.text for token in doc]
    tensor_tokens = torch.tensor([hash(token) for token in tokens], dtype=torch.long, device="cuda")

    if stop_words:
        stop_word_tensors = set(torch.tensor([hash(word) for word in stop_words], dtype=torch.long, device="cuda").tolist())
        filtered_tokens = [tokens[i] for i in range(len(tokens)) if tensor_tokens[i].item() not in stop_word_tensors]
    else:
        filtered_tokens = tokens

    return " ".join(filtered_tokens)

def preprocess_texts(text_list):
    stop_words = get_tfidf_stop_words(text_list, top_n=100)
    cleaned_texts = [clean_text(text, stop_words) for text in text_list]
    return cleaned_texts
