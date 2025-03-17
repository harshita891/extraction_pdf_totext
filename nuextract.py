from transformers import AutoModel, AutoTokenizer
import torch
import json

model_path = "numind/nuextract-1.5"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

device = "cuda"
model.to(device)
