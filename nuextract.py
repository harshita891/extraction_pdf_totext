from transformers import AutoModel, AutoTokenizer
import torch
import json

# Load the NuExtract model (replace with correct OpenVINO path if required)
model_path = "numind/nuextract-1.5"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Move to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
