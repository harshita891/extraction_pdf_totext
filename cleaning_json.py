import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

print(torch.__version__)




device = torch.device('cuda')
print(f"Using device: {device}")



tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)


def clean_text_with_t5(raw_text):
   
    input_text = f"clean: {raw_text}"
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    
    
    cleaned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return cleaned_text


def clean_json_file(input_file, output_file):
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    
    if isinstance(data, list):
        
        for entry in data:
            for key, value in entry.items():
                if isinstance(value, str):  
                    print(f"Cleaning text for key: {key}")
                    entry[key] = clean_text_with_t5(value)
    elif isinstance(data, dict):
        
        for key, value in data.items():
            if isinstance(value, str):
                print(f"Cleaning text for key: {key}")
                data[key] = clean_text_with_t5(value)
    
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Cleaned data saved to {output_file}")


input_json = r"C:\Users\harsh\Desktop\tf-idf\pdf_data_extracted_batch_1.json"
output_json = "cleaned.json"
clean_json_file(input_json, output_json)
