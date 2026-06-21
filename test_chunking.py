import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

text = "Ini adalah contoh teks."
tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
print("Tokens:", tokens.shape)

chunk_size = 510
chunks = [tokens[j:j+chunk_size] for j in range(0, len(tokens), chunk_size)]

chunk_embs = []
for chunk in chunks:
    input_ids = torch.cat([
        torch.tensor([tokenizer.cls_token_id]), 
        chunk, 
        torch.tensor([tokenizer.sep_token_id])
    ]).unsqueeze(0)
    
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print("Outputs:", outputs.last_hidden_state.shape)
