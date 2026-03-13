import torch
from transformers import AutoTokenizer, EsmModel

# Use a larger ESM-2 model for better classification accuracy
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)

def get_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean-pooling: Average across the sequence length (dimension 1)
    return outputs.last_hidden_state.mean(dim=1)

# Test with your first "jailbroken" variant
variant_seq = "HMPPLKEIALHVLRIDHALR..." # Paste the full sequence from your .fa file
embedding = get_embedding(variant_seq)
print(f"Vector Shape: {embedding.shape}") # Should be [1, 1280]