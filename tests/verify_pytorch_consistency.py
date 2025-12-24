"""Verify PyTorch encoder output is consistent."""
import torch
from transformers import FSMTForConditionalGeneration, T5Tokenizer

model_path = r"d:\HyperscaleES\user"

# Load FRESH
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_model.eval()
pt_tokenizer = T5Tokenizer.from_pretrained(model_path)

text = "Hello world"
print(f"Input: '{text}'")

# Run 3 times to check consistency
for i in range(3):
    inputs = pt_tokenizer(text, return_tensors='pt')
    print(f"\nRun {i+1}:")
    print(f"  Token IDs: {inputs['input_ids'].numpy()}")
    
    with torch.no_grad():
        output = pt_model.model.encoder(inputs['input_ids']).last_hidden_state
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output[0,0,:5]: {output[0, 0, :5].numpy()}")
    print(f"  Output[0,1,:5]: {output[0, 1, :5].numpy()}")
    print(f"  Output[0,2,:5]: {output[0, 2, :5].numpy()}")
