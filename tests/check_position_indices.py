"""Check how PyTorch creates position indices."""
import torch
from transformers import FSMTForConditionalGeneration, T5Tokenizer

model_path = r"d:\HyperscaleES\user"
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_tokenizer = T5Tokenizer.from_pretrained(model_path)

text = "Hello world"
inputs = pt_tokenizer(text, return_tensors='pt')

print(f"Input: '{text}'")
print(f"Token IDs: {inputs['input_ids'].numpy()}")

# Get position embeddings
with torch.no_grad():
    # Method 1: Use the module directly
    pos_embeds = pt_model.model.encoder.embed_positions(inputs['input_ids'])
    print(f"\nPosition embeddings shape: {pos_embeds.shape}")
    print(f"Position embeddings[0,0,:5]: {pos_embeds[0, 0, :5].numpy()}")
    print(f"Position embeddings[0,1,:5]: {pos_embeds[0, 1, :5].numpy()}")
    print(f"Position embeddings[0,2,:5]: {pos_embeds[0, 2, :5].numpy()}")
    
    # Check what positions are actually used
    # Access the weight matrix
    print(f"\nPosition weight shape: {pt_model.model.encoder.embed_positions.weight.shape}")
    print(f"Position weight[0,:5]: {pt_model.model.encoder.embed_positions.weight[0, :5].numpy()}")
    print(f"Position weight[1,:5]: {pt_model.model.encoder.embed_positions.weight[1, :5].numpy()}")
    print(f"Position weight[2,:5]: {pt_model.model.encoder.embed_positions.weight[2, :5].numpy()}")
    
    # Compare: are position embeds using indices 0,1,2 or something else?
    print("\nDo position embeddings match weight[0], weight[1], weight[2]?")
    print(f"Match [0]: {torch.allclose(pos_embeds[0,0], pt_model.model.encoder.embed_positions.weight[0], atol=1e-6)}")
    print(f"Match [1]: {torch.allclose(pos_embeds[0,1], pt_model.model.encoder.embed_positions.weight[1], atol=1e-6)}")
    print(f"Match [2]: {torch.allclose(pos_embeds[0,2], pt_model.model.encoder.embed_positions.weight[2], atol=1e-6)}")
    
    # Try other indices
    print("\nTrying offset indices (2, 3, 4):")
    print(f"Match [2]: {torch.allclose(pos_embeds[0,0], pt_model.model.encoder.embed_positions.weight[2], atol=1e-6)}")
    print(f"Match [3]: {torch.allclose(pos_embeds[0,1], pt_model.model.encoder.embed_positions.weight[3], atol=1e-6)}")
    print(f"Match [4]: {torch.allclose(pos_embeds[0,2], pt_model.model.encoder.embed_positions.weight[4], atol=1e-6)}")
