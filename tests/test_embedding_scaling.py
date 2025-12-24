"""Check if PyTorch FSMT actually scales embeddings."""
import torch
from transformers import FSMTForConditionalGeneration, T5Tokenizer

model_path = r"d:\HyperscaleES\user"
model = FSMTForConditionalGeneration.from_pretrained(model_path)
model.eval()
tokenizer = T5Tokenizer.from_pretrained(model_path)

print(f"Config scale_embedding: {model.config.scale_embedding}")
print(f"Config d_model: {model.config.d_model}")

# Test input
text = "Hello world"
inputs = tokenizer(text, return_tensors='pt')

# Get raw embeddings
with torch.no_grad():
    token_embeds = model.model.encoder.embed_tokens(inputs['input_ids'])
    print(f"\nToken embeddings shape: {token_embeds.shape}")
    print(f"Token embeddings (first token, first 5 dims): {token_embeds[0, 0, :5]}")
    
    # Check if model scales embeddings
    embed_weight = model.model.encoder.embed_tokens.weight
    token_id = inputs['input_ids'][0, 0].item()
    raw_embed = embed_weight[token_id]
    
    print(f"\nRaw embedding from weight (first 5): {raw_embed[:5]}")
    print(f"After embed_tokens (first 5): {token_embeds[0, 0, :5]}")
    
    # Check scaling
    scaled_manual = raw_embed * (model.config.d_model ** 0.5)
    print(f"\nManual scaling by sqrt(d_model) (first 5): {scaled_manual[:5]}")
    
    # Check if they match
    matches_raw = torch.allclose(token_embeds[0, 0], raw_embed)
    matches_scaled = torch.allclose(token_embeds[0, 0], scaled_manual)
    
    print(f"\nMatches raw (no scaling): {matches_raw}")
    print(f"Matches scaled (sqrt(d_model)): {matches_scaled}")
    
    # Get full encoder output with embeddings  
    positions = torch.arange(inputs['input_ids'].shape[1]).unsqueeze(0)
    pos_embeds = model.model.encoder.embed_positions(positions)
    
    print(f"\nPosition embeddings (first token, first 5): {pos_embeds[0, 0, :5]}")
    
    full_embeds = token_embeds + pos_embeds
    print(f"Full embeddings (token + pos, first 5): {full_embeds[0, 0, :5]}")
