"""Check PyTorch FSMT encoder structure in detail."""
import torch
from transformers import FSMTForConditionalGeneration, T5Tokenizer

model_path = r"d:\HyperscaleES\user"
model = FSMTForConditionalGeneration.from_pretrained(model_path)
model.eval()
tokenizer = T5Tokenizer.from_pretrained(model_path)

print("Full encoder structure:")
print(model.model.encoder)
print("\n" + "="*80)

# Check if there's a final layer norm
print("Checking for final layer norm after all encoder layers...")
encoder = model.model.encoder
print(f"Encoder has these attributes: {[attr for attr in dir(encoder) if not attr.startswith('_')]}")
print("\n" + "="*80)

# Test actual forward pass
text = "Hello world"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    # Get embeddings
    token_embeds = encoder.embed_tokens(inputs['input_ids'])
    positions = torch.arange(inputs['input_ids'].shape[1]).unsqueeze(0)
    pos_embeds = encoder.embed_positions(positions)
    x = token_embeds + pos_embeds
    
    print(f"After embeddings: {x[0, 0, :5]}")
    
    # Process through layers
    for i, layer in enumerate(encoder.layers):
        x_before = x.clone()
        x = layer(x, encoder_padding_mask=None, layer_head_mask=None)[0]  # layer returns (output, None)
        print(f"After layer {i}: {x[0, 0, :5]}")
        print(f"  Change from previous: {(x - x_before).abs().max():.6f}")
    
    # Check if there's any normalization after layers
    if hasattr(encoder, 'layer_norm') or hasattr(encoder, 'layernorm_embedding'):
        print(f"\nFinal normalization found!")
        if hasattr(encoder, 'layer_norm'):
            x = encoder.layer_norm(x)
            print(f"After final layer_norm: {x[0, 0, :5]}")
    else:
        print(f"\nNo final normalization found")
    
    # Compare with full forward
    full_output = encoder(inputs['input_ids']).last_hidden_state
    print(f"\nFull encoder output: {full_output[0, 0, :5]}")
    print(f"Manual matches full: {torch.allclose(x, full_output, atol=1e-5)}")
