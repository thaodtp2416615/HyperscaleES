"""Compare PyTorch full encoder vs manual loop."""
import torch
from transformers import FSMTForConditionalGeneration, T5Tokenizer

model_path = r"d:\HyperscaleES\user"
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_model.eval()
pt_tokenizer = T5Tokenizer.from_pretrained(model_path)

text = "Hello world"
inputs = pt_tokenizer(text, return_tensors='pt')

print("="*80)
print("Method 1: Full encoder.forward()")
print("="*80)

with torch.no_grad():
    full_output = pt_model.model.encoder(inputs['input_ids']).last_hidden_state

print(f"Output shape: {full_output.shape}")
print(f"Output[0,0,:5]: {full_output[0, 0, :5].numpy()}")

print("\n" + "="*80)
print("Method 2: Manual embeddings + loop through layers")
print("="*80)

with torch.no_grad():
    # Embeddings
    x = pt_model.model.encoder.embed_tokens(inputs['input_ids']) * pt_model.model.encoder.embed_scale
    # CRITICAL: PyTorch uses input_ids for position embeddings, not arange!
    pos_embeds = pt_model.model.encoder.embed_positions(inputs['input_ids'])
    x = x + pos_embeds
    
    print(f"After embeddings: {x[0, 0, :5].numpy()}")
    
    # Apply dropout?
    print(f"encoder.dropout: {pt_model.model.encoder.dropout}")
    print(f"encoder.training: {pt_model.model.encoder.training}")
    x = torch.nn.functional.dropout(x, p=pt_model.model.encoder.dropout, training=pt_model.model.encoder.training)
    print(f"After dropout: {x[0, 0, :5].numpy()}")
    
    # Transpose to [T, B, C]
    x = x.transpose(0, 1)
    
    # Loop through layers
    for i, layer in enumerate(pt_model.model.encoder.layers):
        x, _ = layer(x, encoder_padding_mask=None, layer_head_mask=None, output_attentions=False)
        if i == 0:
            print(f"After layer 0 (T,B,C format): {x[0, 0, :5].numpy()}")
    
    # Transpose back to [B, T, C]
    x = x.transpose(0, 1)
    
    print(f"Final output: {x[0, 0, :5].numpy()}")

print("\n" + "="*80)
print("Comparison")
print("="*80)

diff = torch.abs(full_output - x).max().item()
print(f"Max difference: {diff:.8f}")

if diff < 1e-5:
    print("✅ MATCH!")
else:
    print("❌ MISMATCH!")
    print(f"Full:   {full_output[0, 0, :5].numpy()}")
    print(f"Manual: {x[0, 0, :5].numpy()}")
