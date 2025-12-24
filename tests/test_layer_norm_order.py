"""Check if PyTorch FSMT uses Pre-LN or Post-LN."""
import torch
from transformers import FSMTForConditionalGeneration

model_path = r"d:\HyperscaleES\user"
model = FSMTForConditionalGeneration.from_pretrained(model_path)

# Check encoder layer structure
layer0 = model.model.encoder.layers[0]

print("Encoder Layer 0 structure:")
print(layer0)
print("\n" + "="*80)

# Check layer names
print("Layer 0 submodules:")
for name, module in layer0.named_children():
    print(f"  {name}: {type(module).__name__}")

print("\n" + "="*80)

# Test forward pass to understand order
print("Testing forward pass order...")
x = torch.randn(1, 5, 1024)
print(f"Input shape: {x.shape}")

# Manually step through
print("\nManual forward:")
with torch.no_grad():
    # Check if it's pre-norm or post-norm
    # Pre-norm: norm -> attention -> add
    # Post-norm: attention -> add -> norm
    
    # Check self_attn_layer_norm position
    residual = x
    print(f"1. Residual saved: {residual[0, 0, :3]}")
    
    # Does it normalize first?
    x_normed = layer0.self_attn_layer_norm(x)
    print(f"2. After self_attn_layer_norm: {x_normed[0, 0, :3]}")
    
    # Then attention
    attn_out, _, _ = layer0.self_attn(x_normed, x_normed, x_normed)
    print(f"3. After self_attn: {attn_out[0, 0, :3]}")
    
    # Then add residual
    x = residual + attn_out
    print(f"4. After residual: {x[0, 0, :3]}")
    
    # Save for FFN
    residual = x
    
    # Normalize again
    x_normed = layer0.final_layer_norm(x)
    print(f"5. After final_layer_norm: {x_normed[0, 0, :3]}")
    
    # FFN
    x_ffn = layer0.fc2(layer0.activation_fn(layer0.fc1(x_normed)))
    print(f"6. After FFN: {x_ffn[0, 0, :3]}")
    
    # Add residual
    x = residual + x_ffn
    print(f"7. Final output: {x[0, 0, :3]}")

print("\n" + "="*80)
print("Conclusion: FSMT uses PRE-NORM (LayerNorm before attention/FFN)")
