"""Debug embeddings in encode() step by step."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp
import numpy as np
from transformers import FSMTForConditionalGeneration, T5Tokenizer

from hyperscalees.models.fsmt_loader import load_fsmt_model

model_path = r"d:\HyperscaleES\user"

# Load models
config, params, scan_map, es_map, tokenizer = load_fsmt_model(model_path)
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_model.eval()
pt_tokenizer = T5Tokenizer.from_pretrained(model_path)

# Test input
text = "Hello world"
pt_inputs = pt_tokenizer(text, return_tensors='pt')
input_ids = jnp.array(pt_inputs['input_ids'].numpy())

print(f"Input: '{text}'")
print(f"Token IDs: {input_ids}")
print(f"Shape: {input_ids.shape}")

batch_size, seq_len = input_ids.shape
print(f"batch_size={batch_size}, seq_len={seq_len}")

# Step 1: Token embeddings
print("\n" + "="*80)
print("STEP 1: Token embeddings")
print("="*80)

embed_weight = params['encoder']['embed_tokens']['weight']
print(f"embed_weight shape: {embed_weight.shape}")

jax_token_embeds = embed_weight[input_ids]
print(f"Token embeds shape: {jax_token_embeds.shape}")
print(f"Token embeds[0, 0, :5]: {jax_token_embeds[0, 0, :5]}")

# Step 2: Scale
print("\n" + "="*80)
print("STEP 2: Scale embeddings")
print("="*80)

if config.scale_embedding:
    embed_scale = jnp.sqrt(float(config.d_model))
    jax_token_embeds_scaled = jax_token_embeds * embed_scale
    print(f"Scale: {embed_scale}")
    print(f"Scaled embeds[0, 0, :5]: {jax_token_embeds_scaled[0, 0, :5]}")
else:
    jax_token_embeds_scaled = jax_token_embeds
    print("No scaling")

# Step 3: Position embeddings
print("\n" + "="*80)
print("STEP 3: Position embeddings")
print("="*80)

pos_weight = params['encoder']['embed_positions']['weight']
print(f"pos_weight shape: {pos_weight.shape}")

# Method 1: Direct slicing (WRONG)
pos_embeds_method1 = pos_weight[:seq_len]
print(f"Method 1 (direct slice) shape: {pos_embeds_method1.shape}")
print(f"Method 1[0, :5]: {pos_embeds_method1[0, :5]}")

# Method 2: Advanced indexing (CORRECT)
positions = jnp.arange(seq_len)[None, :]
pos_embeds_method2 = pos_weight[positions]
print(f"Method 2 (advanced indexing) shape: {pos_embeds_method2.shape}")
print(f"Method 2[0, 0, :5]: {pos_embeds_method2[0, 0, :5]}")

# Step 4: Add
print("\n" + "="*80)
print("STEP 4: Add token + position")
print("="*80)

# Method 1
try:
    x_method1 = jax_token_embeds_scaled + pos_embeds_method1
    print(f"Method 1 result shape: {x_method1.shape}")
    print(f"Method 1[0, 0, :5]: {x_method1[0, 0, :5]}")
except Exception as e:
    print(f"Method 1 error: {e}")

# Method 2
x_method2 = jax_token_embeds_scaled + pos_embeds_method2
print(f"Method 2 result shape: {x_method2.shape}")
print(f"Method 2[0, 0, :5]: {x_method2[0, 0, :5]}")

# Compare with PyTorch
print("\n" + "="*80)
print("Compare with PyTorch")
print("="*80)

import torch
with torch.no_grad():
    pt_token_embeds = pt_model.model.encoder.embed_tokens(pt_inputs['input_ids']) * pt_model.model.encoder.embed_scale
    positions_pt = torch.arange(seq_len).unsqueeze(0)
    pt_pos_embeds = pt_model.model.encoder.embed_positions(positions_pt)
    pt_embeds = pt_token_embeds + pt_pos_embeds

print(f"PyTorch embeds shape: {pt_embeds.shape}")
print(f"PyTorch embeds[0, 0, :5]: {pt_embeds[0, 0, :5].numpy()}")

# Compare
diff1 = np.abs(np.array(x_method1) - pt_embeds.numpy()).max()
diff2 = np.abs(np.array(x_method2) - pt_embeds.numpy()).max()

print(f"\nMethod 1 max diff: {diff1:.8f}")
print(f"Method 2 max diff: {diff2:.8f}")

if diff2 < 1e-6:
    print("✅ Method 2 (advanced indexing) MATCHES PyTorch!")
else:
    print("❌ Still mismatch")
