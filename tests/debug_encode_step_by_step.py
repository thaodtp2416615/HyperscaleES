"""Debug encode() function step by step with detailed comparison."""
import sys
import os
from pathlib import Path

# Force module reload
for mod in list(sys.modules.keys()):
    if 'hyperscalees' in mod:
        del sys.modules[mod]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import torch
import numpy as np
from transformers import FSMTForConditionalGeneration, T5Tokenizer
import inspect

from hyperscalees.models.fsmt_loader import load_fsmt_model
from hyperscalees.models.fsmt.forward import FSMTModel

def compare(jax_arr, torch_arr, name):
    torch_np = torch_arr.detach().cpu().numpy() if torch.is_tensor(torch_arr) else torch_arr
    jax_np = np.array(jax_arr)
    max_diff = np.abs(jax_np - torch_np).max()
    mean_diff = np.abs(jax_np - torch_np).mean()
    match = max_diff < 1e-4
    symbol = "✅" if match else "❌"
    print(f"{symbol} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    if not match:
        print(f"   JAX[:5]:     {jax_np.flatten()[:5]}")
        print(f"   PyTorch[:5]: {torch_np.flatten()[:5]}")
    return match

model_path = r"d:\HyperscaleES\user"

print("="*80)
print("Loading models...")
print("="*80)

config, params, scan_map, es_map, tokenizer = load_fsmt_model(model_path, verbose=False)
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_model.eval()
pt_tokenizer = T5Tokenizer.from_pretrained(model_path)

# Check encode() source
print("\n" + "="*80)
print("Checking encode() source code...")
print("="*80)
encode_source = inspect.getsource(FSMTModel.encode)
# Check for the fix
if "jnp.arange(seq_len)[None, :]" in encode_source:
    print("✅ encode() has the position embedding fix (advanced indexing)")
else:
    print("❌ encode() still uses old method (direct slicing)")
    print("First 500 chars of encode():")
    print(encode_source[:500])

# Test input
text = "Hello world"
pt_inputs = pt_tokenizer(text, return_tensors='pt')
input_ids = jnp.array(pt_inputs['input_ids'].numpy())

print(f"\nInput: '{text}'")
print(f"Token IDs: {input_ids}")

batch_size, seq_len = input_ids.shape

# ============================================================================
# STEP 1: Manually replicate encode() embeddings
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Manual embeddings (replicating encode() logic)")
print("="*80)

embed_weight = params['encoder']['embed_tokens']['weight']
pos_weight = params['encoder']['embed_positions']['weight']

# Token embeddings
jax_token_embeds = embed_weight[input_ids]
print(f"Token embeds shape: {jax_token_embeds.shape}")
print(f"Token embeds[0,0,:5]: {jax_token_embeds[0, 0, :5]}")

# Scale
if config.scale_embedding:
    embed_scale = jnp.sqrt(float(config.d_model))
    jax_token_embeds = jax_token_embeds * embed_scale
    print(f"Scaled by {embed_scale}")
    print(f"Scaled token embeds[0,0,:5]: {jax_token_embeds[0, 0, :5]}")

# Position embeddings (CORRECT METHOD)
positions = jnp.arange(seq_len)[None, :]  # [1, seq]
jax_pos_embeds = pos_weight[positions]  # [1, seq, d_model]
print(f"Position embeds shape: {jax_pos_embeds.shape}")
print(f"Position embeds[0,0,:5]: {jax_pos_embeds[0, 0, :5]}")

# Add
jax_embeds_manual = jax_token_embeds + jax_pos_embeds
print(f"Combined embeds shape: {jax_embeds_manual.shape}")
print(f"Combined embeds[0,0,:5]: {jax_embeds_manual[0, 0, :5]}")

# Compare with PyTorch
with torch.no_grad():
    pt_token_embeds = pt_model.model.encoder.embed_tokens(pt_inputs['input_ids']) * pt_model.model.encoder.embed_scale
    positions_pt = torch.arange(seq_len).unsqueeze(0)
    pt_pos_embeds = pt_model.model.encoder.embed_positions(positions_pt)
    pt_embeds = pt_token_embeds + pt_pos_embeds

compare(jax_embeds_manual, pt_embeds, "Manual embeddings vs PyTorch")

# ============================================================================
# STEP 2: Process layer 0 manually
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Process layer 0 manually")
print("="*80)

jax_x = jax_embeds_manual
jax_x_layer0 = FSMTModel.encoder_layer(
    jax_x,
    params['encoder']['layers']['0'],
    config,
    mask=None,
    training=False,
    rng=None
)
print(f"After layer 0: {jax_x_layer0[0, 0, :5]}")

# PyTorch layer 0
pt_x = pt_embeds.transpose(0, 1)  # [B, T, C] -> [T, B, C]
with torch.no_grad():
    pt_x_layer0, _ = pt_model.model.encoder.layers[0](
        pt_x,
        encoder_padding_mask=None,
        layer_head_mask=None,
        output_attentions=False
    )
pt_x_layer0 = pt_x_layer0.transpose(0, 1)  # [T, B, C] -> [B, T, C]

compare(jax_x_layer0, pt_x_layer0, "Layer 0 output")

# ============================================================================
# STEP 3: Call FSMTModel.encode()
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Call FSMTModel.encode() and compare")
print("="*80)

jax_output = FSMTModel.encode(
    input_ids=input_ids,
    params=params,
    config=config,
    attention_mask=None,
    training=False,
    rng=None
)
print(f"JAX encode() output shape: {jax_output.shape}")
print(f"JAX encode() output[0,0,:5]: {jax_output[0, 0, :5]}")

# PyTorch full encoder
with torch.no_grad():
    pt_output = pt_model.model.encoder(pt_inputs['input_ids']).last_hidden_state

print(f"PyTorch output shape: {pt_output.shape}")
print(f"PyTorch output[0,0,:5]: {pt_output[0, 0, :5].numpy()}")

compare(jax_output, pt_output, "Full encode() vs PyTorch")

print("\n" + "="*80)
print("Summary")
print("="*80)
print("If manual embeddings MATCH but encode() output DOESN'T MATCH,")
print("then the issue is in how encode() processes embeddings or layers.")
