"""Test FSMTModel.encode() directly and debug."""
import sys
import os
from pathlib import Path

# Force fresh import
if 'hyperscalees.models.fsmt.forward' in sys.modules:
    del sys.modules['hyperscalees.models.fsmt.forward']

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp
import torch
import numpy as np
from transformers import FSMTForConditionalGeneration, T5Tokenizer

from hyperscalees.models.fsmt_loader import load_fsmt_model
from hyperscalees.models.fsmt.forward import FSMTModel

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
print(f"Config type: {type(config)}")
print(f"Config d_model: {config.d_model}")
print(f"Config scale_embedding: {config.scale_embedding}")
print(f"Params keys: {list(params.keys())}")
print(f"Encoder params keys: {list(params['encoder'].keys())}")

# Call encode
print("\n" + "="*80)
print("Calling FSMTModel.encode()...")
print("="*80)

# First, manually check what embeddings encode() produces
embed_weight = params['encoder']['embed_tokens']['weight']
pos_weight = params['encoder']['embed_positions']['weight']
seq_len = input_ids.shape[1]

manual_token_embeds = embed_weight[input_ids]
if config.scale_embedding:
    embed_scale = jnp.sqrt(float(config.d_model))
    manual_token_embeds = manual_token_embeds * embed_scale
    print(f"Embedding scale: {embed_scale}")

# CRITICAL: Use input_ids for position embeddings (like PyTorch)
manual_pos_embeds = pos_weight[input_ids]
manual_embeds = manual_token_embeds + manual_pos_embeds

print(f"Manual embeddings (first token, first 5): {manual_embeds[0, 0, :5]}")

jax_output = FSMTModel.encode(
    input_ids=input_ids,
    params=params,
    config=config,
    attention_mask=None,
    training=False,
    rng=None
)

print(f"JAX output shape: {jax_output.shape}")
print(f"JAX output[0, 0, :5]: {jax_output[0, 0, :5]}")

# PyTorch output
with torch.no_grad():
    pt_output = pt_model.model.encoder(pt_inputs['input_ids']).last_hidden_state

print(f"\nPyTorch output shape: {pt_output.shape}")
print(f"PyTorch output[0, 0, :5]: {pt_output[0, 0, :5].numpy()}")

# Compare
max_diff = np.abs(np.array(jax_output) - pt_output.numpy()).max()
print(f"\nMax difference: {max_diff:.6f}")

if max_diff < 1e-4:
    print("✅ MATCH!")
else:
    print("❌ MISMATCH!")
