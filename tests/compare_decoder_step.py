"""Compare JAX vs PyTorch decoder step by step."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import torch
import numpy as np
from transformers import FSMTForConditionalGeneration, T5Tokenizer

from hyperscalees.models.fsmt_loader import load_fsmt_model
from hyperscalees.models.fsmt.forward import FSMTModel

model_path = r"d:\HyperscaleES\user"

# Load models
config, params, _, _, _ = load_fsmt_model(model_path)
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_model.eval()
pt_tokenizer = T5Tokenizer.from_pretrained(model_path)

# Test input
text = "Hello, how are you?"
pt_inputs = pt_tokenizer(text, return_tensors='pt')
input_ids = jnp.array(pt_inputs['input_ids'].numpy())

print(f"Source: '{text}'")
print(f"Source IDs: {input_ids[0].tolist()}")

# 1. Compare encoder output
print("\n" + "="*60)
print("Step 1: Encoder output comparison")
print("="*60)

jax_encoder_out = FSMTModel.encode(input_ids, params, config)

with torch.no_grad():
    pt_encoder_out = pt_model.model.encoder(pt_inputs['input_ids']).last_hidden_state

enc_diff = np.abs(np.array(jax_encoder_out) - pt_encoder_out.numpy()).max()
print(f"Encoder max diff: {enc_diff:.6f}")
print(f"✅ Encoder MATCH!" if enc_diff < 1e-4 else "❌ Encoder MISMATCH!")

# 2. Compare decoder output at step 0 (only BOS token)
print("\n" + "="*60)
print("Step 2: Decoder output at step 0 (BOS only)")
print("="*60)

# Decoder input: [BOS] = [1]
decoder_input = jnp.array([[1]], dtype=jnp.int32)

jax_decoder_logits = FSMTModel.decode(
    decoder_input, jax_encoder_out, params, config
)

with torch.no_grad():
    pt_decoder_out = pt_model.model.decoder(
        input_ids=torch.tensor([[1]]),
        encoder_hidden_states=pt_encoder_out
    )
    pt_decoder_logits = pt_model.lm_head(pt_decoder_out.last_hidden_state)

print(f"JAX decoder logits shape: {jax_decoder_logits.shape}")
print(f"PT decoder logits shape: {pt_decoder_logits.shape}")

dec_diff = np.abs(np.array(jax_decoder_logits) - pt_decoder_logits.numpy()).max()
print(f"Decoder logits max diff: {dec_diff:.6f}")

# Check top tokens
jax_top5 = jnp.argsort(jax_decoder_logits[0, 0])[-5:][::-1]
pt_top5 = torch.argsort(pt_decoder_logits[0, 0])[-5:].flip(0)

print(f"\nJAX top-5 tokens: {jax_top5.tolist()}")
print(f"PT top-5 tokens: {pt_top5.tolist()}")

jax_next = int(jnp.argmax(jax_decoder_logits[0, -1]))
pt_next = int(torch.argmax(pt_decoder_logits[0, -1]))

print(f"\nJAX next token (greedy): {jax_next} -> '{pt_tokenizer.decode([jax_next])}'")
print(f"PT next token (greedy): {pt_next} -> '{pt_tokenizer.decode([pt_next])}'")

if dec_diff < 1e-4:
    print("✅ Decoder MATCH!")
else:
    print("❌ Decoder MISMATCH!")
    
    # Detailed comparison
    print("\n--- Detailed comparison ---")
    print(f"JAX logits[0,0,:10]: {jax_decoder_logits[0, 0, :10]}")
    print(f"PT logits[0,0,:10]: {pt_decoder_logits[0, 0, :10].numpy()}")
    
    # Check decoder hidden states
    with torch.no_grad():
        pt_dec_hidden = pt_decoder_out.last_hidden_state
    print(f"\nPT decoder hidden[0,0,:5]: {pt_dec_hidden[0, 0, :5].numpy()}")
    
    # Check JAX decoder hidden (before lm_head)
    # We need to trace through decode() to get hidden states
