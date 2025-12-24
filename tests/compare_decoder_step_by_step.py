"""Compare JAX decoder vs PyTorch decoder step-by-step."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp
import torch
import numpy as np
from transformers import FSMTForConditionalGeneration, T5Tokenizer

from hyperscalees.models.fsmt_loader import load_fsmt_model
from hyperscalees.models.fsmt.forward import FSMTModel

model_path = r"d:\HyperscaleES\user"

# Load models
config, params, _, _, tokenizer = load_fsmt_model(model_path)
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_model.eval()
pt_tokenizer = T5Tokenizer.from_pretrained(model_path)

# Test input
text = "Hello, how are you?"
pt_inputs = pt_tokenizer(text, return_tensors='pt')
input_ids = jnp.array(pt_inputs['input_ids'].numpy())

print(f"Source: '{text}'")
print(f"Input IDs: {input_ids[0].tolist()}")

# Encode once (we know this matches)
print("\n" + "="*80)
print("STEP 1: Encode")
print("="*80)

jax_encoder_output = FSMTModel.encode(input_ids, params, config, None, False)
print(f"JAX encoder output shape: {jax_encoder_output.shape}")

with torch.no_grad():
    pt_encoder_output = pt_model.model.encoder(pt_inputs['input_ids']).last_hidden_state
print(f"PyTorch encoder output shape: {pt_encoder_output.shape}")

max_diff = np.abs(np.array(jax_encoder_output) - pt_encoder_output.numpy()).max()
print(f"Max diff: {max_diff:.6f} {'✅' if max_diff < 1e-4 else '❌'}")

# Now test first decoder step
print("\n" + "="*80)
print("STEP 2: First Decoder Step (input=[1])")
print("="*80)

decoder_input_ids = jnp.array([[1]], dtype=jnp.int32)  # decoder_start_token_id
print(f"Decoder input: {decoder_input_ids[0].tolist()}")

# JAX decode
jax_logits = FSMTModel.decode(
    decoder_input_ids,
    jax_encoder_output,
    params,
    config,
    encoder_attention_mask=None,
    training=False
)
print(f"JAX logits shape: {jax_logits.shape}")
print(f"JAX logits[0, 0, :10]: {jax_logits[0, 0, :10]}")

# Get predicted token
jax_next_token = jnp.argmax(jax_logits[0, 0])
print(f"JAX predicted token: {jax_next_token} → '{tokenizer.decode([int(jax_next_token)])}'")

# PyTorch decode
with torch.no_grad():
    pt_decoder_output = pt_model.model.decoder(
        input_ids=torch.tensor([[1]]),
        encoder_hidden_states=pt_encoder_output
    )
    pt_logits = pt_model.lm_head(pt_decoder_output.last_hidden_state)
    
print(f"PyTorch logits shape: {pt_logits.shape}")
print(f"PyTorch logits[0, 0, :10]: {pt_logits[0, 0, :10].numpy()}")

pt_next_token = torch.argmax(pt_logits[0, 0])
print(f"PyTorch predicted token: {pt_next_token.item()} → '{pt_tokenizer.decode([pt_next_token.item()])}'")

# Compare logits
max_diff_logits = np.abs(np.array(jax_logits) - pt_logits.numpy()).max()
print(f"\nLogits max diff: {max_diff_logits:.6f} {'✅' if max_diff_logits < 1e-4 else '❌'}")

# Test second step if first matches
if jax_next_token == pt_next_token.item():
    print("\n" + "="*80)
    print(f"STEP 3: Second Decoder Step (input=[1, {jax_next_token}])")
    print("="*80)
    
    decoder_input_ids = jnp.array([[1, int(jax_next_token)]], dtype=jnp.int32)
    
    # JAX
    jax_logits = FSMTModel.decode(
        decoder_input_ids,
        jax_encoder_output,
        params,
        config,
        encoder_attention_mask=None,
        training=False
    )
    jax_next_token2 = jnp.argmax(jax_logits[0, 1])
    print(f"JAX predicted token: {jax_next_token2} → '{tokenizer.decode([int(jax_next_token2)])}'")
    
    # PyTorch
    with torch.no_grad():
        pt_decoder_output = pt_model.model.decoder(
            input_ids=torch.tensor([[1, int(jax_next_token)]]),
            encoder_hidden_states=pt_encoder_output
        )
        pt_logits = pt_model.lm_head(pt_decoder_output.last_hidden_state)
    
    pt_next_token2 = torch.argmax(pt_logits[0, 1])
    print(f"PyTorch predicted token: {pt_next_token2.item()} → '{pt_tokenizer.decode([pt_next_token2.item()])}'")
    
    max_diff_logits2 = np.abs(np.array(jax_logits) - pt_logits.numpy()).max()
    print(f"\nLogits max diff: {max_diff_logits2:.6f} {'✅' if max_diff_logits2 < 1e-4 else '❌'}")
else:
    print(f"\n❌ First token already differs! Skipping step 3.")
