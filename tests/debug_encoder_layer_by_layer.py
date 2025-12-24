"""Debug JAX encoder layer by layer."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import torch
import numpy as np
from transformers import FSMTForConditionalGeneration, T5Tokenizer

from hyperscalees.models.fsmt_loader import load_fsmt_model
from hyperscalees.models.fsmt.forward import FSMTModel

def compare_arrays(jax_arr, torch_arr, name):
    """Compare JAX and PyTorch arrays."""
    torch_np = torch_arr.detach().cpu().numpy()
    jax_np = np.array(jax_arr)
    
    max_diff = np.abs(jax_np - torch_np).max()
    mean_diff = np.abs(jax_np - torch_np).mean()
    
    match = max_diff < 1e-4
    symbol = "✅" if match else "❌"
    print(f"{symbol} {name}")
    print(f"   Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    if not match:
        print(f"   JAX[:5]: {jax_np.flatten()[:5]}")
        print(f"   PyTorch[:5]: {torch_np.flatten()[:5]}")
    return match

def main():
    model_path = r"d:\HyperscaleES\user"
    
    print("="*80)
    print("Loading models...")
    print("="*80)
    
    # Load JAX model
    config, params, scan_map, es_map, tokenizer = load_fsmt_model(model_path)
    
    # Load PyTorch model
    pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
    pt_model.eval()
    pt_tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Test input
    text = "Hello world"
    print(f"\nInput: '{text}'")
    
    # Tokenize
    pt_inputs = pt_tokenizer(text, return_tensors='pt')
    input_ids = jnp.array(pt_inputs['input_ids'].numpy())
    print(f"Token IDs: {input_ids}")
    
    batch_size, seq_len = input_ids.shape
    
    # Get initial embeddings (SCALED)
    embed_weight = params['encoder']['embed_tokens']['weight']
    pos_weight = params['encoder']['embed_positions']['weight']
    
    jax_token_embeds = embed_weight[input_ids]
    if config.scale_embedding:
        embed_scale = jnp.sqrt(float(config.d_model))
        jax_token_embeds = jax_token_embeds * embed_scale
    
    positions = jnp.arange(seq_len)[None, :]
    jax_pos_embeds = pos_weight[positions]
    jax_x = jax_token_embeds + jax_pos_embeds
    
    # PyTorch initial embeddings
    with torch.no_grad():
        pt_x = pt_model.model.encoder.embed_tokens(pt_inputs['input_ids']) * pt_model.model.encoder.embed_scale
        positions_pt = torch.arange(seq_len).unsqueeze(0)
        pt_pos_embeds = pt_model.model.encoder.embed_positions(positions_pt)
        pt_x = pt_x + pt_pos_embeds
    
    print("\n" + "="*80)
    print("Initial embeddings (scaled)")
    print("="*80)
    compare_arrays(jax_x, pt_x, "Scaled embeddings")
    
    # PyTorch encoder transposes to [T, B, C] before layers
    print("\nNote: PyTorch encoder transposes to [T, B, C] for layers")
    pt_x = pt_x.transpose(0, 1)  # [B, T, C] -> [T, B, C]
    print(f"PyTorch shape after transpose: {pt_x.shape} (should be [T, B, C])")
    
    # Process layer by layer
    for i in range(config.encoder_layers):
        print(f"\n" + "="*80)
        print(f"Layer {i}")
        print("="*80)
        
        # JAX layer
        jax_x = FSMTModel.encoder_layer(
            jax_x,
            params['encoder']['layers'][str(i)],
            config,
            mask=None,
            training=False,
            rng=None
        )
        
        # PyTorch layer
        with torch.no_grad():
            pt_x, _ = pt_model.model.encoder.layers[i](
                pt_x,
                encoder_padding_mask=None,
                layer_head_mask=None,
                output_attentions=False
            )
        
        compare_arrays(jax_x, pt_x, f"After layer {i}")
        
        if i == 0:
            print(f"   JAX after layer 0: {jax_x[0, 0, :5]}")
            print(f"   PyTorch after layer 0: {pt_x[0, 0, :5].numpy()}")
    
    print("\n" + "="*80)
    print("Final comparison")
    print("="*80)
    
    # Transpose PyTorch back to [B, T, C]
    pt_x = pt_x.transpose(0, 1)  # [T, B, C] -> [B, T, C]
    compare_arrays(jax_x, pt_x, "Manual layer-by-layer")
    
    # Full encoder (JAX)
    jax_full = FSMTModel.encode(config, input_ids, params)
    
    # Full encoder (PyTorch)
    with torch.no_grad():
        pt_full = pt_model.model.encoder(pt_inputs['input_ids']).last_hidden_state
    
    compare_arrays(jax_full, pt_full, "Full encoder (via encode())")
    
    print("\n" + "="*80)
    print("Debugging Complete")
    print("="*80)

if __name__ == "__main__":
    main()
