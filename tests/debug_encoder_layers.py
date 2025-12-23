"""Debug encoder layer by layer to find where outputs diverge."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import torch
import numpy as np
from transformers import FSMTForConditionalGeneration, T5Tokenizer

from src.hyperscalees.models.fsmt_loader import load_fsmt_model
from src.hyperscalees.models.fsmt.forward import FSMTModel

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

def manual_layer_norm(x, gamma, beta, eps=1e-5):
    """Manual layer norm implementation."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_norm + beta

def manual_attention(query, key, value, attention_mask=None):
    """Manual multi-head attention."""
    # query, key, value: [batch, seq_len, d_model]
    d_k = query.shape[-1]
    
    # Compute attention scores
    scores = jnp.matmul(query, key.transpose(0, 2, 1)) / jnp.sqrt(d_k)
    
    if attention_mask is not None:
        scores = scores + attention_mask
    
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, value)
    
    return output

def main():
    model_path = r"d:\HyperscaleES\user"
    
    print("="*80)
    print("Loading models...")
    print("="*80)
    
    # Load JAX model
    config, params, scan_map, es_map, tokenizer = load_fsmt_model(model_path)
    jax_model = FSMTModel(config)
    
    # Load PyTorch model
    pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
    pt_model.eval()
    pt_tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Test input
    text = "Hello world"
    print(f"\nInput: '{text}'")
    
    # Tokenize
    jax_inputs = tokenizer.encode(text, return_tensors='jax')
    pt_inputs = pt_tokenizer(text, return_tensors='pt')
    
    input_ids = jnp.array(pt_inputs['input_ids'].numpy())
    print(f"Token IDs: {input_ids}")
    
    batch_size, seq_len = input_ids.shape
    
    print("\n" + "="*80)
    print("STEP 1: Check embeddings")
    print("="*80)
    
    # JAX embeddings
    embed_weight = params['encoder']['embed_tokens']['weight']
    pos_weight = params['encoder']['embed_positions']['weight']
    
    jax_token_embeds = embed_weight[input_ids]  # [batch, seq, d_model]
    positions = jnp.arange(seq_len)[None, :]  # [1, seq]
    jax_pos_embeds = pos_weight[positions]  # [1, seq, d_model]
    jax_embeds = jax_token_embeds + jax_pos_embeds
    
    # PyTorch embeddings
    with torch.no_grad():
        pt_token_embeds = pt_model.model.encoder.embed_tokens(pt_inputs['input_ids'])
        positions_pt = torch.arange(seq_len).unsqueeze(0)
        pt_pos_embeds = pt_model.model.encoder.embed_positions(positions_pt)
        pt_embeds = pt_token_embeds + pt_pos_embeds
    
    compare_arrays(jax_embeds, pt_embeds, "Token + Position Embeddings")
    
    print("\n" + "="*80)
    print("STEP 2: Check LayerNorm (first layer)")
    print("="*80)
    
    # JAX layer norm
    ln_gamma = params['encoder']['layers']['0']['self_attn_layer_norm']['weight']
    ln_beta = params['encoder']['layers']['0']['self_attn_layer_norm']['bias']
    
    jax_ln_out = manual_layer_norm(jax_embeds, ln_gamma, ln_beta, eps=config.layer_norm_eps)
    
    # PyTorch layer norm
    with torch.no_grad():
        pt_ln_out = pt_model.model.encoder.layers[0].self_attn_layer_norm(pt_embeds)
    
    compare_arrays(jax_ln_out, pt_ln_out, "Layer Norm (before attention)")
    
    print("\n" + "="*80)
    print("STEP 3: Check attention projection weights")
    print("="*80)
    
    # Check if Q, K, V projection weights match
    layer_0 = params['encoder']['layers']['0']
    
    for name in ['q_proj', 'k_proj', 'v_proj']:
        jax_weight = layer_0['self_attn'][name]['weight']
        pt_weight = getattr(pt_model.model.encoder.layers[0].self_attn, name).weight
        compare_arrays(jax_weight, pt_weight, f"Attention {name} weight")
    
    print("\n" + "="*80)
    print("STEP 4: Check Q, K, V projections")
    print("="*80)
    
    # JAX projections
    q_weight = layer_0['self_attn']['q_proj']['weight']
    k_weight = layer_0['self_attn']['k_proj']['weight']
    v_weight = layer_0['self_attn']['v_proj']['weight']
    q_bias = layer_0['self_attn']['q_proj']['bias']
    k_bias = layer_0['self_attn']['k_proj']['bias']
    v_bias = layer_0['self_attn']['v_proj']['bias']
    
    jax_q = jnp.matmul(jax_ln_out, q_weight.T) + q_bias
    jax_k = jnp.matmul(jax_ln_out, k_weight.T) + k_bias
    jax_v = jnp.matmul(jax_ln_out, v_weight.T) + v_bias
    
    # PyTorch projections
    with torch.no_grad():
        pt_q = pt_model.model.encoder.layers[0].self_attn.q_proj(pt_ln_out)
        pt_k = pt_model.model.encoder.layers[0].self_attn.k_proj(pt_ln_out)
        pt_v = pt_model.model.encoder.layers[0].self_attn.v_proj(pt_ln_out)
    
    compare_arrays(jax_q, pt_q, "Query projection")
    compare_arrays(jax_k, pt_k, "Key projection")
    compare_arrays(jax_v, pt_v, "Value projection")
    
    print("\n" + "="*80)
    print("STEP 5: Check multi-head split")
    print("="*80)
    
    num_heads = config.encoder_attention_heads
    head_dim = config.d_model // num_heads
    
    # JAX reshape
    jax_q_heads = jax_q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    jax_k_heads = jax_k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    jax_v_heads = jax_v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    # PyTorch reshape
    pt_q_heads = pt_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    pt_k_heads = pt_k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    pt_v_heads = pt_v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    compare_arrays(jax_q_heads, pt_q_heads, "Query heads [batch, heads, seq, head_dim]")
    compare_arrays(jax_k_heads, pt_k_heads, "Key heads [batch, heads, seq, head_dim]")
    compare_arrays(jax_v_heads, pt_v_heads, "Value heads [batch, heads, seq, head_dim]")
    
    print("\n" + "="*80)
    print("STEP 6: Check attention scores")
    print("="*80)
    
    # JAX attention scores
    jax_scores = jnp.matmul(jax_q_heads, jax_k_heads.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
    
    # PyTorch attention scores
    pt_scores = torch.matmul(pt_q_heads, pt_k_heads.transpose(-2, -1)) / np.sqrt(head_dim)
    
    compare_arrays(jax_scores, pt_scores, "Attention scores [batch, heads, seq, seq]")
    
    print("\n" + "="*80)
    print("STEP 7: Check attention weights (after softmax)")
    print("="*80)
    
    # JAX attention weights
    jax_attn_weights = jax.nn.softmax(jax_scores, axis=-1)
    
    # PyTorch attention weights
    pt_attn_weights = torch.nn.functional.softmax(pt_scores, dim=-1)
    
    compare_arrays(jax_attn_weights, pt_attn_weights, "Attention weights [batch, heads, seq, seq]")
    
    print("\n" + "="*80)
    print("STEP 8: Check attention output")
    print("="*80)
    
    # JAX attention output
    jax_attn_out = jnp.matmul(jax_attn_weights, jax_v_heads)
    jax_attn_out = jax_attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, config.d_model)
    
    # PyTorch attention output
    pt_attn_out = torch.matmul(pt_attn_weights, pt_v_heads)
    pt_attn_out = pt_attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, config.d_model)
    
    compare_arrays(jax_attn_out, pt_attn_out, "Attention output (before out_proj)")
    
    print("\n" + "="*80)
    print("STEP 9: Check output projection")
    print("="*80)
    
    # JAX output projection
    out_weight = layer_0['self_attn']['out_proj']['weight']
    out_bias = layer_0['self_attn']['out_proj']['bias']
    jax_out = jnp.matmul(jax_attn_out, out_weight.T) + out_bias
    
    # PyTorch output projection
    with torch.no_grad():
        pt_out = pt_model.model.encoder.layers[0].self_attn.out_proj(pt_attn_out)
    
    compare_arrays(jax_out, pt_out, "Attention output projection")
    
    print("\n" + "="*80)
    print("STEP 10: Check residual + layer norm")
    print("="*80)
    
    # JAX
    jax_residual = jax_embeds + jax_out
    
    # PyTorch
    pt_residual = pt_embeds + pt_out
    
    compare_arrays(jax_residual, pt_residual, "After residual connection")
    
    # Layer norm
    fc_ln_gamma = params['encoder']['layers']['0']['final_layer_norm']['weight']
    fc_ln_beta = params['encoder']['layers']['0']['final_layer_norm']['bias']
    jax_ln2 = manual_layer_norm(jax_residual, fc_ln_gamma, fc_ln_beta, eps=config.layer_norm_eps)
    
    with torch.no_grad():
        pt_ln2 = pt_model.model.encoder.layers[0].final_layer_norm(pt_residual)
    
    compare_arrays(jax_ln2, pt_ln2, "Layer norm (before FFN)")
    
    print("\n" + "="*80)
    print("STEP 11: Check FFN (Feed-Forward Network)")
    print("="*80)
    
    # JAX FFN
    fc1_weight = layer_0['fc1']['weight']
    fc1_bias = layer_0['fc1']['bias']
    fc2_weight = layer_0['fc2']['weight']
    fc2_bias = layer_0['fc2']['bias']
    
    jax_fc1 = jnp.matmul(jax_ln2, fc1_weight.T) + fc1_bias
    jax_fc1_act = jax.nn.gelu(jax_fc1)
    jax_fc2 = jnp.matmul(jax_fc1_act, fc2_weight.T) + fc2_bias
    
    # PyTorch FFN
    with torch.no_grad():
        pt_fc1 = pt_model.model.encoder.layers[0].fc1(pt_ln2)
        pt_fc1_act = torch.nn.functional.gelu(pt_fc1)
        pt_fc2 = pt_model.model.encoder.layers[0].fc2(pt_fc1_act)
    
    compare_arrays(jax_fc1, pt_fc1, "FC1 (before activation)")
    compare_arrays(jax_fc1_act, pt_fc1_act, "FC1 (after GELU)")
    compare_arrays(jax_fc2, pt_fc2, "FC2 output")
    
    print("\n" + "="*80)
    print("STEP 12: Check final residual (complete layer 0)")
    print("="*80)
    
    # JAX
    jax_layer0_out = jax_residual + jax_fc2
    
    # PyTorch
    pt_layer0_out = pt_residual + pt_fc2
    
    compare_arrays(jax_layer0_out, pt_layer0_out, "Complete Layer 0 output")
    
    print("\n" + "="*80)
    print("STEP 13: Run full encoder and compare")
    print("="*80)
    
    # JAX full encoder
    jax_encoder_out = jax_model.encode(input_ids, params)
    
    # PyTorch full encoder
    with torch.no_grad():
        pt_encoder_out = pt_model.model.encoder(pt_inputs['input_ids']).last_hidden_state
    
    compare_arrays(jax_encoder_out, pt_encoder_out, "Full Encoder output (all 8 layers)")
    
    print("\n" + "="*80)
    print("Debugging Complete")
    print("="*80)

if __name__ == "__main__":
    main()
