"""
Debug script to compare Transformers vs JAX FSMT outputs

This compares outputs at each stage to find where divergence happens.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import jax.numpy as jnp

from hyperscalees.models.fsmt_loader import load_fsmt_model
from hyperscalees.models.fsmt import FSMTModel, generate_translation

# Try to import transformers
try:
    from transformers import FSMTForConditionalGeneration, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not available")


def compare_embeddings():
    """Compare embedding layer outputs"""
    print("\n" + "="*80)
    print("STEP 1: Compare Embeddings")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    # Load JAX model
    config, jax_params, _, _, tokenizer = load_fsmt_model(model_path, verbose=False)
    
    # Load PyTorch model
    if HAS_TRANSFORMERS:
        torch_model = FSMTForConditionalGeneration.from_pretrained(model_path)
        torch_model.eval()
    
    # Test input
    test_text = "Hello world"
    input_ids = tokenizer.encode(test_text, return_tensors='np')
    
    print(f"Input: '{test_text}'")
    print(f"Token IDs: {input_ids[0, :10]}")
    
    # JAX embeddings
    jax_embeddings = jax_params['encoder']['embed_tokens']['weight'][input_ids[0]]
    print(f"\nJAX embeddings shape: {jax_embeddings.shape}")
    print(f"JAX embeddings[0, :5]: {jax_embeddings[0, :5]}")
    
    if HAS_TRANSFORMERS:
        # PyTorch embeddings
        with torch.no_grad():
            torch_input = torch.tensor(input_ids)
            torch_embeddings = torch_model.model.encoder.embed_tokens(torch_input)
            torch_embeddings = torch_embeddings.cpu().numpy()
        
        print(f"\nPyTorch embeddings shape: {torch_embeddings.shape}")
        print(f"PyTorch embeddings[0, 0, :5]: {torch_embeddings[0, 0, :5]}")
        
        # Compare
        diff = np.abs(jax_embeddings - torch_embeddings[0])
        print(f"\nMax difference: {np.max(diff):.6f}")
        print(f"Mean difference: {np.mean(diff):.6f}")
        
        if np.max(diff) < 1e-5:
            print("✅ Embeddings MATCH!")
        else:
            print("❌ Embeddings DIFFER!")
            return False
    
    return True


def compare_encoder():
    """Compare encoder outputs"""
    print("\n" + "="*80)
    print("STEP 2: Compare Encoder Output")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    # Load models
    config, jax_params, _, _, tokenizer = load_fsmt_model(model_path, verbose=False)
    
    if not HAS_TRANSFORMERS:
        print("⚠️  Skipping - transformers not available")
        return True
    
    torch_model = FSMTForConditionalGeneration.from_pretrained(model_path)
    torch_model.eval()
    
    # Test input
    test_text = "Hello world"
    input_ids = tokenizer.encode(test_text, return_tensors='np')
    
    # JAX encoder
    jax_input = jnp.array(input_ids)
    jax_encoder_output = FSMTModel.encode(
        jax_input, jax_params, config, training=False
    )
    
    print(f"JAX encoder output shape: {jax_encoder_output.shape}")
    print(f"JAX encoder output[0, 0, :5]: {jax_encoder_output[0, 0, :5]}")
    
    # PyTorch encoder
    with torch.no_grad():
        torch_input = torch.tensor(input_ids)
        torch_encoder_output = torch_model.model.encoder(torch_input)
        torch_encoder_output = torch_encoder_output.last_hidden_state.cpu().numpy()
    
    print(f"\nPyTorch encoder output shape: {torch_encoder_output.shape}")
    print(f"PyTorch encoder output[0, 0, :5]: {torch_encoder_output[0, 0, :5]}")
    
    # Compare
    diff = np.abs(np.array(jax_encoder_output) - torch_encoder_output)
    print(f"\nMax difference: {np.max(diff):.6f}")
    print(f"Mean difference: {np.mean(diff):.6f}")
    
    if np.max(diff) < 1e-4:
        print("✅ Encoder outputs MATCH!")
        return True
    else:
        print("❌ Encoder outputs DIFFER!")
        return False


def compare_decoder_step():
    """Compare single decoder step"""
    print("\n" + "="*80)
    print("STEP 3: Compare Decoder (Single Step)")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    # Load models
    config, jax_params, _, _, tokenizer = load_fsmt_model(model_path, verbose=False)
    
    if not HAS_TRANSFORMERS:
        print("⚠️  Skipping - transformers not available")
        return True
    
    torch_model = FSMTForConditionalGeneration.from_pretrained(model_path)
    torch_model.eval()
    
    # Test input
    src_text = "Hello world"
    input_ids = tokenizer.encode(src_text, return_tensors='np')
    
    # First get encoder output
    jax_input = jnp.array(input_ids)
    jax_encoder_output = FSMTModel.encode(
        jax_input, jax_params, config, training=False
    )
    
    with torch.no_grad():
        torch_input = torch.tensor(input_ids)
        torch_encoder_output = torch_model.model.encoder(torch_input).last_hidden_state
    
    # Decoder input (just start token)
    decoder_input_ids = np.array([[config.decoder_start_token_id]])
    
    # JAX decoder
    jax_decoder_input = jnp.array(decoder_input_ids)
    jax_logits = FSMTModel.decode(
        jax_decoder_input,
        jax_encoder_output,
        jax_params,
        config,
        training=False
    )
    
    print(f"JAX decoder logits shape: {jax_logits.shape}")
    print(f"JAX logits[0, 0, :10]: {jax_logits[0, 0, :10]}")
    print(f"JAX predicted token: {np.argmax(jax_logits[0, 0])}")
    
    # PyTorch decoder - use generate-like approach
    with torch.no_grad():
        torch_decoder_input = torch.tensor(decoder_input_ids)
        
        # Create masks
        src_len = torch_encoder_output.shape[1]
        encoder_padding_mask = torch.zeros(1, src_len, dtype=torch.bool)
        decoder_padding_mask = torch.zeros(1, 1, dtype=torch.bool)
        
        # Create causal mask
        tgt_len = 1
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
        
        torch_decoder_output = torch_model.model.decoder(
            torch_decoder_input,
            encoder_hidden_states=torch_encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=causal_mask
        )
        torch_logits = torch_model.lm_head(torch_decoder_output[0])
        torch_logits = torch_logits.cpu().numpy()
    
    print(f"\nPyTorch decoder logits shape: {torch_logits.shape}")
    print(f"PyTorch logits[0, 0, :10]: {torch_logits[0, 0, :10]}")
    print(f"PyTorch predicted token: {np.argmax(torch_logits[0, 0])}")
    
    # Compare
    diff = np.abs(np.array(jax_logits) - torch_logits)
    print(f"\nMax difference: {np.max(diff):.6f}")
    print(f"Mean difference: {np.mean(diff):.6f}")
    
    if np.argmax(jax_logits[0, 0]) == np.argmax(torch_logits[0, 0]):
        print("✅ Predicted tokens MATCH!")
    else:
        print("❌ Predicted tokens DIFFER!")
    
    if np.max(diff) < 1e-3:
        print("✅ Decoder logits CLOSE ENOUGH!")
        return True
    else:
        print("⚠️  Decoder logits have some difference")
        return False


def compare_full_generation():
    """Compare full generation"""
    print("\n" + "="*80)
    print("STEP 4: Compare Full Generation")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    # Load models
    config, jax_params, _, _, tokenizer = load_fsmt_model(model_path, verbose=False)
    
    if not HAS_TRANSFORMERS:
        print("⚠️  Skipping - transformers not available")
        return
    
    torch_model = FSMTForConditionalGeneration.from_pretrained(model_path)
    torch_model.eval()
    
    # Test input
    test_text = "Hello, how are you?"
    print(f"Source: '{test_text}'")
    
    input_ids = tokenizer.encode(test_text, return_tensors='np')
    
    # JAX generation (greedy)
    jax_input = jnp.array(input_ids)
    jax_generated = generate_translation(
        jax_input,
        jax_params,
        config,
        tokenizer,
        max_length=20,
        temperature=0.0  # Greedy
    )
    
    jax_text = tokenizer.decode(jax_generated[0], skip_special_tokens=True)
    print(f"\n✅ JAX output: '{jax_text}'")
    print(f"   Token IDs: {jax_generated[0, :15]}")
    
    # PyTorch generation (greedy)
    with torch.no_grad():
        torch_input = torch.tensor(input_ids)
        torch_generated = torch_model.generate(
            torch_input,
            max_length=20,
            num_beams=1,  # Greedy
            do_sample=False
        )
        torch_generated = torch_generated.cpu().numpy()
    
    torch_text = tokenizer.decode(torch_generated[0], skip_special_tokens=True)
    print(f"\n✅ PyTorch output: '{torch_text}'")
    print(f"   Token IDs: {torch_generated[0, :15]}")
    
    # Compare
    if jax_text == torch_text:
        print("\n✅✅✅ OUTPUTS MATCH PERFECTLY! ✅✅✅")
    else:
        print("\n⚠️  Outputs differ:")
        print(f"   JAX:     '{jax_text}'")
        print(f"   PyTorch: '{torch_text}'")
        
        # Compare token by token
        min_len = min(len(jax_generated[0]), len(torch_generated[0]))
        match_count = 0
        for i in range(min_len):
            if jax_generated[0, i] == torch_generated[0, i]:
                match_count += 1
            else:
                print(f"\n   First diff at position {i}:")
                print(f"     JAX token: {jax_generated[0, i]}")
                print(f"     PyTorch token: {torch_generated[0, i]}")
                break
        
        print(f"\n   Matched {match_count}/{min_len} tokens ({100*match_count/min_len:.1f}%)")


def main():
    print("="*80)
    print("FSMT JAX vs PyTorch Comparison")
    print("="*80)
    
    if not HAS_TRANSFORMERS:
        print("\n❌ transformers library not installed")
        print("   Install with: pip install transformers torch")
        return
    
    # Run comparisons
    emb_ok = compare_embeddings()
    
    if emb_ok:
        enc_ok = compare_encoder()
        
        if enc_ok:
            dec_ok = compare_decoder_step()
    
    # Always try full generation comparison
    compare_full_generation()
    
    print("\n" + "="*80)
    print("Comparison Complete")
    print("="*80)


if __name__ == "__main__":
    main()
