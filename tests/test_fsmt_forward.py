"""
Test script for FSMT forward pass and generation

Tests the JAX implementation of FSMT encoder-decoder.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp

from hyperscalees.models.fsmt_loader import load_fsmt_model
from hyperscalees.models.fsmt import FSMTModel, generate_translation


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("="*80)
    print("TEST: FSMT Forward Pass")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    if not Path(model_path).exists():
        print("❌ Model path does not exist")
        return
    
    try:
        # Load model
        print("\nLoading model...")
        config, params, scan_map, es_map, tokenizer = load_fsmt_model(
            model_path,
            freeze_nonlora=True,
            verbose=False
        )
        
        print(f"✅ Model loaded")
        print(f"  Config: {config.encoder_layers}enc + {config.decoder_layers}dec")
        print(f"  Vocab size: {config.vocab_size}")
        
        # Create dummy input
        batch_size = 2
        src_len = 10
        tgt_len = 8
        
        print(f"\n Testing with batch_size={batch_size}, src_len={src_len}, tgt_len={tgt_len}")
        
        # Random token IDs
        rng = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(rng, (batch_size, src_len), 3, 1000)
        decoder_input_ids = jax.random.randint(rng, (batch_size, tgt_len), 3, 1000)
        
        # Create attention masks (all 1s = no padding)
        attention_mask = jnp.ones((batch_size, src_len))
        decoder_attention_mask = jnp.ones((batch_size, tgt_len))
        
        print("\n Encoding...")
        encoder_output = FSMTModel.encode(
            input_ids, params, config, attention_mask, training=False
        )
        print(f"  ✅ Encoder output shape: {encoder_output.shape}")
        print(f"     Expected: ({batch_size}, {src_len}, {config.d_model})")
        
        print("\n Decoding...")
        logits = FSMTModel.decode(
            decoder_input_ids,
            encoder_output,
            params,
            config,
            decoder_attention_mask,
            attention_mask,
            training=False
        )
        print(f"  ✅ Decoder logits shape: {logits.shape}")
        print(f"     Expected: ({batch_size}, {tgt_len}, {config.vocab_size})")
        
        print("\n Full forward pass...")
        full_logits = FSMTModel.forward(
            input_ids,
            decoder_input_ids,
            params,
            config,
            attention_mask,
            decoder_attention_mask,
            training=False
        )
        print(f"  ✅ Full forward logits shape: {full_logits.shape}")
        
        # Check if logits are reasonable
        print(f"\n Logit statistics:")
        print(f"  Mean: {jnp.mean(full_logits):.4f}")
        print(f"  Std: {jnp.std(full_logits):.4f}")
        print(f"  Min: {jnp.min(full_logits):.4f}")
        print(f"  Max: {jnp.max(full_logits):.4f}")
        
        # Check probabilities
        probs = jax.nn.softmax(full_logits, axis=-1)
        print(f"\n Probability statistics:")
        print(f"  Max prob per position: {jnp.max(probs, axis=-1)[0, :5]}")
        
        print("\n✅ Forward pass test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_generation():
    """Test generation with real text"""
    print("\n" + "="*80)
    print("TEST: FSMT Generation")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    if not Path(model_path).exists():
        print("❌ Model path does not exist")
        return
    
    try:
        # Load model
        print("\nLoading model...")
        config, params, scan_map, es_map, tokenizer = load_fsmt_model(
            model_path,
            freeze_nonlora=True,
            verbose=False
        )
        
        print(f"✅ Model loaded")
        
        # Test with real text
        source_text = "Hello, how are you?"
        print(f"\n Source text: '{source_text}'")
        
        # Tokenize
        input_ids = tokenizer.encode(source_text, return_tensors='np')
        input_ids = jnp.array(input_ids)
        
        print(f"  Tokenized: {input_ids.shape}")
        print(f"  Tokens: {input_ids[0, :10]}...")
        
        # Generate (greedy)
        print("\n Generating (greedy)...")
        generated_ids = generate_translation(
            input_ids,
            params,
            config,
            tokenizer,
            max_length=50,
            temperature=0.0
        )
        
        print(f"  Generated shape: {generated_ids.shape}")
        print(f"  Generated tokens: {generated_ids[0, :20]}...")
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\n✅ Generated translation:")
        print(f"     '{generated_text}'")
        
        # Test with sampling
        print("\n Generating (with temperature=1.0)...")
        rng = jax.random.PRNGKey(123)
        generated_ids_sample = generate_translation(
            input_ids,
            params,
            config,
            tokenizer,
            max_length=50,
            temperature=1.0,
            rng=rng
        )
        
        generated_text_sample = tokenizer.decode(generated_ids_sample[0], skip_special_tokens=True)
        print(f"\n  Generated translation (sampled):")
        print(f"     '{generated_text_sample}'")
        
        print("\n✅ Generation test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_compilation():
    """Test JIT compilation"""
    print("\n" + "="*80)
    print("TEST: JIT Compilation")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    if not Path(model_path).exists():
        print("❌ Model path does not exist")
        return
    
    try:
        # Load model
        config, params, scan_map, es_map, tokenizer = load_fsmt_model(
            model_path,
            freeze_nonlora=True,
            verbose=False
        )
        
        print("\n Compiling forward pass...")
        
        # Create dummy inputs
        input_ids = jnp.ones((1, 10), dtype=jnp.int32)
        decoder_input_ids = jnp.ones((1, 8), dtype=jnp.int32)
        
        # JIT compile
        @jax.jit
        def forward_fn(input_ids, decoder_input_ids):
            return FSMTModel.forward(
                input_ids, decoder_input_ids, params, config,
                training=False
            )
        
        import time
        
        # First call (compilation)
        start = time.time()
        _ = forward_fn(input_ids, decoder_input_ids).block_until_ready()
        compile_time = time.time() - start
        print(f"  First call (compile): {compile_time:.3f}s")
        
        # Second call (cached)
        start = time.time()
        _ = forward_fn(input_ids, decoder_input_ids).block_until_ready()
        run_time = time.time() - start
        print(f"  Second call (cached): {run_time:.3f}s")
        
        print(f"\n  Speedup: {compile_time/run_time:.1f}x")
        
        print("\n✅ Compilation test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("FSMT Forward Pass and Generation Tests\n")
    
    test_forward_pass()
    test_generation()
    test_compilation()
    
    print("\n" + "="*80)
    print("All Tests Complete")
    print("="*80)
