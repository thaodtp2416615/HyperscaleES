"""
Test script for FSMT Model Loader

This script tests loading FSMT model from local directory
and validates the parameter structure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from hyperscalees.models.fsmt_loader import load_fsmt_model, get_fsmt_model
from hyperscalees.models.fsmt_analysis import FSMTParameterAnalyzer

def test_local_load():
    """Test loading from local directory"""
    print("="*80)
    print("TEST: Loading FSMT from Local Directory")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    print(f"\nModel path: {model_path}")
    print(f"Path exists: {Path(model_path).exists()}")
    
    if not Path(model_path).exists():
        print("❌ Model path does not exist. Please update the path.")
        return
    
    try:
        print("\n" + "-"*80)
        print("Loading model with freeze_nonlora=True (LoRA-style)...")
        print("-"*80)
        
        config, params, scan_map, es_map, tokenizer = load_fsmt_model(
            model_path,
            freeze_nonlora=True,
            verbose=True
        )
        
        print("\n✅ Model loaded successfully!")
        print(f"\nConfig:")
        print(f"  - Model type: {config.model_type}")
        print(f"  - Encoder layers: {config.encoder_layers}")
        print(f"  - Decoder layers: {config.decoder_layers}")
        print(f"  - d_model: {config.d_model}")
        print(f"  - Vocab size: {config.vocab_size}")
        print(f"  - Languages: {config.langs}")
        
        print(f"\nTokenizer:")
        print(f"  - Type: {type(tokenizer).__name__}")
        print(f"  - Vocab size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"\nTokenization test:")
        print(f"  - Input: '{test_text}'")
        print(f"  - Tokens: {tokens[:10]}... ({len(tokens)} total)")
        print(f"  - Decoded: '{decoded}'")
        
        # Test parameter structure
        print(f"\nParameter structure:")
        def count_params(tree, prefix=''):
            total = 0
            if isinstance(tree, dict):
                for key, value in tree.items():
                    total += count_params(value, f"{prefix}.{key}" if prefix else key)
            else:
                # JAX array
                import jax.numpy as jnp
                if hasattr(tree, 'shape'):
                    num = int(jnp.prod(jnp.array(tree.shape)))
                    total += num
            return total
        
        total_params = count_params(params)
        print(f"  - Total parameters: {total_params:,}")
        
        # Count evolved parameters
        def count_evolved(param_tree, es_tree):
            full = 0
            lora = 0
            if isinstance(param_tree, dict):
                for key in param_tree:
                    f, l = count_evolved(param_tree[key], es_tree[key])
                    full += f
                    lora += l
            else:
                import jax.numpy as jnp
                if hasattr(param_tree, 'shape'):
                    num = int(jnp.prod(jnp.array(param_tree.shape)))
                    if es_tree == 0:  # FULL evolution
                        full += num
                    else:  # LORA/frozen
                        lora += num
            return full, lora
        
        evolved_params, frozen_params = count_evolved(params, es_map)
        print(f"  - Evolved parameters: {evolved_params:,} ({100*evolved_params/total_params:.1f}%)")
        print(f"  - Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_get_model_interface():
    """Test get_fsmt_model interface (compatible with existing ES code)"""
    print("\n" + "="*80)
    print("TEST: get_fsmt_model() Interface")
    print("="*80)
    
    model_path = r"d:\HyperscaleES\user"
    
    if not Path(model_path).exists():
        print("❌ Model path does not exist. Skipping test.")
        return
    
    try:
        print("\nTesting interface compatible with existing ES code...")
        
        _, full_params, tokenizer = get_fsmt_model(
            model_name="local",
            model_path=model_path,
            freeze_nonlora=True,
            verbose=False
        )
        
        config, params, scan_map, es_map = full_params
        
        print(f"✅ Interface test passed!")
        print(f"  - Config: {config.encoder_layers}enc + {config.decoder_layers}dec layers")
        print(f"  - Params: pytree structure")
        print(f"  - Scan map: generated")
        print(f"  - ES map: generated")
        print(f"  - Tokenizer: {type(tokenizer).__name__}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_local_load()
    test_get_model_interface()
    
    print("\n" + "="*80)
    print("Testing Complete")
    print("="*80)
