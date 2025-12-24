"""Compare JAX vs PyTorch generation output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from hyperscalees.models.fsmt_loader import load_fsmt_model
from hyperscalees.models.fsmt import generate_translation

model_path = r"d:\HyperscaleES\user"

# Load models
print("Loading models...")
config, params, scan_map, es_map, jax_tokenizer = load_fsmt_model(model_path)
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_model.eval()
pt_tokenizer = FSMTTokenizer.from_pretrained(model_path)

# Test texts
test_texts = [
    "Hello, how are you?",
    "The weather is nice today.",
    "Machine learning is fascinating."
]

for text in test_texts:
    print(f"\n{'='*80}")
    print(f"Source: '{text}'")
    print('='*80)
    
    # PyTorch generation
    pt_inputs = pt_tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        pt_output = pt_model.generate(
            pt_inputs['input_ids'],
            max_length=50,
            num_beams=1,
            do_sample=False
        )
    pt_translation = pt_tokenizer.decode(pt_output[0], skip_special_tokens=True)
    
    print(f"\nPyTorch (greedy):")
    print(f"  Tokens: {pt_output[0].numpy()[:20]}")
    print(f"  Text: '{pt_translation}'")
    
    # JAX generation
    jax_inputs = jax_tokenizer(text, return_tensors='pt')
    input_ids = jnp.array(jax_inputs['input_ids'].numpy())
    
    jax_output = generate_translation(
        input_ids,
        params,
        config,
        jax_tokenizer,
        max_length=50,
        temperature=0.0
    )
    jax_translation = jax_tokenizer.decode(jax_output[0], skip_special_tokens=True)
    
    print(f"\nJAX (greedy):")
    print(f"  Tokens: {jax_output[0][:20]}")
    print(f"  Text: '{jax_translation}'")
    
    # Compare
    match = "✅ MATCH" if pt_translation == jax_translation else "❌ DIFFERENT"
    print(f"\n{match}")
