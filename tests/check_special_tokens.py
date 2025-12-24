"""Check FSMT special tokens."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import FSMTForConditionalGeneration, T5Tokenizer
from hyperscalees.models.fsmt_loader import load_fsmt_model

model_path = r"d:\HyperscaleES\user"

# PyTorch
pt_model = FSMTForConditionalGeneration.from_pretrained(model_path)
pt_tokenizer = T5Tokenizer.from_pretrained(model_path)

print("PyTorch Model Config:")
print(f"  pad_token_id: {pt_model.config.pad_token_id}")
print(f"  eos_token_id: {pt_model.config.eos_token_id}")
print(f"  bos_token_id: {pt_model.config.bos_token_id}")
print(f"  decoder_start_token_id: {pt_model.config.decoder_start_token_id}")

print("\nPyTorch Tokenizer:")
print(f"  pad_token: '{pt_tokenizer.pad_token}' (id={pt_tokenizer.pad_token_id})")
print(f"  eos_token: '{pt_tokenizer.eos_token}' (id={pt_tokenizer.eos_token_id})")
print(f"  bos_token: '{pt_tokenizer.bos_token}' (id={getattr(pt_tokenizer, 'bos_token_id', None)})")

# JAX
config, params, _, _, tokenizer = load_fsmt_model(model_path)

print("\nJAX Config:")
print(f"  pad_token_id: {config.pad_token_id}")
print(f"  eos_token_id: {config.eos_token_id}")
print(f"  bos_token_id: {config.bos_token_id}")
print(f"  decoder_start_token_id: {config.decoder_start_token_id}")

print("\nTokenizer special tokens:")
print(f"  pad: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
print(f"  eos: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

# Check PyTorch generation behavior
import torch
text = "Hello, how are you?"
inputs = pt_tokenizer(text, return_tensors='pt')

print(f"\nPyTorch generation test:")
print(f"  Input IDs: {inputs['input_ids'][0].tolist()}")

with torch.no_grad():
    outputs = pt_model.generate(inputs['input_ids'], max_length=20, num_beams=1, do_sample=False)
    
print(f"  Generated IDs: {outputs[0].tolist()}")
print(f"  Generated text: '{pt_tokenizer.decode(outputs[0], skip_special_tokens=True)}'")
