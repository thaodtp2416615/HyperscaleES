"""Trace PyTorch FSMT encoder.forward() to see what it actually does."""
import torch
from transformers import FSMTForConditionalGeneration, T5Tokenizer
from transformers.models.fsmt.modeling_fsmt import FSMTEncoder
import inspect

model_path = r"d:\HyperscaleES\user"
model = FSMTForConditionalGeneration.from_pretrained(model_path)

print("="*80)
print("FSMTEncoder.forward() source code:")
print("="*80)
print(inspect.getsource(model.model.encoder.forward))
