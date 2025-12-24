"""Check actual embed_scale value in PyTorch FSMT."""
from transformers import FSMTForConditionalGeneration
import math

model_path = r"d:\HyperscaleES\user"
model = FSMTForConditionalGeneration.from_pretrained(model_path)

print(f"Config scale_embedding: {model.config.scale_embedding}")
print(f"Config d_model: {model.config.d_model}")

encoder = model.model.encoder
print(f"\nEncoder embed_scale: {encoder.embed_scale}")
print(f"Expected sqrt(d_model): {math.sqrt(model.config.d_model)}")
print(f"Match sqrt? {abs(encoder.embed_scale - math.sqrt(model.config.d_model)) < 1e-6}")

decoder = model.model.decoder
print(f"\nDecoder embed_scale: {decoder.embed_scale}")
