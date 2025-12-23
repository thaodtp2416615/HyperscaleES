"""Test which GELU variant PyTorch FSMT uses."""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import FSMTForConditionalGeneration

def gelu_tanh_approx(x):
    """GELU approximation using tanh."""
    return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def gelu_exact(x):
    """Exact GELU using erf."""
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

# Load model
model_path = r"d:\HyperscaleES\user"
model = FSMTForConditionalGeneration.from_pretrained(model_path)
model.eval()

# Test input
x = torch.randn(1, 5, 1024)

# Get actual model activation
print("Testing FSMT activation function...")
print(f"activation_function in config: {model.config.activation_function}")

# Apply fc1 and get activation
with torch.no_grad():
    fc1_out = model.model.encoder.layers[0].fc1(x)
    actual_activation = model.model.encoder.layers[0].activation_fn(fc1_out)
    
    # Compare with different GELU variants
    gelu_new = F.gelu(fc1_out)  # PyTorch default (exact)
    gelu_tanh = gelu_tanh_approx(fc1_out)
    gelu_exact_manual = gelu_exact(fc1_out)

# Compare differences
print("\nComparing GELU variants:")
print(f"actual vs gelu (exact): max diff = {(actual_activation - gelu_new).abs().max():.8f}")
print(f"actual vs gelu_tanh: max diff = {(actual_activation - gelu_tanh).abs().max():.8f}")
print(f"actual vs gelu_exact: max diff = {(actual_activation - gelu_exact_manual).abs().max():.8f}")

print("\nActual activation values (first 5):", actual_activation.flatten()[:5].numpy())
print("F.gelu (exact) values (first 5):", gelu_new.flatten()[:5].numpy())
print("gelu_tanh values (first 5):", gelu_tanh.flatten()[:5].numpy())

# Check if model uses 'new_gelu' or 'gelu'
print("\nChecking activation_fn type:", type(model.model.encoder.layers[0].activation_fn))
print("Activation function name:", model.model.encoder.layers[0].activation_fn.__name__ if hasattr(model.model.encoder.layers[0].activation_fn, '__name__') else 'N/A')
