"""Test JAX GELU variants."""
import jax
import jax.numpy as jnp
import numpy as np

# Test input
x = jnp.array([0.5, 1.0, -0.5, 2.0, -1.0])

print("Testing JAX GELU variants:")
print(f"Input: {x}")

# Default GELU
gelu_default = jax.nn.gelu(x)
print(f"\njax.nn.gelu(x): {gelu_default}")

# Exact GELU (approximate=False)
gelu_exact = jax.nn.gelu(x, approximate=False)
print(f"jax.nn.gelu(x, approximate=False): {gelu_exact}")

# Approximate GELU (approximate=True)
gelu_approx = jax.nn.gelu(x, approximate=True)
print(f"jax.nn.gelu(x, approximate=True): {gelu_approx}")

# Check difference
print(f"\nDifference (exact vs approx): {jnp.abs(gelu_exact - gelu_approx).max():.8f}")
print(f"Are default and exact same? {jnp.allclose(gelu_default, gelu_exact)}")
print(f"Are default and approx same? {jnp.allclose(gelu_default, gelu_approx)}")

# Manual exact GELU
def gelu_manual_exact(x):
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))

gelu_manual = gelu_manual_exact(x)
print(f"\nManual exact GELU: {gelu_manual}")
print(f"Manual vs jax exact diff: {jnp.abs(gelu_manual - gelu_exact).max():.8f}")
