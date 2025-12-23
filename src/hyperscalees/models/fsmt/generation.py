"""
Generation utilities for FSMT translation model

Implements greedy and beam search generation for neural machine translation.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple, List
from functools import partial

from .forward import FSMTModel


def generate_translation(
    input_ids: jnp.ndarray,
    params: Dict,
    config,
    tokenizer,
    max_length: Optional[int] = None,
    temperature: float = 0.0,
    attention_mask: Optional[jnp.ndarray] = None,
    rng: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Greedy generation for translation
    
    Args:
        input_ids: Source token IDs [batch, src_len]
        params: Model parameters
        config: Model configuration
        tokenizer: Tokenizer (for special token IDs)
        max_length: Maximum generation length (default: config.max_length)
        temperature: Sampling temperature (0.0 = greedy)
        attention_mask: Source padding mask
        rng: Random key for sampling
        
    Returns:
        Generated token IDs [batch, generated_len]
    """
    if max_length is None:
        max_length = config.max_length
    
    batch_size = input_ids.shape[0]
    
    # Encode source once
    encoder_output = FSMTModel.encode(
        input_ids, params, config, attention_mask, training=False
    )
    
    # Initialize with decoder_start_token_id
    decoder_input_ids = jnp.full((batch_size, 1), config.decoder_start_token_id, dtype=jnp.int32)
    
    # Track which sequences have finished
    finished = jnp.zeros(batch_size, dtype=jnp.bool_)
    
    # Generate tokens one by one
    for step in range(max_length - 1):
        # Decode
        logits = FSMTModel.decode(
            decoder_input_ids,
            encoder_output,
            params,
            config,
            encoder_attention_mask=attention_mask,
            training=False
        )
        
        # Get logits for last position
        next_token_logits = logits[:, -1, :]
        
        # Sample or greedy
        if temperature > 0.0 and rng is not None:
            # Temperature sampling
            next_token_logits = next_token_logits / temperature
            step_rng = jax.random.fold_in(rng, step)
            next_tokens = jax.random.categorical(step_rng, next_token_logits, axis=-1)
        else:
            # Greedy
            next_tokens = jnp.argmax(next_token_logits, axis=-1)
        
        # Mask finished sequences (keep generating pad tokens)
        next_tokens = jnp.where(finished, config.pad_token_id, next_tokens)
        
        # Append to decoder_input_ids
        decoder_input_ids = jnp.concatenate(
            [decoder_input_ids, next_tokens[:, None]], axis=1
        )
        
        # Update finished status
        finished = finished | (next_tokens == config.eos_token_id)
        
        # Stop if all sequences finished
        if jnp.all(finished):
            break
    
    # Remove decoder_start_token from output
    generated_ids = decoder_input_ids[:, 1:]
    
    return generated_ids


def beam_search_generate(
    input_ids: jnp.ndarray,
    params: Dict,
    config,
    tokenizer,
    num_beams: int = 5,
    max_length: Optional[int] = None,
    length_penalty: float = 1.0,
    attention_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Beam search generation for translation
    
    Args:
        input_ids: Source token IDs [batch, src_len]
        params: Model parameters
        config: Model configuration
        tokenizer: Tokenizer
        num_beams: Number of beams
        max_length: Maximum generation length
        length_penalty: Length penalty factor (>1 prefers longer, <1 prefers shorter)
        attention_mask: Source padding mask
        
    Returns:
        Best generated sequence per batch [batch, generated_len]
    """
    if max_length is None:
        max_length = config.max_length
    
    batch_size, src_len = input_ids.shape
    
    # Encode source
    encoder_output = FSMTModel.encode(
        input_ids, params, config, attention_mask, training=False
    )
    
    # Expand encoder output for beams [batch * num_beams, src_len, d_model]
    encoder_output = jnp.repeat(encoder_output, num_beams, axis=0)
    if attention_mask is not None:
        attention_mask = jnp.repeat(attention_mask, num_beams, axis=0)
    
    # Initialize beams
    # Shape: [batch * num_beams, 1]
    beam_decoder_input_ids = jnp.full(
        (batch_size * num_beams, 1),
        config.decoder_start_token_id,
        dtype=jnp.int32
    )
    
    # Beam scores: [batch, num_beams]
    beam_scores = jnp.zeros((batch_size, num_beams))
    beam_scores = beam_scores.at[:, 1:].set(-1e9)  # Only first beam is active initially
    beam_scores = beam_scores.reshape(-1)  # [batch * num_beams]
    
    # Track finished beams
    beam_finished = jnp.zeros(batch_size * num_beams, dtype=jnp.bool_)
    
    for step in range(max_length - 1):
        # Decode current sequence
        logits = FSMTModel.decode(
            beam_decoder_input_ids,
            encoder_output,
            params,
            config,
            encoder_attention_mask=attention_mask,
            training=False
        )
        
        # Get logits for last position: [batch * num_beams, vocab_size]
        next_token_logits = logits[:, -1, :]
        next_token_scores = jax.nn.log_softmax(next_token_logits, axis=-1)
        
        # Add to beam scores: [batch * num_beams, vocab_size]
        next_token_scores = next_token_scores + beam_scores[:, None]
        
        # Reshape to [batch, num_beams * vocab_size]
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.reshape(batch_size, num_beams * vocab_size)
        
        # Get top 2*num_beams candidates
        top_scores, top_indices = jax.lax.top_k(next_token_scores, 2 * num_beams)
        
        # Determine which beam and token each candidate corresponds to
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        # Select top num_beams
        # This is simplified; production code would handle EOS properly
        beam_indices = beam_indices[:, :num_beams]
        token_indices = token_indices[:, :num_beams]
        top_scores = top_scores[:, :num_beams]
        
        # Update decoder_input_ids
        # Gather the sequences for selected beams
        batch_beam_idx = jnp.arange(batch_size)[:, None] * num_beams + beam_indices
        batch_beam_idx = batch_beam_idx.reshape(-1)
        
        beam_decoder_input_ids = beam_decoder_input_ids[batch_beam_idx]
        
        # Append new tokens
        new_tokens = token_indices.reshape(-1, 1)
        beam_decoder_input_ids = jnp.concatenate(
            [beam_decoder_input_ids, new_tokens], axis=1
        )
        
        # Update scores
        beam_scores = top_scores.reshape(-1)
        
        # Check for EOS (simplified)
        beam_finished = beam_finished | (new_tokens.squeeze(-1) == config.eos_token_id)
        
        if jnp.all(beam_finished):
            break
    
    # Select best beam for each batch (first beam)
    best_beam_ids = jnp.arange(batch_size) * num_beams
    best_sequences = beam_decoder_input_ids[best_beam_ids]
    
    # Remove decoder_start_token
    best_sequences = best_sequences[:, 1:]
    
    return best_sequences


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def generate_single_greedy(
    input_ids: jnp.ndarray,
    params: Dict,
    config,
    max_length: int,
    decoder_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    attention_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    JIT-compiled greedy generation (single sequence)
    
    This is a simplified version optimized for ES where we generate many
    sequences in parallel.
    
    Args:
        input_ids: Source tokens [src_len]
        params: Model parameters
        config: Model configuration
        max_length: Maximum generation length
        decoder_start_token_id: Start token ID
        eos_token_id: End token ID
        pad_token_id: Padding token ID
        attention_mask: Source mask [src_len]
        
    Returns:
        Generated tokens [max_length]
    """
    # Add batch dimension
    input_ids = input_ids[None, :]
    if attention_mask is not None:
        attention_mask = attention_mask[None, :]
    
    # Encode
    encoder_output = FSMTModel.encode(
        input_ids, params, config, attention_mask, training=False
    )
    
    # Initialize decoder input
    decoder_input_ids = jnp.array([[decoder_start_token_id]], dtype=jnp.int32)
    
    def generation_step(carry, _):
        decoder_input_ids, finished = carry
        
        # Decode
        logits = FSMTModel.decode(
            decoder_input_ids,
            encoder_output,
            params,
            config,
            encoder_attention_mask=attention_mask,
            training=False
        )
        
        # Get next token (greedy)
        next_token = jnp.argmax(logits[0, -1, :])
        
        # Mask if finished
        next_token = jnp.where(finished, pad_token_id, next_token)
        
        # Append
        decoder_input_ids = jnp.concatenate(
            [decoder_input_ids, next_token[None, None]], axis=1
        )
        
        # Update finished
        finished = finished | (next_token == eos_token_id)
        
        return (decoder_input_ids, finished), next_token
    
    # Generate tokens
    init_carry = (decoder_input_ids, jnp.array(False))
    _, generated_tokens = jax.lax.scan(
        generation_step,
        init_carry,
        None,
        length=max_length - 1
    )
    
    # Pad if needed
    current_len = generated_tokens.shape[0]
    if current_len < max_length:
        padding = jnp.full(max_length - current_len, pad_token_id, dtype=jnp.int32)
        generated_tokens = jnp.concatenate([generated_tokens, padding])
    
    return generated_tokens
