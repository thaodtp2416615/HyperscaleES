"""
FSMT Forward Pass Implementation in JAX

Implements encoder-decoder transformer for machine translation with:
- Multi-head self-attention
- Cross-attention (encoder-decoder)
- Feed-forward networks
- Layer normalization
- Position embeddings
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional
from functools import partial


class FSMTModel:
    """FSMT Encoder-Decoder Transformer Model"""
    
    @staticmethod
    def sinusoidal_position_embedding(seq_len: int, d_model: int, max_len: int = 512) -> jnp.ndarray:
        """
        Create sinusoidal position embeddings
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
            max_len: Maximum sequence length
            
        Returns:
            Position embeddings [seq_len, d_model]
        """
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        
        pe = jnp.zeros((seq_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        return pe
    
    @staticmethod
    def layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
        """
        Layer normalization
        
        Args:
            x: Input [batch, seq_len, d_model]
            gamma: Scale parameter [d_model]
            beta: Shift parameter [d_model]
            eps: Epsilon for numerical stability
            
        Returns:
            Normalized output [batch, seq_len, d_model]
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + eps)
        return gamma * normalized + beta
    
    @staticmethod
    def scaled_dot_product_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        dropout_rate: float = 0.0,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Scaled dot-product attention
        
        Args:
            query: [batch, num_heads, seq_len_q, head_dim]
            key: [batch, num_heads, seq_len_k, head_dim]
            value: [batch, num_heads, seq_len_v, head_dim]
            mask: Optional mask [batch, 1, seq_len_q, seq_len_k]
            dropout_rate: Dropout probability
            training: Whether in training mode
            rng: Random key for dropout
            
        Returns:
            Attention output [batch, num_heads, seq_len_q, head_dim]
        """
        d_k = query.shape[-1]
        
        # Compute attention scores
        scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        
        # Apply softmax
        attention_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply dropout if training
        if training and dropout_rate > 0.0 and rng is not None:
            keep_prob = 1.0 - dropout_rate
            keep = jax.random.bernoulli(rng, keep_prob, attention_weights.shape)
            attention_weights = jnp.where(keep, attention_weights / keep_prob, 0.0)
        
        # Apply attention to values
        output = jnp.matmul(attention_weights, value)
        
        return output
    
    @staticmethod
    def multi_head_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        params: Dict,
        num_heads: int,
        mask: Optional[jnp.ndarray] = None,
        dropout_rate: float = 0.0,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Multi-head attention mechanism
        
        Args:
            query: [batch, seq_len_q, d_model]
            key: [batch, seq_len_k, d_model]
            value: [batch, seq_len_v, d_model]
            params: Dictionary with 'q_proj', 'k_proj', 'v_proj', 'out_proj' weights
            num_heads: Number of attention heads
            mask: Optional attention mask
            dropout_rate: Dropout probability
            training: Whether in training mode
            rng: Random key
            
        Returns:
            Output [batch, seq_len_q, d_model]
        """
        batch_size, seq_len_q, d_model = query.shape
        head_dim = d_model // num_heads
        
        # Linear projections
        Q = jnp.matmul(query, params['q_proj']['weight'].T) + params['q_proj']['bias']
        K = jnp.matmul(key, params['k_proj']['weight'].T) + params['k_proj']['bias']
        V = jnp.matmul(value, params['v_proj']['weight'].T) + params['v_proj']['bias']
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        Q = Q.reshape(batch_size, seq_len_q, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Apply attention
        attention_output = FSMTModel.scaled_dot_product_attention(
            Q, K, V, mask, dropout_rate, training, rng
        )
        
        # Reshape back to [batch, seq_len_q, d_model]
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)
        
        # Output projection
        output = jnp.matmul(attention_output, params['out_proj']['weight'].T) + params['out_proj']['bias']
        
        return output
    
    @staticmethod
    def feed_forward(x: jnp.ndarray, params: Dict, activation_fn: str = 'gelu') -> jnp.ndarray:
        """
        Position-wise feed-forward network
        
        Args:
            x: Input [batch, seq_len, d_model]
            params: Dictionary with 'fc1' and 'fc2' weights
            activation_fn: Activation function ('gelu', 'relu')
            
        Returns:
            Output [batch, seq_len, d_model]
        """
        # First linear layer
        hidden = jnp.matmul(x, params['fc1']['weight'].T) + params['fc1']['bias']
        
        # Activation
        if activation_fn == 'gelu':
            hidden = jax.nn.gelu(hidden)
        elif activation_fn == 'relu':
            hidden = jax.nn.relu(hidden)
        else:
            raise ValueError(f"Unknown activation: {activation_fn}")
        
        # Second linear layer
        output = jnp.matmul(hidden, params['fc2']['weight'].T) + params['fc2']['bias']
        
        return output
    
    @staticmethod
    def encoder_layer(
        x: jnp.ndarray,
        params: Dict,
        config,
        mask: Optional[jnp.ndarray] = None,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Single encoder layer (self-attention + FFN)
        
        Args:
            x: Input [batch, seq_len, d_model]
            params: Layer parameters
            config: Model configuration
            mask: Attention mask
            training: Training mode
            rng: Random key
            
        Returns:
            Output [batch, seq_len, d_model]
        """
        # Self-attention
        attn_output = FSMTModel.multi_head_attention(
            x, x, x,
            params['self_attn'],
            config.encoder_attention_heads,
            mask,
            config.attention_dropout if training else 0.0,
            training,
            rng
        )
        
        # Add & Norm
        x = x + attn_output
        x = FSMTModel.layer_norm(x, params['self_attn_layer_norm']['weight'], params['self_attn_layer_norm']['bias'])
        
        # Feed-forward
        ffn_output = FSMTModel.feed_forward(x, params, config.activation_function)
        
        # Add & Norm
        x = x + ffn_output
        x = FSMTModel.layer_norm(x, params['final_layer_norm']['weight'], params['final_layer_norm']['bias'])
        
        return x
    
    @staticmethod
    def decoder_layer(
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        params: Dict,
        config,
        self_attn_mask: Optional[jnp.ndarray] = None,
        cross_attn_mask: Optional[jnp.ndarray] = None,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Single decoder layer (self-attention + cross-attention + FFN)
        
        Args:
            x: Decoder input [batch, tgt_len, d_model]
            encoder_output: Encoder output [batch, src_len, d_model]
            params: Layer parameters
            config: Model configuration
            self_attn_mask: Self-attention mask (causal)
            cross_attn_mask: Cross-attention mask (padding)
            training: Training mode
            rng: Random key
            
        Returns:
            Output [batch, tgt_len, d_model]
        """
        # Self-attention (masked/causal)
        self_attn_output = FSMTModel.multi_head_attention(
            x, x, x,
            params['self_attn'],
            config.decoder_attention_heads,
            self_attn_mask,
            config.attention_dropout if training else 0.0,
            training,
            rng
        )
        
        # Add & Norm
        x = x + self_attn_output
        x = FSMTModel.layer_norm(x, params['self_attn_layer_norm']['weight'], params['self_attn_layer_norm']['bias'])
        
        # Cross-attention (encoder-decoder)
        cross_attn_output = FSMTModel.multi_head_attention(
            x, encoder_output, encoder_output,
            params['encoder_attn'],
            config.decoder_attention_heads,
            cross_attn_mask,
            config.attention_dropout if training else 0.0,
            training,
            rng
        )
        
        # Add & Norm
        x = x + cross_attn_output
        x = FSMTModel.layer_norm(x, params['encoder_attn_layer_norm']['weight'], params['encoder_attn_layer_norm']['bias'])
        
        # Feed-forward
        ffn_output = FSMTModel.feed_forward(x, params, config.activation_function)
        
        # Add & Norm
        x = x + ffn_output
        x = FSMTModel.layer_norm(x, params['final_layer_norm']['weight'], params['final_layer_norm']['bias'])
        
        return x
    
    @staticmethod
    def encode(
        input_ids: jnp.ndarray,
        params: Dict,
        config,
        attention_mask: Optional[jnp.ndarray] = None,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Encode source sequence
        
        Args:
            input_ids: Input token IDs [batch, src_len]
            params: Model parameters
            config: Model configuration
            attention_mask: Padding mask [batch, src_len]
            training: Training mode
            rng: Random key
            
        Returns:
            Encoder output [batch, src_len, d_model]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embeddings = params['encoder']['embed_tokens']['weight'][input_ids]
        
        # Scale embeddings (like FSMT)
        if config.scale_embedding:
            embeddings = embeddings * jnp.sqrt(config.d_model)
        
        # Add positional embeddings
        pos_embeddings = params['encoder']['embed_positions']['weight'][:seq_len]
        x = embeddings + pos_embeddings
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to [batch, 1, 1, seq_len] for broadcasting
            attn_mask = attention_mask[:, None, None, :]
            attn_mask = (attn_mask == 0)  # True where padding
        else:
            attn_mask = None
        
        # Apply encoder layers
        for i in range(config.encoder_layers):
            layer_rng = jax.random.fold_in(rng, i) if rng is not None else None
            x = FSMTModel.encoder_layer(
                x,
                params['encoder']['layers'][str(i)],
                config,
                attn_mask,
                training,
                layer_rng
            )
        
        return x
    
    @staticmethod
    def decode(
        decoder_input_ids: jnp.ndarray,
        encoder_output: jnp.ndarray,
        params: Dict,
        config,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Decode target sequence
        
        Args:
            decoder_input_ids: Target token IDs [batch, tgt_len]
            encoder_output: Encoder output [batch, src_len, d_model]
            params: Model parameters
            config: Model configuration
            decoder_attention_mask: Padding mask for decoder
            encoder_attention_mask: Padding mask for encoder
            training: Training mode
            rng: Random key
            
        Returns:
            Logits [batch, tgt_len, vocab_size]
        """
        batch_size, tgt_len = decoder_input_ids.shape
        
        # Embedding (tied with encoder - decoder uses encoder's embed_tokens)
        embeddings = params['encoder']['embed_tokens']['weight'][decoder_input_ids]
        
        # Scale embeddings
        if config.scale_embedding:
            embeddings = embeddings * jnp.sqrt(config.d_model)
        
        # Add positional embeddings
        pos_embeddings = params['decoder']['embed_positions']['weight'][:tgt_len]
        x = embeddings + pos_embeddings
        
        # Prepare causal mask for self-attention
        causal_mask = jnp.tril(jnp.ones((tgt_len, tgt_len)))
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, tgt_len, tgt_len]
        
        # Combine with padding mask if provided
        if decoder_attention_mask is not None:
            padding_mask = decoder_attention_mask[:, None, None, :]
            self_attn_mask = causal_mask * padding_mask
        else:
            self_attn_mask = causal_mask
        
        self_attn_mask = (self_attn_mask == 0)  # True where masked
        
        # Prepare cross-attention mask
        if encoder_attention_mask is not None:
            cross_attn_mask = encoder_attention_mask[:, None, None, :]
            cross_attn_mask = (cross_attn_mask == 0)
        else:
            cross_attn_mask = None
        
        # Apply decoder layers
        for i in range(config.decoder_layers):
            layer_rng = jax.random.fold_in(rng, i + 100) if rng is not None else None
            x = FSMTModel.decoder_layer(
                x,
                encoder_output,
                params['decoder']['layers'][str(i)],
                config,
                self_attn_mask,
                cross_attn_mask,
                training,
                layer_rng
            )
        
        # Project to vocabulary (tied embeddings - use encoder's embed_tokens)
        logits = jnp.matmul(x, params['encoder']['embed_tokens']['weight'].T)
        
        return logits
    
    @staticmethod
    def forward(
        input_ids: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        params: Dict,
        config,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Full forward pass (encode + decode)
        
        Args:
            input_ids: Source token IDs [batch, src_len]
            decoder_input_ids: Target token IDs [batch, tgt_len]
            params: Model parameters
            config: Model configuration
            attention_mask: Source padding mask
            decoder_attention_mask: Target padding mask
            training: Training mode
            rng: Random key
            
        Returns:
            Logits [batch, tgt_len, vocab_size]
        """
        # Encode
        encoder_rng = jax.random.fold_in(rng, 0) if rng is not None else None
        encoder_output = FSMTModel.encode(
            input_ids, params, config, attention_mask, training, encoder_rng
        )
        
        # Decode
        decoder_rng = jax.random.fold_in(rng, 1) if rng is not None else None
        logits = FSMTModel.decode(
            decoder_input_ids, encoder_output, params, config,
            decoder_attention_mask, attention_mask, training, decoder_rng
        )
        
        return logits
