"""
FSMT (FairSeq Machine Translation) Model Configuration
Configuration for English-German translation model
"""

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class FSMTConfig:
    """Configuration for FSMT Translation Model"""
    
    # Model architecture
    model_type: str = "fsmt"
    is_encoder_decoder: bool = True
    
    # Model dimensions
    d_model: int = 1024
    max_position_embeddings: int = 512
    
    # Encoder configuration
    encoder_layers: int = 8
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    encoder_layerdrop: float = 0.0
    
    # Decoder configuration
    decoder_layers: int = 2
    decoder_attention_heads: int = 16
    decoder_ffn_dim: int = 4096
    decoder_layerdrop: float = 0.0
    
    # Vocabulary
    src_vocab_size: int = 40963
    tgt_vocab_size: int = 40963
    vocab_size: int = 40963  # For compatibility
    
    # Special tokens (based on T5Tokenizer format)
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 0
    unk_token_id: int = 2
    decoder_start_token_id: int = 1
    forced_eos_token_id: int = 1
    
    # Languages
    langs: List[str] = None
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    
    # Activation
    activation_function: str = "gelu"
    
    # Generation
    max_length: int = 200
    num_beams: Optional[int] = None
    use_cache: bool = True
    
    # Tokenizer (T5-based)
    tokenizer_class: str = "T5Tokenizer"
    add_prefix_space: bool = True
    
    # Training
    init_std: float = 0.02
    scale_embedding: bool = True
    tie_word_embeddings: bool = True
    
    # Evolution Strategy specific
    num_hidden_layers: int = 8  # Total layers for compatibility
    
    def __post_init__(self):
        if self.langs is None:
            self.langs = ["en", "de"]
    
    @classmethod
    def from_pretrained_config(cls, config_dict):
        """Create FSMTConfig from pretrained model config dict"""
        return cls(
            model_type=config_dict.get("model_type", "fsmt"),
            is_encoder_decoder=config_dict.get("is_encoder_decoder", True),
            d_model=config_dict.get("d_model", 1024),
            max_position_embeddings=config_dict.get("max_position_embeddings", 512),
            encoder_layers=config_dict.get("encoder_layers", 8),
            encoder_attention_heads=config_dict.get("encoder_attention_heads", 16),
            encoder_ffn_dim=config_dict.get("encoder_ffn_dim", 4096),
            encoder_layerdrop=config_dict.get("encoder_layerdrop", 0.0),
            decoder_layers=config_dict.get("decoder_layers", 2),
            decoder_attention_heads=config_dict.get("decoder_attention_heads", 16),
            decoder_ffn_dim=config_dict.get("decoder_ffn_dim", 4096),
            decoder_layerdrop=config_dict.get("decoder_layerdrop", 0.0),
            src_vocab_size=config_dict.get("src_vocab_size", 40963),
            tgt_vocab_size=config_dict.get("tgt_vocab_size", 40963),
            unk_token_id=config_dict.get("unk_token_id", 2),
            decoder_start_token_id=config_dict.get("decoder_start_token_id", 1),
            forced_eos_token_id=config_dict.get("forced_eos_token_id", 1
            eos_token_id=config_dict.get("eos_token_id", 1),
            pad_token_id=config_dict.get("pad_token_id", 0),
            decoder_start_token_id=config_dict.get("decoder_start_token_id", 1),
            forced_eos_token_id=config_dict.get("forced_eos_token_id"),
            langs=config_dict.get("langs", ["en", "de"]),
            dropout=config_dict.get("dropout", 0.1),
            attention_dropout=config_dict.get("attention_dropout", 0.0),
            activation_dropout=config_dict.get("activation_dropout", 0.0),
            activation_function=config_dict.get("activation_function", "gelu"),
            max_length=config_dict.get("max_length", 200),
            num_beams=config_dict.get("num_beams"),
            use_cache=config_dict.get("use_cache", True),
            init_std=config_dict.get("init_std", 0.02),
            scale_embedding=config_dict.get("scale_embedding", True),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            tokenizer_class=config_dict.get("tokenizer_class", "T5Tokenizer"),
            add_prefix_space=config_dict.get("add_prefix_space", True),
            num_hidden_layers=config_dict.get("num_hidden_layers", 8),
        )


# Default configurations for different model sizes
FSMT_SMALL = FSMTConfig(
    d_model=512,
    encoder_layers=4,
    decoder_layers=2,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    encoder_ffn_dim=2048,
    decoder_ffn_dim=2048,
)

FSMT_BASE = FSMTConfig(
    d_model=1024,
    encoder_layers=8,
    decoder_layers=2,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    encoder_ffn_dim=4096,
    decoder_ffn_dim=4096,
)

FSMT_LARGE = FSMTConfig(
    d_model=1024,
    encoder_layers=12,
    decoder_layers=6,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    encoder_ffn_dim=4096,
    decoder_ffn_dim=4096,
)
