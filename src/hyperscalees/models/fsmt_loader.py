"""
FSMT Model Loader for Evolution Strategies

Loads FSMT (FairSeq Machine Translation) models from:
- Local directory (with config.json, model.safetensors/pytorch_model.bin)
- Hugging Face Hub (model_name like 'facebook/wmt19-en-de')

Converts PyTorch parameters to JAX format and creates ES-compatible structure.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import warnings

import jax
import jax.numpy as jnp
import numpy as np

try:
    from transformers import AutoTokenizer, FSMTForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not installed. Install with: pip install transformers")

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    warnings.warn("safetensors not installed. Install with: pip install safetensors")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("torch not installed. Install with: pip install torch")

from .fsmt_config import FSMTConfig
from .fsmt_analysis import FSMTParameterAnalyzer


class FSMTModelLoader:
    """Load FSMT models and convert to JAX format"""
    
    def __init__(self, model_path: Union[str, Path], verbose: bool = True):
        """
        Args:
            model_path: Path to local model directory or Hugging Face model name
            verbose: Print loading information
        """
        self.model_path = Path(model_path) if isinstance(model_path, str) and os.path.exists(model_path) else model_path
        self.verbose = verbose
        self.config = None
        self.tokenizer = None
        
    def _log(self, msg: str):
        """Print if verbose"""
        if self.verbose:
            print(f"[FSMTLoader] {msg}")
    
    def load_config(self) -> FSMTConfig:
        """Load model configuration"""
        if isinstance(self.model_path, Path):
            # Load from local directory
            config_path = self.model_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"config.json not found in {self.model_path}")
            
            self._log(f"Loading config from {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            # Load from Hugging Face Hub
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers required for loading from HF Hub")
            
            self._log(f"Loading config from Hugging Face Hub: {self.model_path}")
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(self.model_path)
            config_dict = hf_config.to_dict()
        
        self.config = FSMTConfig.from_pretrained_config(config_dict)
        self._log(f"Config loaded: {self.config.encoder_layers} encoder layers, {self.config.decoder_layers} decoder layers")
        
        return self.config
    
    def load_tokenizer(self):
        """Load tokenizer (T5Tokenizer with SentencePiece)"""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for tokenizer")
        
        if isinstance(self.model_path, Path):
            self._log(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        else:
            self._log(f"Loading tokenizer from Hugging Face Hub: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self._log(f"Tokenizer loaded: {type(self.tokenizer).__name__}")
        return self.tokenizer
    
    def _load_safetensors(self, path: Path) -> Dict[str, np.ndarray]:
        """Load parameters from safetensors file"""
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors required. Install with: pip install safetensors")
        
        self._log(f"Loading from safetensors: {path}")
        params = {}
        
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                params[key] = f.get_tensor(key)
        
        return params
    
    def _load_pytorch(self, path: Path) -> Dict[str, np.ndarray]:
        """Load parameters from PyTorch checkpoint"""
        if not HAS_TORCH:
            raise ImportError("torch required. Install with: pip install torch")
        
        self._log(f"Loading from PyTorch checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Convert to numpy
        params = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                params[key] = value.detach().cpu().numpy()
            else:
                params[key] = value
        
        return params
    
    def _load_from_hf_model(self) -> Dict[str, np.ndarray]:
        """Load parameters from Hugging Face model"""
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            raise ImportError("transformers and torch required")
        
        self._log(f"Loading model from Hugging Face Hub: {self.model_path}")
        model = FSMTForConditionalGeneration.from_pretrained(self.model_path)
        
        params = {}
        for name, param in model.named_parameters():
            params[name] = param.detach().cpu().numpy()
        
        return params
    
    def load_parameters(self) -> Dict[str, np.ndarray]:
        """Load model parameters from various sources"""
        if isinstance(self.model_path, Path):
            # Try safetensors first, then pytorch_model.bin
            safetensors_path = self.model_path / "model.safetensors"
            pytorch_path = self.model_path / "pytorch_model.bin"
            
            if safetensors_path.exists():
                return self._load_safetensors(safetensors_path)
            elif pytorch_path.exists():
                return self._load_pytorch(pytorch_path)
            else:
                raise FileNotFoundError(
                    f"No model file found in {self.model_path}. "
                    f"Expected 'model.safetensors' or 'pytorch_model.bin'"
                )
        else:
            # Load from Hugging Face Hub
            return self._load_from_hf_model()
    
    def _convert_to_jax_pytree(self, params: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
        """Convert numpy parameters to JAX pytree structure"""
        self._log("Converting parameters to JAX format...")
        
        jax_params = {}
        
        # Organize into nested structure
        for name, value in params.items():
            # Remove 'model.' prefix if present
            if name.startswith('model.'):
                name = name[6:]
            
            # Split by dots to create nested dict
            parts = name.split('.')
            current = jax_params
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Convert to JAX array
            current[parts[-1]] = jnp.array(value)
        
        return jax_params
    
    def _create_scan_map(self, params: Dict) -> Dict:
        """
        Create scan map for associative scan operations
        
        For FSMT/Transformer, we don't use associative scan like RWKV,
        so this returns a dummy map with all 0s (no scan).
        """
        def create_zeros(tree):
            if isinstance(tree, dict):
                return {k: create_zeros(v) for k, v in tree.items()}
            else:
                return 0  # No scan
        
        return create_zeros(params)
    
    def _create_es_map(self, params: Dict, freeze_nonlora: bool = True) -> Dict:
        """
        Create ES map indicating which parameters to evolve
        
        Args:
            params: Parameter pytree
            freeze_nonlora: If True, only evolve attention parameters (LoRA-style)
                          If False, evolve all parameters
        
        Returns:
            ES map with 0 = FULL evolution, 1 = LORA/frozen
        """
        self._log(f"Creating ES map (freeze_nonlora={freeze_nonlora})...")
        
        analyzer = FSMTParameterAnalyzer(freeze_nonlora=freeze_nonlora)
        
        def create_map(tree, prefix=''):
            if isinstance(tree, dict):
                result = {}
                for key, value in tree.items():
                    full_name = f"{prefix}.{key}" if prefix else key
                    result[key] = create_map(value, full_name)
                return result
            else:
                # Determine if this parameter should be evolved
                layer_type, _ = analyzer.analyze_parameter_name(prefix)
                should_evolve_full = analyzer.should_evolve_full(layer_type, prefix)
                return 0 if should_evolve_full else 1
        
        return create_map(params)
    
    def load_model(self, freeze_nonlora: bool = True) -> Tuple[FSMTConfig, Dict, Dict, Dict, Any]:
        """
        Load complete model for ES training
        
        Args:
            freeze_nonlora: Whether to freeze non-LoRA parameters
        
        Returns:
            (config, params, scan_map, es_map, tokenizer)
            - config: FSMTConfig object
            - params: JAX pytree of parameters
            - scan_map: Scan map for associative operations (dummy for transformer)
            - es_map: Evolution strategy map (0=FULL, 1=LORA/frozen)
            - tokenizer: Loaded tokenizer
        """
        # Load all components
        config = self.load_config()
        tokenizer = self.load_tokenizer()
        numpy_params = self.load_parameters()
        
        # Convert to JAX
        params = self._convert_to_jax_pytree(numpy_params)
        
        # Create maps
        scan_map = self._create_scan_map(params)
        es_map = self._create_es_map(params, freeze_nonlora)
        
        # Analyze parameters
        if self.verbose:
            self._log("Analyzing parameter structure...")
            analyzer = FSMTParameterAnalyzer(freeze_nonlora=freeze_nonlora)
            param_infos, _ = analyzer.analyze_params(params)
            analyzer.print_analysis(param_infos)
            
            stats = analyzer.get_evolution_stats(param_infos)
            self._log(f"Total parameters: {stats['total_params']:,}")
            self._log(f"Evolved parameters: {stats['full_params']:,} ({100*stats['full_params']/stats['total_params']:.1f}%)")
        
        return config, params, scan_map, es_map, tokenizer


def load_fsmt_model(
    model_path: Union[str, Path],
    freeze_nonlora: bool = True,
    verbose: bool = True
) -> Tuple[FSMTConfig, Dict, Dict, Dict, Any]:
    """
    Convenience function to load FSMT model
    
    Args:
        model_path: Path to local directory or Hugging Face model name
        freeze_nonlora: Whether to freeze non-LoRA parameters
        verbose: Print loading information
    
    Returns:
        (config, params, scan_map, es_map, tokenizer)
    
    Example:
        >>> # Load from local directory
        >>> config, params, scan_map, es_map, tokenizer = load_fsmt_model(
        ...     "d:/HyperscaleES/user/",
        ...     freeze_nonlora=True
        ... )
        
        >>> # Load from Hugging Face Hub
        >>> config, params, scan_map, es_map, tokenizer = load_fsmt_model(
        ...     "facebook/wmt19-en-de",
        ...     freeze_nonlora=True
        ... )
    """
    loader = FSMTModelLoader(model_path, verbose=verbose)
    return loader.load_model(freeze_nonlora=freeze_nonlora)


# For compatibility with existing code (similar to get_model in llm/auto.py)
def get_fsmt_model(
    model_name: str = "local",
    model_path: Optional[str] = None,
    freeze_nonlora: bool = True,
    verbose: bool = True,
    **kwargs
) -> Tuple[None, Tuple[FSMTConfig, Dict, Dict, Dict], Any]:
    """
    Load FSMT model with interface compatible with existing ES framework
    
    Args:
        model_name: Model identifier ('local' or HuggingFace name)
        model_path: Path to local model directory (required if model_name='local')
        freeze_nonlora: Whether to freeze non-LoRA parameters
        verbose: Print loading information
    
    Returns:
        (None, (config, params, scan_map, es_map), tokenizer)
        Note: First element is None because FSMT doesn't have a forward class like RWKV
    
    Example:
        >>> # Load from local directory
        >>> _, full_params, tokenizer = get_fsmt_model(
        ...     model_name="local",
        ...     model_path="d:/HyperscaleES/user/"
        ... )
        >>> config, params, scan_map, es_map = full_params
    """
    if model_name == "local":
        if model_path is None:
            raise ValueError("model_path required when model_name='local'")
        path = model_path
    else:
        path = model_name
    
    config, params, scan_map, es_map, tokenizer = load_fsmt_model(
        path, freeze_nonlora=freeze_nonlora, verbose=verbose
    )
    
    # Return in format compatible with existing code
    full_params = (config, params, scan_map, es_map)
    
    return None, full_params, tokenizer


if __name__ == "__main__":
    # Test loading
    import sys
    
    print("="*80)
    print("FSMT Model Loader Test")
    print("="*80)
    
    # Example: Load from local directory
    model_path = r"d:\HyperscaleES\user"
    
    if os.path.exists(model_path):
        print(f"\nTesting load from local directory: {model_path}")
        try:
            config, params, scan_map, es_map, tokenizer = load_fsmt_model(
                model_path,
                freeze_nonlora=True,
                verbose=True
            )
            print("\n✅ Model loaded successfully!")
            print(f"Config: {config.encoder_layers} encoder layers, {config.decoder_layers} decoder layers")
            print(f"Tokenizer: {type(tokenizer).__name__}")
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nℹ️  Model path not found: {model_path}")
        print("To test, update the model_path variable to point to your model directory")
    
    print("\n" + "="*80)
