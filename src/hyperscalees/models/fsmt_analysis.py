"""
FSMT Model Structure Analysis Tool

This module analyzes the parameter structure of FSMT models
and determines which parameters should be evolved during ES training.
"""

import jax.numpy as jnp
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

@dataclass
class ParameterInfo:
    """Information about a single parameter"""
    name: str
    shape: Tuple[int, ...]
    num_params: int
    layer_type: str  # 'embedding', 'encoder', 'decoder', 'attention', 'ffn', 'output'
    layer_num: int  # Which layer (0-indexed, -1 for non-layer params)
    evolve_full: bool  # Whether to evolve this fully (vs LoRA)
    
    def __repr__(self):
        evolve_str = "FULL" if self.evolve_full else "LORA"
        return f"{self.name:60s} {str(self.shape):20s} {self.num_params:>12,d} [{self.layer_type:12s}] [{evolve_str}]"


class FSMTParameterAnalyzer:
    """Analyzes FSMT model parameters for Evolution Strategy"""
    
    def __init__(self, freeze_nonlora: bool = True):
        """
        Args:
            freeze_nonlora: If True, only evolve LoRA-style parameters (attention projections)
                          If False, evolve all parameters
        """
        self.freeze_nonlora = freeze_nonlora
        
    def analyze_parameter_name(self, name: str) -> Tuple[str, int]:
        """
        Determine layer type and layer number from parameter name
        
        Returns:
            (layer_type, layer_num)
        """
        # Embedding layers
        if 'embed_tokens' in name or 'embed_positions' in name:
            return 'embedding', -1
        
        # Encoder layers
        if 'encoder.layers' in name:
            layer_num = self._extract_layer_num(name, 'encoder.layers')
            if 'self_attn' in name:
                return 'encoder_attn', layer_num
            elif 'fc1' in name or 'fc2' in name:
                return 'encoder_ffn', layer_num
            else:
                return 'encoder_norm', layer_num
        
        # Decoder layers
        if 'decoder.layers' in name:
            layer_num = self._extract_layer_num(name, 'decoder.layers')
            if 'self_attn' in name:
                return 'decoder_self_attn', layer_num
            elif 'encoder_attn' in name or 'cross_attn' in name:
                return 'decoder_cross_attn', layer_num
            elif 'fc1' in name or 'fc2' in name:
                return 'decoder_ffn', layer_num
            else:
                return 'decoder_norm', layer_num
        
        # Output/head layers
        if 'lm_head' in name or 'output' in name:
            return 'output_head', -1
        
        # Layernorm
        if 'layernorm' in name or 'layer_norm' in name:
            return 'layernorm', -1
        
        return 'other', -1
    
    def _extract_layer_num(self, name: str, prefix: str) -> int:
        """Extract layer number from parameter name"""
        try:
            start = name.index(prefix) + len(prefix) + 1
            end = name.index('.', start)
            return int(name[start:end])
        except (ValueError, IndexError):
            return -1
    
    def should_evolve_full(self, layer_type: str, name: str) -> bool:
        """
        Determine if parameter should be evolved fully or LoRA-style
        
        Strategy:
        - If freeze_nonlora=False: evolve all parameters fully
        - If freeze_nonlora=True: only evolve attention projections (Q, K, V, O)
        """
        if not self.freeze_nonlora:
            return True  # Evolve everything
        
        # Only evolve attention projection matrices (LoRA-style)
        attention_keywords = [
            'q_proj', 'k_proj', 'v_proj', 'out_proj',  # Self-attention
            'query', 'key', 'value', 'output',  # Alternative naming
        ]
        
        is_attention = any(keyword in name.lower() for keyword in attention_keywords)
        
        return is_attention
    
    def analyze_params(self, params: Dict[str, Any]) -> Tuple[List[ParameterInfo], Dict]:
        """
        Analyze parameter structure
        
        Args:
            params: PyTree of model parameters
            
        Returns:
            (param_info_list, es_map)
            - param_info_list: List of ParameterInfo objects
            - es_map: PyTree with same structure as params, values are 0 (FULL) or 1 (LORA)
        """
        param_infos = []
        es_map = {}
        
        def analyze_recursive(tree, prefix=''):
            nonlocal param_infos, es_map
            
            if isinstance(tree, dict):
                result = {}
                for key, value in tree.items():
                    full_name = f"{prefix}.{key}" if prefix else key
                    result[key] = analyze_recursive(value, full_name)
                return result
            
            elif hasattr(tree, 'shape'):  # JAX array
                name = prefix
                shape = tree.shape
                num_params = int(jnp.prod(jnp.array(shape)))
                
                layer_type, layer_num = self.analyze_parameter_name(name)
                evolve_full = self.should_evolve_full(layer_type, name)
                
                info = ParameterInfo(
                    name=name,
                    shape=shape,
                    num_params=num_params,
                    layer_type=layer_type,
                    layer_num=layer_num,
                    evolve_full=evolve_full
                )
                param_infos.append(info)
                
                # ES map: 0 = FULL evolution, 1 = LORA/frozen
                # Note: In the original code, FULL=0, LORA=1
                return 0 if evolve_full else 1
            
            else:
                return tree
        
        es_map = analyze_recursive(params)
        
        return param_infos, es_map
    
    def print_analysis(self, param_infos: List[ParameterInfo]):
        """Print detailed parameter analysis"""
        print("\n" + "="*120)
        print("FSMT PARAMETER ANALYSIS")
        print("="*120)
        print(f"{'Parameter Name':<60} {'Shape':<20} {'Num Params':>12} {'Layer Type':>14} {'Evolution'}")
        print("-"*120)
        
        total_params = 0
        full_params = 0
        lora_params = 0
        
        # Group by layer type
        by_type = {}
        for info in param_infos:
            if info.layer_type not in by_type:
                by_type[info.layer_type] = []
            by_type[info.layer_type].append(info)
            
            total_params += info.num_params
            if info.evolve_full:
                full_params += info.num_params
            else:
                lora_params += info.num_params
        
        # Print grouped
        for layer_type in sorted(by_type.keys()):
            print(f"\n{'─'*120}")
            print(f"{layer_type.upper()}")
            print(f"{'─'*120}")
            for info in by_type[layer_type]:
                print(info)
        
        print("\n" + "="*120)
        print(f"TOTAL PARAMETERS: {total_params:,}")
        print(f"  - Full Evolution: {full_params:,} ({100*full_params/total_params:.1f}%)")
        print(f"  - LoRA/Frozen:    {lora_params:,} ({100*lora_params/total_params:.1f}%)")
        print("="*120 + "\n")
    
    def get_evolution_stats(self, param_infos: List[ParameterInfo]) -> Dict[str, int]:
        """Get statistics about parameters"""
        stats = {
            'total_params': 0,
            'full_params': 0,
            'lora_params': 0,
            'embedding_params': 0,
            'encoder_params': 0,
            'decoder_params': 0,
            'attention_params': 0,
            'ffn_params': 0,
        }
        
        for info in param_infos:
            stats['total_params'] += info.num_params
            
            if info.evolve_full:
                stats['full_params'] += info.num_params
            else:
                stats['lora_params'] += info.num_params
            
            if 'embedding' in info.layer_type:
                stats['embedding_params'] += info.num_params
            elif 'encoder' in info.layer_type:
                stats['encoder_params'] += info.num_params
            elif 'decoder' in info.layer_type:
                stats['decoder_params'] += info.num_params
            
            if 'attn' in info.layer_type:
                stats['attention_params'] += info.num_params
            elif 'ffn' in info.layer_type:
                stats['ffn_params'] += info.num_params
        
        return stats


def create_es_map_for_fsmt(config, freeze_nonlora: bool = True) -> Dict[str, int]:
    """
    Create ES map for FSMT model based on configuration
    
    This is a placeholder that will be properly implemented once we have actual model loading
    
    Args:
        config: FSMTConfig object
        freeze_nonlora: Whether to freeze non-LoRA parameters
        
    Returns:
        es_map: Dictionary mapping parameter names to evolution type (0=FULL, 1=LORA)
    """
    # TODO: This will be implemented properly in Task 2 when we load actual models
    print(f"Creating ES map for FSMT model:")
    print(f"  - Encoder layers: {config.encoder_layers}")
    print(f"  - Decoder layers: {config.decoder_layers}")
    print(f"  - Freeze non-LoRA: {freeze_nonlora}")
    
    return {}


if __name__ == "__main__":
    # Example usage when we have actual parameters
    print("FSMT Parameter Analyzer")
    print("This tool will be used to analyze model parameters once loaded in Task 2")
    
    from .fsmt_config import FSMT_BASE
    
    analyzer = FSMTParameterAnalyzer(freeze_nonlora=True)
    print(f"\nConfiguration: FSMT Base")
    print(f"  Model dimension: {FSMT_BASE.d_model}")
    print(f"  Encoder layers: {FSMT_BASE.encoder_layers}")
    print(f"  Decoder layers: {FSMT_BASE.decoder_layers}")
    print(f"  Attention heads: {FSMT_BASE.encoder_attention_heads}")
    print(f"\nReady to analyze parameters once model is loaded.")
