"""
FSMT (FairSeq Machine Translation) Model Package

JAX implementation of FSMT encoder-decoder transformer for Evolution Strategies.
"""

from .forward import FSMTModel
from .generation import generate_translation, beam_search_generate

__all__ = [
    'FSMTModel',
    'generate_translation',
    'beam_search_generate',
]
