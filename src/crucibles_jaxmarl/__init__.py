"""Crucibles: explicit PRNG, pure JAX transitions and agent-keyed API."""
from .registry import make, registered_envs, game_info

__version__ = '0.1.0'
__all__ = ['make', 'registered_envs', 'game_info']
