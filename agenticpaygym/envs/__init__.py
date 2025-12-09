"""Environment Registration Module

References Gymnasium's design, provides environment registration and creation functionality.
"""

from agenticpaygym.envs.registration import (
    register,
    make,
    spec,
    pprint_registry,
    registry,
    EnvSpec,
)

__all__ = [
    "register",
    "make",
    "spec",
    "pprint_registry",
    "registry",
    "EnvSpec",
    "NegotiationEnv",
    "MultiProductNegotiationEnv",
]

# Import environment classes
from agenticpaygym.envs.negotiation_env import NegotiationEnv
from agenticpaygym.envs.multi_product_negotiation_env import MultiProductNegotiationEnv

# Automatically register all environments
register(
    id="Negotiation-v0",
    entry_point="agenticpaygym.envs.negotiation_env:NegotiationEnv",
    max_episode_steps=20,
)

register(
    id="MultiProductNegotiation-v0",
    entry_point="agenticpaygym.envs.multi_product_negotiation_env:MultiProductNegotiationEnv",
    max_episode_steps=20,
)

