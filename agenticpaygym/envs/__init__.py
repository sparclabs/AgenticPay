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
    "Task1BasicPriceNegotiation",
    "Task2ClosePriceNegotiation",
    "Task3CloseToMarketPriceNegotiation",
    "MultiProductNegotiationEnv",
]

# Import environment classes
from agenticpaygym.envs.single_buyer_product_seller.Task1_basic_price_negotiation import Task1BasicPriceNegotiation
from agenticpaygym.envs.single_buyer_product_seller.Task2_close_price_negotiation import Task2ClosePriceNegotiation
from agenticpaygym.envs.single_buyer_product_seller.Task3_close_to_market_price_negotiation import Task3CloseToMarketPriceNegotiation
from agenticpaygym.envs.only_multi_products.multi_product_negotiation_env import MultiProductNegotiationEnv

# Automatically register all environments
register(
    id="Task1_basic_price_negotiation-v0",
    entry_point="agenticpaygym.envs.single_buyer_product_seller.Task1_basic_price_negotiation:Task1BasicPriceNegotiation",
    max_episode_steps=20,
)

register(
    id="Task2_close_price_negotiation-v0",
    entry_point="agenticpaygym.envs.single_buyer_product_seller.Task2_close_price_negotiation:Task2ClosePriceNegotiation",
    max_episode_steps=20,
)

register(
    id="Task3_close_to_market_price_negotiation-v0",
    entry_point="agenticpaygym.envs.single_buyer_product_seller.Task3_close_to_market_price_negotiation:Task3CloseToMarketPriceNegotiation",
    max_episode_steps=20,
)

register(
    id="MultiProductNegotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_products.multi_product_negotiation_env:MultiProductNegotiationEnv",
    max_episode_steps=20,
)

