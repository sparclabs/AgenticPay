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
    "Task1MultiProductNegotiation",
    "Task2TwoProductNegotiation",
    "Task3FiveProductNegotiation",
    "Task4SelectThreeFromFiveNegotiation",
    "Task1ParallelTwoSellerNegotiation",
    "Task2ParallelThreeSellerNegotiation",
    "Task3SequentialTwoSellerNegotiation",
    "Task4SequentialThreeSellerNegotiation",
]

# Import environment classes
from agenticpaygym.envs.single_buyer_product_seller.Task1_basic_price_negotiation import Task1BasicPriceNegotiation
from agenticpaygym.envs.single_buyer_product_seller.Task2_close_price_negotiation import Task2ClosePriceNegotiation
from agenticpaygym.envs.single_buyer_product_seller.Task3_close_to_market_price_negotiation import Task3CloseToMarketPriceNegotiation
from agenticpaygym.envs.only_multi_products.Task1_multi_product_negotiation import Task1MultiProductNegotiation
from agenticpaygym.envs.only_multi_products.Task2_two_product_negotiation import Task2TwoProductNegotiation
from agenticpaygym.envs.only_multi_products.Task3_five_product_negotiation import Task3FiveProductNegotiation
from agenticpaygym.envs.only_multi_products.Task4_select_three_from_five_negotiation import Task4SelectThreeFromFiveNegotiation
from agenticpaygym.envs.only_multi_seller.Task1_parallel_two_seller_negotiation import Task1ParallelTwoSellerNegotiation
from agenticpaygym.envs.only_multi_seller.Task2_parallel_three_seller_negotiation import Task2ParallelThreeSellerNegotiation
from agenticpaygym.envs.only_multi_seller.Task3_sequential_two_seller_negotiation import Task3SequentialTwoSellerNegotiation
from agenticpaygym.envs.only_multi_seller.Task4_sequential_three_seller_negotiation import Task4SequentialThreeSellerNegotiation

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
    id="Task1_multi_product_negotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_products.Task1_multi_product_negotiation:Task1MultiProductNegotiation",
    max_episode_steps=20,
)

register(
    id="Task2_two_product_negotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_products.Task2_two_product_negotiation:Task2TwoProductNegotiation",
    max_episode_steps=20,
)

register(
    id="Task3_five_product_negotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_products.Task3_five_product_negotiation:Task3FiveProductNegotiation",
    max_episode_steps=20,
)

register(
    id="Task4_select_three_from_five_negotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_products.Task4_select_three_from_five_negotiation:Task4SelectThreeFromFiveNegotiation",
    max_episode_steps=20,
)

register(
    id="Task1_parallel_two_seller_negotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_seller.Task1_parallel_two_seller_negotiation:Task1ParallelTwoSellerNegotiation",
    max_episode_steps=20,
)

register(
    id="Task2_parallel_three_seller_negotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_seller.Task2_parallel_three_seller_negotiation:Task2ParallelThreeSellerNegotiation",
    max_episode_steps=20,
)

register(
    id="Task3_sequential_two_seller_negotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_seller.Task3_sequential_two_seller_negotiation:Task3SequentialTwoSellerNegotiation",
    max_episode_steps=20,
)

register(
    id="Task4_sequential_three_seller_negotiation-v0",
    entry_point="agenticpaygym.envs.only_multi_seller.Task4_sequential_three_seller_negotiation:Task4SequentialThreeSellerNegotiation",
    max_episode_steps=20,
)

