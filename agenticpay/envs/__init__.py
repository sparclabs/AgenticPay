"""Environment Registration Module

References Gymnasium's design, provides environment registration and creation functionality.
"""

from agenticpay.envs.registration import (
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
    "Task15QuantityDiscountNegotiation",
    "Task1MultiProductNegotiation",
    "Task2TwoProductNegotiation",
    "Task3FiveProductNegotiation",
    "Task4SelectThreeFromFiveNegotiation",
    "Task1ParallelTwoSellerNegotiation",
    "Task2ParallelThreeSellerNegotiation",
    "Task3SequentialTwoSellerNegotiation",
    "Task4SequentialThreeSellerNegotiation",
    "Task15ParallelTwoBuyerQuantityDiscountNegotiation",
    "Task15ParallelTwoSellerQuantityDiscountNegotiation",
    "Task15MultiProductQuantityDiscountNegotiation",
    "Task15TwoBuyerTwoSellerQuantityDiscountNegotiation",
    "Task15TwoBuyerMultiProductQuantityDiscountNegotiation",
    "Task15MultiProductTwoSellerQuantityDiscountNegotiation",
    "Task15TwoBuyerTwoSellerMultiProductQuantityDiscountNegotiation",
]

# Import environment classes
from agenticpay.envs.single_buyer_product_seller.Task1_basic_price_negotiation import Task1BasicPriceNegotiation
from agenticpay.envs.single_buyer_product_seller.Task2_close_price_negotiation import Task2ClosePriceNegotiation
from agenticpay.envs.single_buyer_product_seller.Task3_close_to_market_price_negotiation import Task3CloseToMarketPriceNegotiation
from agenticpay.envs.single_buyer_product_seller.Task15_quantity_discount_negotiation import Task15QuantityDiscountNegotiation
from agenticpay.envs.only_multi_products.Task1_multi_product_negotiation import Task1MultiProductNegotiation
from agenticpay.envs.only_multi_products.Task2_two_product_negotiation import Task2TwoProductNegotiation
from agenticpay.envs.only_multi_products.Task3_five_product_negotiation import Task3FiveProductNegotiation
from agenticpay.envs.only_multi_products.Task4_select_three_from_five_negotiation import Task4SelectThreeFromFiveNegotiation
from agenticpay.envs.only_multi_seller.Task1_parallel_two_seller_negotiation import Task1ParallelTwoSellerNegotiation
from agenticpay.envs.only_multi_seller.Task2_parallel_three_seller_negotiation import Task2ParallelThreeSellerNegotiation
from agenticpay.envs.only_multi_seller.Task3_sequential_two_seller_negotiation import Task3SequentialTwoSellerNegotiation
from agenticpay.envs.only_multi_seller.Task4_sequential_three_seller_negotiation import Task4SequentialThreeSellerNegotiation
from agenticpay.envs.only_multi_buyer.Task15_quantity_discount_negotiation import Task15ParallelTwoBuyerQuantityDiscountNegotiation
from agenticpay.envs.only_multi_seller.Task15_quantity_discount_negotiation import Task15ParallelTwoSellerQuantityDiscountNegotiation
from agenticpay.envs.only_multi_products.Task15_quantity_discount_negotiation import Task15MultiProductQuantityDiscountNegotiation
from agenticpay.envs.multi_buyer_multi_seller.Task15_quantity_discount_negotiation import Task15TwoBuyerTwoSellerQuantityDiscountNegotiation
from agenticpay.envs.multi_buyer_multi_products.Task15_quantity_discount_negotiation import Task15TwoBuyerMultiProductQuantityDiscountNegotiation
from agenticpay.envs.multi_products_multi_seller.Task15_quantity_discount_negotiation import Task15MultiProductTwoSellerQuantityDiscountNegotiation
from agenticpay.envs.multi_buyer_multi_products_multi_seller.Task15_quantity_discount_negotiation import Task15TwoBuyerTwoSellerMultiProductQuantityDiscountNegotiation

# Automatically register all environments
register(
    id="Task1_basic_price_negotiation-v0",
    entry_point="agenticpay.envs.single_buyer_product_seller.Task1_basic_price_negotiation:Task1BasicPriceNegotiation",
    max_episode_steps=20,
)

register(
    id="Task15_quantity_discount_negotiation-v0",
    entry_point="agenticpay.envs.single_buyer_product_seller.Task15_quantity_discount_negotiation:Task15QuantityDiscountNegotiation",
    max_episode_steps=20,
)

register(
    id="Task2_close_price_negotiation-v0",
    entry_point="agenticpay.envs.single_buyer_product_seller.Task2_close_price_negotiation:Task2ClosePriceNegotiation",
    max_episode_steps=20,
)

register(
    id="Task3_close_to_market_price_negotiation-v0",
    entry_point="agenticpay.envs.single_buyer_product_seller.Task3_close_to_market_price_negotiation:Task3CloseToMarketPriceNegotiation",
    max_episode_steps=20,
)

register(
    id="Task1_multi_product_negotiation-v0",
    entry_point="agenticpay.envs.only_multi_products.Task1_multi_product_negotiation:Task1MultiProductNegotiation",
    max_episode_steps=20,
)

register(
    id="Task2_two_product_negotiation-v0",
    entry_point="agenticpay.envs.only_multi_products.Task2_two_product_negotiation:Task2TwoProductNegotiation",
    max_episode_steps=20,
)

register(
    id="Task3_five_product_negotiation-v0",
    entry_point="agenticpay.envs.only_multi_products.Task3_five_product_negotiation:Task3FiveProductNegotiation",
    max_episode_steps=20,
)

register(
    id="Task4_select_three_from_five_negotiation-v0",
    entry_point="agenticpay.envs.only_multi_products.Task4_select_three_from_five_negotiation:Task4SelectThreeFromFiveNegotiation",
    max_episode_steps=20,
)

register(
    id="Task1_parallel_two_seller_negotiation-v0",
    entry_point="agenticpay.envs.only_multi_seller.Task1_parallel_two_seller_negotiation:Task1ParallelTwoSellerNegotiation",
    max_episode_steps=20,
)

register(
    id="Task2_parallel_three_seller_negotiation-v0",
    entry_point="agenticpay.envs.only_multi_seller.Task2_parallel_three_seller_negotiation:Task2ParallelThreeSellerNegotiation",
    max_episode_steps=20,
)

register(
    id="Task3_sequential_two_seller_negotiation-v0",
    entry_point="agenticpay.envs.only_multi_seller.Task3_sequential_two_seller_negotiation:Task3SequentialTwoSellerNegotiation",
    max_episode_steps=20,
)

register(
    id="Task4_sequential_three_seller_negotiation-v0",
    entry_point="agenticpay.envs.only_multi_seller.Task4_sequential_three_seller_negotiation:Task4SequentialThreeSellerNegotiation",
    max_episode_steps=20,
)

register(
    id="Task15_quantity_discount_negotiation_multi_buyer-v0",
    entry_point="agenticpay.envs.only_multi_buyer.Task15_quantity_discount_negotiation:Task15ParallelTwoBuyerQuantityDiscountNegotiation",
    max_episode_steps=20,
)

register(
    id="Task15_quantity_discount_negotiation_multi_seller-v0",
    entry_point="agenticpay.envs.only_multi_seller.Task15_quantity_discount_negotiation:Task15ParallelTwoSellerQuantityDiscountNegotiation",
    max_episode_steps=20,
)

register(
    id="Task15_quantity_discount_negotiation_multi_products-v0",
    entry_point="agenticpay.envs.only_multi_products.Task15_quantity_discount_negotiation:Task15MultiProductQuantityDiscountNegotiation",
    max_episode_steps=20,
)

register(
    id="Task15_quantity_discount_negotiation_multi_buyer_multi_seller-v0",
    entry_point="agenticpay.envs.multi_buyer_multi_seller.Task15_quantity_discount_negotiation:Task15TwoBuyerTwoSellerQuantityDiscountNegotiation",
    max_episode_steps=20,
)

register(
    id="Task15_quantity_discount_negotiation_multi_buyer_multi_products-v0",
    entry_point="agenticpay.envs.multi_buyer_multi_products.Task15_quantity_discount_negotiation:Task15TwoBuyerMultiProductQuantityDiscountNegotiation",
    max_episode_steps=20,
)

register(
    id="Task15_quantity_discount_negotiation_multi_products_multi_seller-v0",
    entry_point="agenticpay.envs.multi_products_multi_seller.Task15_quantity_discount_negotiation:Task15MultiProductTwoSellerQuantityDiscountNegotiation",
    max_episode_steps=20,
)

register(
    id="Task15_quantity_discount_negotiation_multi_buyer_multi_products_multi_seller-v0",
    entry_point="agenticpay.envs.multi_buyer_multi_products_multi_seller.Task15_quantity_discount_negotiation:Task15TwoBuyerTwoSellerMultiProductQuantityDiscountNegotiation",
    max_episode_steps=20,
)
