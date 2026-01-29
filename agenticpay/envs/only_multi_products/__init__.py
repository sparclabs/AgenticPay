"""Only Multi-Products Environment

This category includes environments with:
- Single buyer agent
- Multiple products
- Single seller agent
"""

from agenticpay.envs.only_multi_products.Task1_multi_product_negotiation import Task1MultiProductNegotiation
from agenticpay.envs.only_multi_products.Task2_two_product_negotiation import Task2TwoProductNegotiation
from agenticpay.envs.only_multi_products.Task3_five_product_negotiation import Task3FiveProductNegotiation
from agenticpay.envs.only_multi_products.Task4_select_three_from_five_negotiation import Task4SelectThreeFromFiveNegotiation

__all__ = ["Task1MultiProductNegotiation", "Task2TwoProductNegotiation", "Task3FiveProductNegotiation", "Task4SelectThreeFromFiveNegotiation"]

