"""Task3 Close to Market Price Negotiation Environment Implementation

This environment is similar to Task1, but designed to test scenarios where
seller_min_price is close to the market price (product_info["price"]) to see
if a deal can be reached.
"""

from __future__ import annotations

from agenticpaygym.envs.single_buyer_product_seller.Task1_basic_price_negotiation import Task1BasicPriceNegotiation


class Task3CloseToMarketPriceNegotiation(Task1BasicPriceNegotiation):
    """Task3 Close to Market Price Negotiation Environment
    
    This environment extends Task1BasicPriceNegotiation to test scenarios
    where the seller's minimum price is close to the market price (product's
    listed price). The negotiation logic remains the same, but this environment
    is specifically designed for testing edge cases where seller's bottom line
    is near the market price.
    """
    
    def __init__(
        self,
        buyer_agent,
        seller_agent,
        max_rounds: int = 20,
        initial_seller_price: float = 100.0,
        buyer_max_price=None,
        seller_min_price=None,
        environment_info=None,
        price_tolerance: float = 1.0,
    ):
        """Initialize Task3 negotiation environment
        
        Args:
            buyer_agent: Buyer Agent
            seller_agent: Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller_price: Initial price offered by seller
            buyer_max_price: Maximum acceptable price for buyer (confidential)
            seller_min_price: Minimum acceptable price for seller (confidential)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
        """
        super().__init__(
            buyer_agent=buyer_agent,
            seller_agent=seller_agent,
            max_rounds=max_rounds,
            initial_seller_price=initial_seller_price,
            buyer_max_price=buyer_max_price,
            seller_min_price=seller_min_price,
            environment_info=environment_info,
            price_tolerance=price_tolerance,
        )

