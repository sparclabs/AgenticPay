"""Task1: Price + Quality Negotiation

Agents negotiate on two parameters:
  - price   : continuous (dollars)
  - quality : ordinal — "Standard" | "Premium" | "Luxury"

This is the simplest multi-parameter task, introducing the quality/price trade-off:
the buyer wants higher quality at a lower price; the seller wants a higher price
for the lowest quality tier they can offer.
"""

from agenticpay.envs.multi_param.base_multi_param_negotiation import BaseMultiParamNegotiation


class Task1PriceQualityNegotiation(BaseMultiParamNegotiation):
    """Task1: Price and quality negotiation."""

    ACTIVE_PARAMS = ["price", "quality"]
