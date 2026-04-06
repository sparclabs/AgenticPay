"""Task2: Price + Quality + Delivery Negotiation

Agents negotiate on three parameters:
  - price         : continuous (dollars)
  - quality       : ordinal — "Standard" | "Premium" | "Luxury"
  - delivery_days : integer (days until delivery)

New tension vs Task1:
  The buyer has a factory go-live deadline (max_delivery_days). Faster delivery
  costs the seller more (rush fulfilment). Agents must trade off price and speed.
"""

from agenticpay.envs.multi_param.base_multi_param_negotiation import BaseMultiParamNegotiation


class Task2PriceQualityDeliveryNegotiation(BaseMultiParamNegotiation):
    """Task2: Price, quality, and delivery timeline negotiation."""

    ACTIVE_PARAMS = ["price", "quality", "delivery_days"]
