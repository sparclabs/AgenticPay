"""Task3: Price + Quality + Delivery + Warranty Negotiation

Agents negotiate on four parameters:
  - price           : continuous (dollars)
  - quality         : ordinal — "Standard" | "Premium" | "Luxury"
  - delivery_days   : integer (days)
  - warranty_months : integer (months of warranty coverage)

New tension vs Task2:
  Warranty shifts liability. The buyer wants long coverage to reduce maintenance
  risk; the seller wants minimal obligation (shorter warranty = lower support cost).
  Agents can offer faster delivery in exchange for shorter warranty, etc.
"""

from agenticpay.envs.multi_param.base_multi_param_negotiation import BaseMultiParamNegotiation


class Task3PriceQualityDeliveryWarrantyNegotiation(BaseMultiParamNegotiation):
    """Task3: Price, quality, delivery, and warranty negotiation."""

    ACTIVE_PARAMS = ["price", "quality", "delivery_days", "warranty_months"]
