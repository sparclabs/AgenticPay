"""Task4: All 5 Parameters Negotiation

Agents negotiate on five parameters simultaneously:
  - price           : continuous (dollars)
  - quality         : ordinal — "Standard" | "Premium" | "Luxury"
  - delivery_days   : integer (days)
  - warranty_months : integer (months)
  - payment         : ordinal — "upfront" | "30-day" | "installments"

New tension vs Task3:
  Payment schedule requires exact-match agreement (no tolerance). The seller
  prefers upfront cash; the buyer prefers installments for cash-flow reasons.
  One party must yield completely, or both converge on "30-day net". This
  creates the highest-complexity negotiation scenario in the suite.
"""

from agenticpay.envs.multi_param.base_multi_param_negotiation import BaseMultiParamNegotiation


class Task4AllParamsNegotiation(BaseMultiParamNegotiation):
    """Task4: Full 5-parameter negotiation (price, quality, delivery, warranty, payment)."""

    ACTIVE_PARAMS = ["price", "quality", "delivery_days", "warranty_months", "payment"]
