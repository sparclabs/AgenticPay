"""Base Multi-Parameter Negotiation Environment

Shared logic for all multi-param negotiation tasks (Task1–Task4).
Subclasses set ACTIVE_PARAMS to control which parameters are negotiated.

Parameters available:
  - "price"           : continuous (dollars)
  - "quality"         : ordinal — "Standard" | "Premium" | "Luxury"
  - "delivery_days"   : integer (days)
  - "warranty_months" : integer (months)
  - "payment"         : ordinal — "upfront" | "30-day" | "installments"

Agreement logic (active params only):
  - price:           seller_price <= buyer_price
  - quality:         QUALITY_ORDER[seller] >= QUALITY_ORDER[buyer]  (seller meets buyer's floor)
  - delivery_days:   seller_days <= buyer_days                       (seller within buyer's deadline)
  - warranty_months: seller_months >= buyer_months                   (seller meets buyer's floor)
  - payment:         seller_payment == buyer_payment                 (exact match)

Agreed deal = seller's offer (consistent with existing codebase convention).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory


# ---------------------------------------------------------------------------
# Ordinal value mappings
# ---------------------------------------------------------------------------

QUALITY_ORDER: Dict[str, int] = {"Standard": 0, "Premium": 1, "Luxury": 2}
PAYMENT_ORDER: Dict[str, int] = {"upfront": 0, "30-day": 1, "installments": 2}

# Regex patterns for extracting each field from the offer tag body
FIELD_PATTERNS: Dict[str, str] = {
    "price":            r'PRICE\(\$([\d,]+\.?\d*)\)',
    "quality":          r'QUALITY\((Standard|Premium|Luxury)\)',
    "delivery_days":    r'DELIVERY\((\d+)\s*days?\)',
    "warranty_months":  r'WARRANTY\((\d+)\s*months?\)',
    "payment":          r'PAYMENT\((upfront|30-day|installments)\)',
}

BUYER_OFFER_PATTERN  = r'###\s*BUYER_OFFER:\s*(.*?)\s*###'
SELLER_OFFER_PATTERN = r'###\s*SELLER_OFFER:\s*(.*?)\s*###'


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MultiParamOffer:
    """Holds a negotiating party's current offer across all parameters."""
    price: Optional[float] = None
    quality: Optional[str] = None
    delivery_days: Optional[int] = None
    warranty_months: Optional[int] = None
    payment: Optional[str] = None

    def is_complete(self, active_params: List[str]) -> bool:
        """True only when every active param has been filled."""
        return all(getattr(self, p) is not None for p in active_params)

    def to_dict(self, active_params: List[str]) -> Dict[str, Any]:
        return {p: getattr(self, p) for p in active_params}


@dataclass
class ParamPreferences:
    """Each party's reservation values and utility weights.

    Reservation values are hard limits — a deal that crosses them is rejected.
    Utility weights are used only for reward computation; never revealed to the
    counterpart.

    Buyer interpretation:
        price_limit        = maximum price willing to pay
        min_quality        = minimum acceptable quality level
        max_delivery_days  = latest acceptable delivery date (days from now)
        min_warranty_months= minimum acceptable warranty coverage
        preferred_payment  = most-preferred payment schedule

    Seller interpretation:
        price_limit        = minimum acceptable price
        min_quality        = maximum quality tier they will offer (cost ceiling)
        max_delivery_days  = minimum lead time they need (earliest they can ship)
        min_warranty_months= minimum warranty they will commit to
        preferred_payment  = most-preferred payment schedule
    """
    price_limit: float
    min_quality: str
    max_delivery_days: int
    min_warranty_months: int
    preferred_payment: str

    # Utility weights (should sum to ~1.0; renormalized to active params at runtime)
    price_weight: float = 0.40
    quality_weight: float = 0.25
    delivery_weight: float = 0.15
    warranty_weight: float = 0.10
    payment_weight: float = 0.10

    def weight_for(self, param: str) -> float:
        mapping = {
            "price":            self.price_weight,
            "quality":          self.quality_weight,
            "delivery_days":    self.delivery_weight,
            "warranty_months":  self.warranty_weight,
            "payment":          self.payment_weight,
        }
        return mapping[param]


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------

class BaseMultiParamNegotiation(BaseEnv):
    """Abstract base for multi-parameter negotiation environments.

    Subclasses must set ACTIVE_PARAMS as a class-level attribute.
    """

    ACTIVE_PARAMS: List[str] = []  # overridden by subclasses

    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller_agent: BaseAgent,
        buyer_preferences: ParamPreferences,
        seller_preferences: ParamPreferences,
        scenario_params: Dict[str, Any],
        max_rounds: int = 20,
        environment_info: Optional[Dict[str, Any]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        gamma: float = 0.99,
        deal_score_weight: float = 30.0,
        quality_score_weight: float = 55.0,
        efficiency_score_weight: float = 15.0,
        failure_penalty_weight: float = 15.0,
        buyer_deal_weight: float = 30.0,
        buyer_utility_weight: float = 55.0,
        buyer_efficiency_weight: float = 15.0,
        buyer_failure_penalty_weight: float = 15.0,
        seller_deal_weight: float = 30.0,
        seller_utility_weight: float = 55.0,
        seller_efficiency_weight: float = 15.0,
        seller_failure_penalty_weight: float = 15.0,
    ):
        """
        Args:
            buyer_agent: Buyer agent instance.
            seller_agent: Seller agent instance.
            buyer_preferences: Buyer's reservation values and utility weights.
            seller_preferences: Seller's reservation values and utility weights.
            scenario_params: Feasible ranges for all parameters:
                {
                    "price_range": (seller_min, buyer_max),
                    "quality_options": ["Standard", "Premium", "Luxury"],
                    "delivery_range": (min_days, max_days),
                    "warranty_range": (min_months, max_months),
                    "payment_options": ["upfront", "30-day", "installments"],
                }
            max_rounds: Maximum negotiation rounds before timeout.
            environment_info: Contextual info passed to agents (season, region, etc.)
            reward_weights: Override default reward weights
                {"buyer_utility": 1.0, "seller_utility": 1.0, "time_cost": 0.1}
            gamma: Discount factor for score calculations (controls efficiency penalty).
            deal_score_weight: D — weight for deal bonus in GlobalScore.
            quality_score_weight: W — weight for quality/utility component in GlobalScore.
            efficiency_score_weight: E — weight for efficiency bonus in GlobalScore.
            failure_penalty_weight: F — penalty magnitude on negotiation failure.
            buyer_deal_weight / buyer_utility_weight / buyer_efficiency_weight: BuyerScore weights.
            seller_deal_weight / seller_utility_weight / seller_efficiency_weight: SellerScore weights.
        """
        super().__init__()

        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.buyer_preferences = buyer_preferences
        self.seller_preferences = seller_preferences
        self.scenario_params = scenario_params
        self.max_rounds = max_rounds
        self.environment_info = environment_info or {}

        default_reward_weights = {
            "buyer_utility":  1.0,
            "seller_utility": 1.0,
            "time_cost":      0.1,
        }
        if reward_weights:
            default_reward_weights.update(reward_weights)
        self.reward_weights = default_reward_weights

        self.gamma = gamma
        self.deal_score_weight = deal_score_weight
        self.quality_score_weight = quality_score_weight
        self.efficiency_score_weight = efficiency_score_weight
        self.failure_penalty_weight = failure_penalty_weight
        self.buyer_deal_weight = buyer_deal_weight
        self.buyer_utility_weight = buyer_utility_weight
        self.buyer_efficiency_weight = buyer_efficiency_weight
        self.buyer_failure_penalty_weight = buyer_failure_penalty_weight
        self.seller_deal_weight = seller_deal_weight
        self.seller_utility_weight = seller_utility_weight
        self.seller_efficiency_weight = seller_efficiency_weight
        self.seller_failure_penalty_weight = seller_failure_penalty_weight

        # Runtime state (populated by reset())
        self.memory = ConversationMemory()
        self.current_round: int = 0
        self.buyer_offer = MultiParamOffer()
        self.seller_offer = MultiParamOffer()
        self.agreed_offer: Optional[MultiParamOffer] = None
        self.negotiation_info = NegotiationInfo()
        self.current_product_name: str = ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and initialize agents.

        Args:
            user_requirement: What the buyer is looking for.
            product_info: Product being negotiated.
            user_profile: Optional buyer persona description.

        Returns:
            (observation, info)
        """
        self.memory.clear()
        self.current_round = 0
        self.buyer_offer = MultiParamOffer()
        self.seller_offer = MultiParamOffer()
        self.agreed_offer = None
        self.negotiation_info = NegotiationInfo()
        self.current_product_name = (product_info or {}).get("name", "Unknown Product")

        negotiation_params = {
            "active_params":    self.ACTIVE_PARAMS,
            "quality_options":  self.scenario_params.get("quality_options", ["Standard", "Premium", "Luxury"]),
            "payment_options":  self.scenario_params.get("payment_options", ["upfront", "30-day", "installments"]),
            "delivery_range":   self.scenario_params.get("delivery_range", (1, 60)),
            "warranty_range":   self.scenario_params.get("warranty_range", (1, 48)),
            "price_range":      self.scenario_params.get("price_range", (0.0, 999999.0)),
        }

        buyer_context = {
            "user_requirement":    user_requirement,
            "product_info":        product_info or {},
            "user_profile":        user_profile,
            "environment_info":    self.environment_info,
            "negotiation_params":  negotiation_params,
            "max_price":           self.buyer_preferences.price_limit,
        }
        self.buyer_agent.initialize(buyer_context)

        seller_context = {
            "product_info":        product_info or {},
            "environment_info":    self.environment_info,
            "negotiation_params":  negotiation_params,
            "initial_price":       self.scenario_params.get("price_range", (0, 0))[1],
            "min_price":           self.seller_preferences.price_limit,
        }
        self.seller_agent.initialize(seller_context)

        return self._get_observation(), self._get_info()

    def step(
        self,
        buyer_action: Optional[str] = None,
        seller_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation round.

        Args:
            buyer_action:  Buyer's response text.
            seller_action: Seller's response text.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if buyer_action is not None:
            self.memory.add_message("buyer", buyer_action, self.current_round)
            offer = self._extract_offer(buyer_action, "buyer")
            self._merge_offer(self.buyer_offer, offer)
            self.negotiation_info.buyer_price = self.buyer_offer.price

        if seller_action is not None:
            self.memory.add_message("seller", seller_action, self.current_round)
            offer = self._extract_offer(seller_action, "seller")
            self._merge_offer(self.seller_offer, offer)
            self.negotiation_info.seller_price = self.seller_offer.price

        terminated = truncated = False
        reward = seller_reward = buyer_reward = 0.0

        if self._check_agreement():
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            self.agreed_offer = copy.deepcopy(self.seller_offer)
            if self.agreed_offer.price is not None:
                self.negotiation_info.current_price = self.agreed_offer.price
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            seller_reward = self._calculate_seller_reward()
            buyer_reward = self._calculate_buyer_reward()

        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            seller_reward = self._calculate_seller_reward()
            buyer_reward = self._calculate_buyer_reward()

        else:
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round

        step_seller_reward = self._calculate_step_seller_reward()
        step_buyer_reward  = self._calculate_step_buyer_reward()

        observation = self._get_observation()
        info = self._get_info()
        info["step_seller_reward"] = step_seller_reward
        info["step_buyer_reward"]  = step_buyer_reward

        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["seller_reward"] = seller_reward
            info["buyer_reward"]  = buyer_reward
            info["global_score"]  = self._calculate_global_score(print_details=False)
            info["buyer_score"]   = self._calculate_buyer_score(print_details=False)
            info["seller_score"]  = self._calculate_seller_score(print_details=False)
            if self.agreed_offer:
                info["agreed_offer"] = self.agreed_offer.to_dict(self.ACTIVE_PARAMS)

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """Print per-round offer table and agreement status."""
        lines = []

        lines.append(f"\n{'='*60}")
        lines.append(f"Product: {self.current_product_name}")
        lines.append(f"Active Parameters: {', '.join(self.ACTIVE_PARAMS)}")
        lines.append(f"{'='*60}")

        history = self.memory.get_history()
        if history:
            round_to_display = self.current_round - 1 if self.current_round > 0 else 0
            round_messages = [m for m in history if m["round"] == round_to_display]
            if round_messages:
                lines.append(f"\n{'='*60}")
                lines.append(f"Round {self.current_round} - Negotiation Output")
                lines.append(f"{'='*60}")
                buyer_msg = next((m for m in round_messages if m["role"] == "buyer"), None)
                if buyer_msg:
                    lines.append(f"\n[BUYER Output]:")
                    lines.append(f"  {buyer_msg['content']}")
                seller_msg = next((m for m in round_messages if m["role"] == "seller"), None)
                if seller_msg:
                    lines.append(f"\n[SELLER Output]:")
                    lines.append(f"  {seller_msg['content']}")

        lines.append(f"\n{'-'*60}")
        lines.append(f"Round {self.current_round} Summary:")
        lines.append(f"{'-'*60}")

        # Side-by-side offer comparison for active params
        param_labels = {
            "price":            "Price",
            "quality":          "Quality",
            "delivery_days":    "Delivery",
            "warranty_months":  "Warranty",
            "payment":          "Payment",
        }
        for p in self.ACTIVE_PARAMS:
            label = param_labels.get(p, p)
            b_val = getattr(self.buyer_offer, p)
            s_val = getattr(self.seller_offer, p)
            b_str = self._format_param(p, b_val)
            s_str = self._format_param(p, s_val)
            agreed = self._param_agreed(p)
            flag = "✓" if agreed else "✗"
            lines.append(f"  {flag} {label:12s}  Buyer: {b_str:20s}  Seller: {s_str}")

        is_agreed = self._check_agreement()
        lines.append(f"\n  Agreement Status: {'✓ AGREED' if is_agreed else '✗ NOT AGREED'}")
        lines.append(f"  Reason: {self._get_agreement_reason()}")

        if self.agreed_offer:
            lines.append(f"\n  Agreed Deal:")
            for p in self.ACTIVE_PARAMS:
                label = param_labels.get(p, p)
                val = getattr(self.agreed_offer, p)
                lines.append(f"    {label}: {self._format_param(p, val)}")

        status_display = {
            NegotiationStatus.ONGOING: "Ongoing",
            NegotiationStatus.AGREED:  "Agreed",
            NegotiationStatus.FAILED:  "Failed",
            NegotiationStatus.TIMEOUT: "Timeout",
        }
        lines.append(f"  Negotiation Status: {status_display.get(self.negotiation_info.status, 'Unknown')}")
        lines.append(f"{'='*60}\n")

        output = "\n".join(lines)
        if mode == "human":
            print(output)
            return None
        return output

    def close(self):
        self.memory.clear()
        self.buyer_offer  = MultiParamOffer()
        self.seller_offer = MultiParamOffer()
        self.agreed_offer = None

    # ------------------------------------------------------------------
    # Score printing helpers (called from example scripts)
    # ------------------------------------------------------------------

    def _print_global_score_details(self):
        self._calculate_global_score(print_details=True)

    def _print_buyer_score_details(self):
        self._calculate_buyer_score(print_details=True)

    def _print_seller_score_details(self):
        self._calculate_seller_score(print_details=True)

    # ------------------------------------------------------------------
    # Offer extraction
    # ------------------------------------------------------------------

    def _extract_offer(self, text: str, role: str) -> MultiParamOffer:
        """Extract offer fields from agent output text."""
        offer = MultiParamOffer()
        pattern = BUYER_OFFER_PATTERN if role == "buyer" else SELLER_OFFER_PATTERN
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not m:
            return offer
        body = m.group(1)
        for param in self.ACTIVE_PARAMS:
            field_pattern = FIELD_PATTERNS[param]
            fm = re.search(field_pattern, body, re.IGNORECASE)
            if fm:
                raw = fm.group(1)
                if param == "price":
                    try:
                        val = float(raw.replace(",", ""))
                        if val > 0:
                            offer.price = val
                    except ValueError:
                        pass
                elif param == "delivery_days":
                    try:
                        offer.delivery_days = int(raw)
                    except ValueError:
                        pass
                elif param == "warranty_months":
                    try:
                        offer.warranty_months = int(raw)
                    except ValueError:
                        pass
                else:
                    setattr(offer, param, raw)
        return offer

    def _merge_offer(self, current: MultiParamOffer, new: MultiParamOffer):
        """Update only the fields that were newly specified."""
        for p in self.ACTIVE_PARAMS:
            new_val = getattr(new, p)
            if new_val is not None:
                setattr(current, p, new_val)

    # ------------------------------------------------------------------
    # Agreement checking
    # ------------------------------------------------------------------

    def _check_agreement(self) -> bool:
        """Return True when all active parameters have converged."""
        bo = self.buyer_offer
        so = self.seller_offer
        if not bo.is_complete(self.ACTIVE_PARAMS) or not so.is_complete(self.ACTIVE_PARAMS):
            return False
        return all(self._param_agreed(p) for p in self.ACTIVE_PARAMS)

    def _param_agreed(self, param: str) -> bool:
        """Check convergence for a single parameter."""
        b_val = getattr(self.buyer_offer, param)
        s_val = getattr(self.seller_offer, param)
        if b_val is None or s_val is None:
            return False
        if param == "price":
            return s_val <= b_val
        elif param == "quality":
            return QUALITY_ORDER.get(s_val, -1) >= QUALITY_ORDER.get(b_val, 999)
        elif param == "delivery_days":
            return s_val <= b_val
        elif param == "warranty_months":
            return s_val >= b_val
        elif param == "payment":
            return s_val == b_val
        return False

    def _get_agreement_reason(self) -> str:
        """Human-readable explanation of which params are still unresolved."""
        if not self.buyer_offer.is_complete(self.ACTIVE_PARAMS):
            missing = [p for p in self.ACTIVE_PARAMS if getattr(self.buyer_offer, p) is None]
            return f"Buyer has not specified: {', '.join(missing)}"
        if not self.seller_offer.is_complete(self.ACTIVE_PARAMS):
            missing = [p for p in self.ACTIVE_PARAMS if getattr(self.seller_offer, p) is None]
            return f"Seller has not specified: {', '.join(missing)}"
        unresolved = [p for p in self.ACTIVE_PARAMS if not self._param_agreed(p)]
        if not unresolved:
            return "All parameters agreed"
        reasons = []
        for p in unresolved:
            b_val = getattr(self.buyer_offer, p)
            s_val = getattr(self.seller_offer, p)
            reasons.append(
                f"{p}: buyer={self._format_param(p, b_val)}, seller={self._format_param(p, s_val)}"
            )
        return "Gap on: " + "; ".join(reasons)

    # ------------------------------------------------------------------
    # Utility computation
    # ------------------------------------------------------------------

    def _buyer_param_utility(self, param: str, offer: MultiParamOffer) -> float:
        """Buyer's utility for a single parameter, normalized to [0, 1]."""
        val = getattr(offer, param)
        if val is None:
            return 0.0
        if param == "price":
            p_min, p_max = self.scenario_params["price_range"]
            span = p_max - p_min
            if span == 0:
                return 0.5
            return max(0.0, min(1.0, (p_max - val) / span))
        elif param == "quality":
            return QUALITY_ORDER.get(val, 0) / 2.0
        elif param == "delivery_days":
            d_min, d_max = self.scenario_params["delivery_range"]
            span = d_max - d_min
            if span == 0:
                return 0.5
            return max(0.0, min(1.0, (d_max - val) / span))
        elif param == "warranty_months":
            w_min, w_max = self.scenario_params["warranty_range"]
            span = w_max - w_min
            if span == 0:
                return 0.5
            return max(0.0, min(1.0, (val - w_min) / span))
        elif param == "payment":
            return PAYMENT_ORDER.get(val, 0) / 2.0
        return 0.0

    def _seller_param_utility(self, param: str, offer: MultiParamOffer) -> float:
        """Seller's utility for a single parameter, normalized to [0, 1]."""
        val = getattr(offer, param)
        if val is None:
            return 0.0
        if param == "price":
            p_min, p_max = self.scenario_params["price_range"]
            span = p_max - p_min
            if span == 0:
                return 0.5
            return max(0.0, min(1.0, (val - p_min) / span))
        elif param == "quality":
            return 1.0 - QUALITY_ORDER.get(val, 0) / 2.0
        elif param == "delivery_days":
            d_min, d_max = self.scenario_params["delivery_range"]
            span = d_max - d_min
            if span == 0:
                return 0.5
            return max(0.0, min(1.0, (val - d_min) / span))
        elif param == "warranty_months":
            w_min, w_max = self.scenario_params["warranty_range"]
            span = w_max - w_min
            if span == 0:
                return 0.5
            return max(0.0, min(1.0, (w_max - val) / span))
        elif param == "payment":
            return 1.0 - PAYMENT_ORDER.get(val, 0) / 2.0
        return 0.0

    def _weighted_utility(
        self,
        offer: MultiParamOffer,
        prefs: ParamPreferences,
        perspective: str,  # "buyer" or "seller"
    ) -> float:
        """Weighted utility for given offer and preferences, normalized to [0, 1].

        Weights are renormalized across active params so the result is always in [0, 1].
        """
        raw_weights = {p: prefs.weight_for(p) for p in self.ACTIVE_PARAMS}
        total = sum(raw_weights.values())
        if total == 0:
            return 0.0
        utility = 0.0
        for p in self.ACTIVE_PARAMS:
            w = raw_weights[p] / total
            if perspective == "buyer":
                u = self._buyer_param_utility(p, offer)
            else:
                u = self._seller_param_utility(p, offer)
            utility += w * u
        return utility

    def _unweighted_utility(self, offer: MultiParamOffer, perspective: str) -> float:
        """Equal-weighted utility across active params (used for GlobalScore Q)."""
        n = len(self.ACTIVE_PARAMS)
        if n == 0:
            return 0.0
        utils = []
        for p in self.ACTIVE_PARAMS:
            if perspective == "buyer":
                utils.append(self._buyer_param_utility(p, offer))
            else:
                utils.append(self._seller_param_utility(p, offer))
        return sum(utils) / n

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------

    def _calculate_reward(self) -> float:
        """Combined reward (buyer utility + seller utility − time cost)."""
        time_cost = -self.current_round * self.reward_weights["time_cost"]
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.agreed_offer:
            b_util = self._weighted_utility(self.agreed_offer, self.buyer_preferences,  "buyer")
            s_util = self._weighted_utility(self.agreed_offer, self.seller_preferences, "seller")
            reward = (
                b_util * self.reward_weights["buyer_utility"] +
                s_util * self.reward_weights["seller_utility"] +
                time_cost
            )
            print(
                f"Reward = buyer_util({b_util:.4f}*{self.reward_weights['buyer_utility']:.2f}) "
                f"+ seller_util({s_util:.4f}*{self.reward_weights['seller_utility']:.2f}) "
                f"+ time_cost({time_cost:.3f}) = {reward:.3f}"
            )
            return reward
        print(f"Reward = time_cost({time_cost:.3f}) (no deal, round={self.current_round})")
        return time_cost

    def _calculate_buyer_reward(self) -> float:
        """Buyer's individual reward."""
        time_cost = -self.current_round * self.reward_weights["time_cost"]
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.agreed_offer:
            b_util = self._weighted_utility(self.agreed_offer, self.buyer_preferences, "buyer")
            reward = b_util * self.reward_weights["buyer_utility"] + time_cost
            print(
                f"Buyer Reward = buyer_util({b_util:.4f}*{self.reward_weights['buyer_utility']:.2f}) "
                f"+ time_cost({time_cost:.3f}) = {reward:.3f}"
            )
            return reward
        print(f"Buyer Reward = time_cost({time_cost:.3f}) (no deal)")
        return time_cost

    def _calculate_seller_reward(self) -> float:
        """Seller's individual reward."""
        time_cost = -self.current_round * self.reward_weights["time_cost"]
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.agreed_offer:
            s_util = self._weighted_utility(self.agreed_offer, self.seller_preferences, "seller")
            reward = s_util * self.reward_weights["seller_utility"] + time_cost
            print(
                f"Seller Reward = seller_util({s_util:.4f}*{self.reward_weights['seller_utility']:.2f}) "
                f"+ time_cost({time_cost:.3f}) = {reward:.3f}"
            )
            return reward
        print(f"Seller Reward = time_cost({time_cost:.3f}) (no deal)")
        return time_cost

    def _calculate_step_buyer_reward(self) -> float:
        """Per-round buyer reward based on their own current offer."""
        round_cost = -self.current_round * self.reward_weights["time_cost"]
        if self.buyer_offer.is_complete(self.ACTIVE_PARAMS):
            b_util = self._weighted_utility(self.buyer_offer, self.buyer_preferences, "buyer")
            return b_util * self.reward_weights["buyer_utility"] + round_cost
        return round_cost

    def _calculate_step_seller_reward(self) -> float:
        """Per-round seller reward based on their own current offer."""
        round_cost = -self.current_round * self.reward_weights["time_cost"]
        if self.seller_offer.is_complete(self.ACTIVE_PARAMS):
            s_util = self._weighted_utility(self.seller_offer, self.seller_preferences, "seller")
            return s_util * self.reward_weights["seller_utility"] + round_cost
        return round_cost

    # ------------------------------------------------------------------
    # Score calculation (GlobalScore, BuyerScore, SellerScore)
    # ------------------------------------------------------------------

    def _calculate_global_score(self, print_details: bool = True) -> float:
        """Calculate GlobalScore.

        GlobalScore = discount * (D + W * Q + E)   if deal agreed in valid range
                    = -F * (1 - discount)            otherwise

        where:
          Q = 4 * avg_buyer_util * avg_seller_util  (unweighted, in [0,1])
          discount = gamma^round_index

        Rationale:
          - D (deal bonus): rewards any successful agreement
          - W * Q (quality bonus): rewards Pareto-efficient deals where both parties
            do well; Q is maximized when utilities are balanced (u_b = u_s = 0.5)
          - E (efficiency bonus): rewards reaching agreement quickly
          - Failure penalty grows as rounds increase without a deal
        """
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        feasible_deal = self.negotiation_info.status == NegotiationStatus.AGREED

        if print_details:
            print(f"\n[GlobalScore Calculation]")
            print(f"  Active params: {self.ACTIVE_PARAMS}")
            print(f"  round_index={round_index}, gamma={self.gamma}, discount=γ^{round_index}={discount:.6f}")
            print(f"  feasible_deal={feasible_deal}")

        if feasible_deal and self.agreed_offer:
            u_b = self._unweighted_utility(self.agreed_offer, "buyer")
            u_s = self._unweighted_utility(self.agreed_offer, "seller")
            Q = 4.0 * u_b * u_s  # in [0, 1], maximized when u_b = u_s = 0.5
            deal_score      = self.deal_score_weight      * discount
            quality_score   = self.quality_score_weight   * Q * discount
            efficiency_score= self.efficiency_score_weight* discount
            global_score    = deal_score + quality_score + efficiency_score

            if print_details:
                print(f"  u_b (unweighted avg buyer util)  = {u_b:.4f}")
                print(f"  u_s (unweighted avg seller util) = {u_s:.4f}")
                print(f"  Q = 4 * u_b * u_s = {Q:.4f}  (1.0 = perfectly balanced deal)")
                print(f"  DealScore      = D({self.deal_score_weight:.1f}) * discount({discount:.6f}) = {deal_score:.3f}")
                print(f"  QualityScore   = W({self.quality_score_weight:.1f}) * Q({Q:.4f}) * discount = {quality_score:.3f}")
                print(f"  EfficiencyScore= E({self.efficiency_score_weight:.1f}) * discount = {efficiency_score:.3f}")
                print(f"  GlobalScore = {deal_score:.3f} + {quality_score:.3f} + {efficiency_score:.3f} = {global_score:.3f}")
            return global_score

        failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
        if print_details:
            print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount) = {failure_penalty:.3f}")
            print(f"  GlobalScore = {failure_penalty:.3f}")
        return failure_penalty

    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """Calculate BuyerScore.

        BuyerScore = discount * (Db + Wb * buyer_utility + Eb)   if deal agreed
                   = -Fb * (1 - discount)                         otherwise

        buyer_utility = weighted average of per-param utilities using buyer's
                        own preference weights (preference-weighted, not equal-weighted).

        Rationale: reflects how well the buyer did *given their own priorities*.
                   Buyer who cares most about price and gets a great price scores
                   higher than one who got a mediocre price on a less-important param.
        """
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        feasible_deal = self.negotiation_info.status == NegotiationStatus.AGREED

        if print_details:
            print(f"\n[BuyerScore Calculation]")
            print(f"  round_index={round_index}, discount={discount:.6f}, feasible_deal={feasible_deal}")

        if feasible_deal and self.agreed_offer:
            buyer_util = self._weighted_utility(self.agreed_offer, self.buyer_preferences, "buyer")
            buyer_score = discount * (
                self.buyer_deal_weight +
                self.buyer_utility_weight * buyer_util +
                self.buyer_efficiency_weight
            )
            if print_details:
                print(f"  buyer_utility (preference-weighted) = {buyer_util:.4f}")
                print(f"  BuyerScore = discount({discount:.6f}) * (Db({self.buyer_deal_weight:.1f}) + Wb({self.buyer_utility_weight:.1f})*{buyer_util:.4f} + Eb({self.buyer_efficiency_weight:.1f})) = {buyer_score:.3f}")
            return buyer_score

        buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
        if print_details:
            print(f"  BuyerScore (failure) = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount) = {buyer_score:.3f}")
        return buyer_score

    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """Calculate SellerScore.

        SellerScore = discount * (Ds + Ws * seller_utility + Es)   if deal agreed
                    = -Fs * (1 - discount)                          otherwise

        seller_utility = weighted average of per-param utilities using seller's
                         own preference weights.
        """
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        feasible_deal = self.negotiation_info.status == NegotiationStatus.AGREED

        if print_details:
            print(f"\n[SellerScore Calculation]")
            print(f"  round_index={round_index}, discount={discount:.6f}, feasible_deal={feasible_deal}")

        if feasible_deal and self.agreed_offer:
            seller_util = self._weighted_utility(self.agreed_offer, self.seller_preferences, "seller")
            seller_score = discount * (
                self.seller_deal_weight +
                self.seller_utility_weight * seller_util +
                self.seller_efficiency_weight
            )
            if print_details:
                print(f"  seller_utility (preference-weighted) = {seller_util:.4f}")
                print(f"  SellerScore = discount({discount:.6f}) * (Ds({self.seller_deal_weight:.1f}) + Ws({self.seller_utility_weight:.1f})*{seller_util:.4f} + Es({self.seller_efficiency_weight:.1f})) = {seller_score:.3f}")
            return seller_score

        seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
        if print_details:
            print(f"  SellerScore (failure) = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount) = {seller_score:.3f}")
        return seller_score

    # ------------------------------------------------------------------
    # Observation / info helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "conversation_history": self.memory.get_history(),
            "current_round":        self.current_round,
            "buyer_offer":          self.buyer_offer.to_dict(self.ACTIVE_PARAMS),
            "seller_offer":         self.seller_offer.to_dict(self.ACTIVE_PARAMS),
            "agreed_offer":         self.agreed_offer.to_dict(self.ACTIVE_PARAMS) if self.agreed_offer else None,
            "status":               self.negotiation_info.status.value,
            "active_params":        self.ACTIVE_PARAMS,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "round":          self.current_round,
            "status":         self.negotiation_info.status.value,
            "buyer_offer":    self.buyer_offer.to_dict(self.ACTIVE_PARAMS),
            "seller_offer":   self.seller_offer.to_dict(self.ACTIVE_PARAMS),
            "agreed_offer":   self.agreed_offer.to_dict(self.ACTIVE_PARAMS) if self.agreed_offer else None,
            "buyer_price":    self.buyer_offer.price,
            "seller_price":   self.seller_offer.price,
            "agreed_price":   self.agreed_offer.price if self.agreed_offer else None,
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_param(param: str, val: Any) -> str:
        if val is None:
            return "—"
        if param == "price":
            return f"${val:,.2f}"
        elif param == "delivery_days":
            return f"{val} days"
        elif param == "warranty_months":
            return f"{val} months"
        return str(val)
