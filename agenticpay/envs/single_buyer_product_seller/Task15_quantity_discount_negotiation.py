"""Quantity / Bulk-Discount Negotiation Environment

Extends the single-buyer × single-seller setting with a second negotiable
dimension: **quantity**.  The seller holds a private tiered pricing table
(unit price drops at volume thresholds), while the buyer has a private
per-unit budget cap and a target order size.

Price tags follow the same convention as all other tasks in the benchmark:
``### BUYER_PRICE($X) ###`` and ``### SELLER_PRICE($X) ###`` contain the
**TOTAL price** for the order (matching the hardcoded guidance in
BuyerAgent / SellerAgent).  Quantity tags are new:

    Buyer:  ### BUYER_QUANTITY(50) ###   ### BUYER_PRICE($380) ###
    Seller: ### SELLER_QUANTITY(50) ###  ### SELLER_PRICE($390) ###

The env internally derives per-unit prices:
    buyer_unit  = buyer_total  / buyer_quantity
    seller_unit = seller_total / seller_quantity

Agreement is reached when the quantity tags converge (within
``quantity_tolerance``) AND the per-unit prices converge (within
``price_tolerance``).

Scores (GlobalScore / BuyerScore / SellerScore) are calculated on the
per-unit price ZOPA so that results are comparable with all other tasks.
The agreed total deal value (agreed_quantity × agreed_unit_price) is
recorded separately in the result JSON.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task15QuantityDiscountNegotiation(BaseEnv):
    """Quantity / bulk-discount negotiation environment.

    Agents negotiate both the order quantity and the total order price.
    The env derives per-unit prices internally for ZOPA scoring.
    The seller's tiered pricing is private information that they may
    choose to reveal strategically during the negotiation.
    """

    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_unit_price: float = 20.0,
        buyer_max_unit_price: Optional[float] = None,
        buyer_target_quantity: int = 50,
        seller_min_unit_price: Optional[float] = None,
        seller_tiers: Optional[List[Tuple[int, float]]] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 0.5,
        quantity_tolerance: int = 0,
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
        """Initialise the environment.

        Args:
            buyer_agent: Buyer agent instance.
            seller_agent: Seller agent instance.
            max_rounds: Maximum negotiation rounds.
            initial_seller_unit_price: Seller's publicly listed price per unit.
            buyer_max_unit_price: Private ceiling price *per unit* for the buyer.
            buyer_target_quantity: Number of units the buyer needs (public information).
            seller_min_unit_price: Private floor price *per unit* for the seller.
            seller_tiers: Private tiered pricing list of (min_qty, unit_price) tuples,
                sorted ascending by min_qty, e.g.
                [(1, 10.0), (10, 9.0), (50, 8.0), (100, 7.0)]
            environment_info: Optional environment context dict.
            price_tolerance: Absolute difference in $/unit to trigger agreement.
                Applied to *derived* per-unit prices (total / qty).
            quantity_tolerance: Max unit difference in quantity to trigger agreement
                (0 = exact match required).
            reward_weights: Dict with keys buyer_savings, seller_profit, time_cost.
            gamma: Discount factor for score calculation (default 0.99).
            deal_score_weight: D weight (default 30).
            quality_score_weight: W weight (default 55).
            efficiency_score_weight: E weight (default 15).
            failure_penalty_weight: F weight (default 15).
            buyer_deal_weight: Db weight (default 30).
            buyer_utility_weight: Wb weight (default 55).
            buyer_efficiency_weight: Eb weight (default 15).
            buyer_failure_penalty_weight: Fb weight (default 15).
            seller_deal_weight: Ds weight (default 30).
            seller_utility_weight: Ws weight (default 55).
            seller_efficiency_weight: Es weight (default 15).
            seller_failure_penalty_weight: Fs weight (default 15).
        """
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_unit_price = initial_seller_unit_price

        # Internal aliases used throughout scoring methods (per-unit basis)
        self.buyer_max_price = buyer_max_unit_price      # $/unit
        self.seller_min_price = seller_min_unit_price    # $/unit

        self.buyer_target_quantity = buyer_target_quantity
        self.seller_tiers = seller_tiers or [(1, initial_seller_unit_price)]
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance       # $/unit tolerance
        self.quantity_tolerance = quantity_tolerance

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

        default_weights = {
            "buyer_savings": 1.0,
            "seller_profit": 1.0,
            "time_cost": 0.1,
        }
        if reward_weights is not None:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights

        super().__init__()

        # State management — buyer_price / seller_price store TOTAL order prices
        self.memory = ConversationMemory()
        self.state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()

        # Quantity state (tracked separately from NegotiationState)
        self.buyer_quantity: Optional[int] = None
        self.seller_quantity: Optional[int] = None
        self.agreed_quantity: Optional[int] = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and start a new negotiation.

        Injects quantity-specific instructions into the agents via
        system_prompt_suffix so they know to include quantity tags.

        Args:
            user_requirement: Buyer's stated requirement.
            product_info: Product information dict.
            user_profile: Optional user profile (string or UserProfile).

        Returns:
            (observation, info) initial state tuple.
        """
        # Reset price / round state
        self.memory.clear()
        self.state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()

        # Reset quantity state
        self.buyer_quantity = None
        self.seller_quantity = None
        self.agreed_quantity = None

        product_info = product_info or {}
        product_images = kwargs.get("product_images")
        if product_images is None:
            product_images = product_info.get("product_images") or product_info.get("images")
        if product_images is None:
            img_path = product_info.get("image_path") or product_info.get("image_url")
            if img_path is not None:
                product_images = [img_path]
        if product_images is not None and not isinstance(product_images, list):
            product_images = [product_images]
        self.product_images = product_images

        qty = self.buyer_target_quantity
        listed_total = qty * self.initial_seller_unit_price

        # Prompt suffixes: BUYER_PRICE / SELLER_PRICE remain TOTAL prices (matching
        # the hardcoded guidance in BuyerAgent / SellerAgent).  Only the quantity
        # tag is new.
        buyer_qty_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- You need to purchase approximately {qty} units of this product.\n"
            f"- In EVERY turn you MUST include your desired quantity using this tag:\n"
            f"  ### BUYER_QUANTITY({qty}) ###\n"
            f"- BUYER_PRICE($X) is the TOTAL PRICE for the entire order (as always).\n"
            f"  Example: for {qty} units at $7.60 each, use ### BUYER_PRICE(${qty * 7.60:.0f}) ###\n"
            f"- You can negotiate both quantity and per-unit rate to get a better deal.\n"
            f"- The seller may offer lower per-unit prices for larger orders — use this strategically.\n"
        )

        tier_hint = "Volume discounts are available — larger orders unlock lower per-unit prices."
        seller_qty_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- You sell this product with volume-based (tiered) pricing.\n"
            f"- {tier_hint}\n"
            f"- In EVERY turn you MUST include the quantity you are prepared to fulfil:\n"
            f"  ### SELLER_QUANTITY({qty}) ###\n"
            f"- SELLER_PRICE($X) is the TOTAL PRICE for the entire order (as always).\n"
            f"  Example: for {qty} units at $8.00 each, use ### SELLER_PRICE(${qty * 8.0:.0f}) ###\n"
            f"- You may strategically reveal discount tiers to encourage larger purchases.\n"
        )

        # Inject into agents (reset() always re-initialises)
        self.buyer_agent.system_prompt_suffix = buyer_qty_suffix
        self.seller_agent.system_prompt_suffix = seller_qty_suffix

        buyer_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer_max_price * qty if self.buyer_max_price else None,
            "target_quantity": self.buyer_target_quantity,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info,
            "product_images": product_images,
        }
        self.buyer_agent.initialize(buyer_context)

        tier_lines = []
        for min_qty, unit_price in sorted(self.seller_tiers, key=lambda t: t[0]):
            tier_lines.append(f"  {min_qty}+ units: ${unit_price:.2f}/unit")
        tier_description = "\n".join(tier_lines) if tier_lines else "No tiers defined."

        seller_context = {
            "product_info": product_info,
            "initial_price": listed_total,     # total for target qty at listed price
            "min_price": self.seller_min_price * qty if self.seller_min_price else None,
            "pricing_tiers": tier_description,
            "environment_info": self.environment_info,
            "product_images": product_images,
        }
        self.seller_agent.initialize(seller_context)

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        buyer_action: Optional[str] = None,
        seller_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation round.

        Extracted BUYER_PRICE / SELLER_PRICE values are total order prices.
        Per-unit prices are derived internally via _derive_unit_price().

        Args:
            buyer_action: Buyer's response text.
            seller_action: Seller's response text.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if buyer_action is not None:
            self.memory.add_message("buyer", buyer_action, self.current_round)
            buyer_total = self._extract_price(buyer_action)
            if buyer_total is not None:
                self.state.update(buyer_price=buyer_total)
                self.negotiation_info.buyer_price = buyer_total
                self.negotiation_info.current_price = buyer_total
            buyer_qty = self._extract_quantity(buyer_action, role="buyer")
            if buyer_qty is not None:
                self.buyer_quantity = buyer_qty
            elif self.buyer_quantity is None:
                self.buyer_quantity = self.buyer_target_quantity

        if seller_action is not None:
            self.memory.add_message("seller", seller_action, self.current_round)
            seller_total = self._extract_price(seller_action)
            if seller_total is not None:
                self.state.update(seller_price=seller_total)
                self.negotiation_info.seller_price = seller_total
                self.negotiation_info.current_price = seller_total
            seller_qty = self._extract_quantity(seller_action, role="seller")
            if seller_qty is not None:
                self.seller_quantity = seller_qty
            elif self.seller_quantity is None:
                self.seller_quantity = self.buyer_target_quantity

        terminated = False
        truncated = False
        reward = 0.0
        seller_reward = 0.0
        buyer_reward = 0.0

        if self._check_agreement():
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            # Determine agreed total: seller total ≤ buyer total → deal at seller's; else midpoint
            bq = self.buyer_quantity or self.buyer_target_quantity
            sq = self.seller_quantity or self.buyer_target_quantity
            buyer_unit = self._derive_unit_price(self.state.buyer_price, bq)
            seller_unit = self._derive_unit_price(self.state.seller_price, sq)
            if seller_unit <= buyer_unit:
                agreed_unit = seller_unit
            else:
                agreed_unit = (buyer_unit + seller_unit) / 2
            # agreed_quantity: average of the two converged quantities
            self.agreed_quantity = round((bq + sq) / 2)
            agreed_total = agreed_unit * self.agreed_quantity
            self.state.update(agreed_price=agreed_total)   # total stored in state
            self.negotiation_info.current_price = agreed_total
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
        step_buyer_reward = self._calculate_step_buyer_reward()

        observation = self._get_observation()
        info = self._get_info()
        info["step_seller_reward"] = step_seller_reward
        info["step_buyer_reward"] = step_buyer_reward

        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["seller_reward"] = seller_reward
            info["buyer_reward"] = buyer_reward
            global_score = self._calculate_global_score(print_details=False)
            info["global_score"] = global_score
            buyer_score = self._calculate_buyer_score(print_details=False)
            info["buyer_score"] = buyer_score
            seller_score = self._calculate_seller_score(print_details=False)
            info["seller_score"] = seller_score
            info["agreed_quantity"] = self.agreed_quantity
            info["buyer_quantity"] = self.buyer_quantity
            info["seller_quantity"] = self.seller_quantity
            # Expose per-unit agreed price (derived from total / quantity)
            agreed_unit = self._get_final_unit_price()
            info["agreed_unit_price"] = round(agreed_unit, 4) if agreed_unit is not None else None
            agreed_total = self.state.agreed_price
            info["total_deal_value"] = round(agreed_total, 4) if agreed_total is not None else None

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        """Render negotiation state to console or return as string."""
        output_lines = []

        history = self.memory.get_history()
        if history:
            round_to_display = self.current_round - 1 if self.current_round > 0 else 0
            round_messages = [msg for msg in history if msg["round"] == round_to_display]

            if round_messages:
                display_round = self.current_round
                output_lines.append(f"\n{'='*60}")
                output_lines.append(f"Round {display_round} - Negotiation Output")
                output_lines.append(f"{'='*60}")

                buyer_msg = next((m for m in round_messages if m["role"] == "buyer"), None)
                if buyer_msg:
                    output_lines.append(f"\n[BUYER Output]:")
                    output_lines.append(f"  {buyer_msg['content']}")

                seller_msg = next((m for m in round_messages if m["role"] == "seller"), None)
                if seller_msg:
                    output_lines.append(f"\n[SELLER Output]:")
                    output_lines.append(f"  {seller_msg['content']}")

        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")

        # Show quantities
        bq = self.buyer_quantity if self.buyer_quantity is not None else "N/A"
        sq = self.seller_quantity if self.seller_quantity is not None else "N/A"
        output_lines.append(f"  Buyer Quantity: {bq}  |  Seller Quantity: {sq}")

        if self.state.agreed_price is not None:
            aq = self.agreed_quantity if self.agreed_quantity is not None else "N/A"
            agreed_unit = self._get_final_unit_price()
            if agreed_unit is not None:
                output_lines.append(f"  Buyer Unit Price: ${agreed_unit:.2f}")
                output_lines.append(f"  Seller Unit Price: ${agreed_unit:.2f}")
            output_lines.append(f"  Agreed Quantity: {aq}")
            if self.agreed_quantity is not None and agreed_unit is not None:
                total = self.agreed_quantity * agreed_unit
                output_lines.append(f"  Total Deal Value: ${total:.2f}")
        else:
            # Show derived unit prices when available
            b_unit = self._derive_unit_price(self.state.buyer_price, self.buyer_quantity)
            s_unit = self._derive_unit_price(self.state.seller_price, self.seller_quantity)
            if b_unit is not None:
                bt = self.state.buyer_price
                output_lines.append(f"  Buyer Unit Price: ${b_unit:.2f}  (total: ${bt:.2f})")
            else:
                output_lines.append(f"  Buyer Unit Price: Not specified")
            if s_unit is not None:
                st = self.state.seller_price
                output_lines.append(f"  Seller Unit Price: ${s_unit:.2f}  (total: ${st:.2f})")
            else:
                output_lines.append(f"  Seller Unit Price: Not specified")

        is_agreed = self._check_agreement()
        agreement_reason = self._get_agreement_reason()
        output_lines.append(f"  Agreement Status: {'✓ AGREED' if is_agreed else '✗ NOT AGREED'}")
        output_lines.append(f"  Reason: {agreement_reason}")

        status_display = {
            NegotiationStatus.ONGOING: "Ongoing",
            NegotiationStatus.AGREED: "Agreed",
            NegotiationStatus.FAILED: "Failed",
            NegotiationStatus.TIMEOUT: "Timeout",
        }
        output_lines.append(
            f"  Negotiation Status: {status_display.get(self.negotiation_info.status, 'Unknown')}"
        )
        output_lines.append(f"{'='*60}\n")

        output = "\n".join(output_lines)
        if mode == "human":
            print(output)
            return None
        return output

    def close(self):
        """Close environment and clear state."""
        self.memory.clear()
        self.state = NegotiationState()

    # ------------------------------------------------------------------
    # Observation / info helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, Any]:
        obs = {
            "conversation_history": self.memory.get_history(),
            "current_round": self.current_round,
            "seller_price": self.state.seller_price,
            "buyer_price": self.state.buyer_price,
            "buyer_quantity": self.buyer_quantity,
            "seller_quantity": self.seller_quantity,
            "status": self.negotiation_info.status.value,
        }
        if getattr(self, "product_images", None):
            obs["product_images"] = self.product_images
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "seller_price": self.state.seller_price,
            "buyer_price": self.state.buyer_price,
            "agreed_price": self.state.agreed_price,
            "buyer_quantity": self.buyer_quantity,
            "seller_quantity": self.seller_quantity,
            "negotiation_info": self.negotiation_info,
        }

    # ------------------------------------------------------------------
    # Price / quantity extraction
    # ------------------------------------------------------------------

    def _extract_price(self, text: str) -> Optional[float]:
        """Extract total order price from agent text.

        Priority:
        1. ### BUYER_PRICE($X) ### or ### SELLER_PRICE($X) ###
        2. ### $X ###
        3. Fallback patterns
        """
        def parse_price(s: str) -> Optional[float]:
            try:
                v = float(s.replace(",", ""))
                return v if v > 0 else None
            except ValueError:
                return None

        labeled = r'###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###'
        matches = re.findall(labeled, text, re.IGNORECASE)
        if matches:
            p = parse_price(matches[-1])
            if p is not None:
                return p

        triple = r'###\s*\$([\d,]+\.?\d*)\s*###'
        matches = re.findall(triple, text, re.IGNORECASE)
        if matches:
            p = parse_price(matches[-1])
            if p is not None:
                return p

        fallbacks = [
            r'\$([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*dollars?',
            r'([\d,]+\.?\d*)\s*USD',
            r'price.*?([\d,]+\.?\d*)',
            r'offer.*?([\d,]+\.?\d*)',
        ]
        for pattern in fallbacks:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                p = parse_price(matches[-1])
                if p is not None:
                    return p
        return None

    def _extract_quantity(self, text: str, role: str = "buyer") -> Optional[int]:
        """Extract quantity from agent text.

        Looks for:
          ### BUYER_QUANTITY(X) ###   (when role='buyer')
          ### SELLER_QUANTITY(X) ###  (when role='seller')

        Falls back to any ### *_QUANTITY(X) ### tag.

        Args:
            text: Agent response text.
            role: 'buyer' or 'seller' — selects which labeled tag to prefer.

        Returns:
            Integer quantity, or None if not found.
        """
        role_upper = role.upper()
        labeled = rf'###\s*{role_upper}_QUANTITY\s*\(\s*(\d+)\s*\)\s*###'
        matches = re.findall(labeled, text, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                pass

        generic = r'###\s*(?:BUYER_QUANTITY|SELLER_QUANTITY|QUANTITY)\s*\(\s*(\d+)\s*\)\s*###'
        matches = re.findall(generic, text, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                pass

        return None

    # ------------------------------------------------------------------
    # Unit-price derivation helper
    # ------------------------------------------------------------------

    def _derive_unit_price(
        self, total_price: Optional[float], quantity: Optional[int]
    ) -> Optional[float]:
        """Return total_price / quantity, falling back to total_price if no valid quantity.

        Args:
            total_price: Total order price as output by the agent.
            quantity: Number of units for this offer.

        Returns:
            Per-unit price, or None if total_price is None.
        """
        if total_price is None:
            return None
        q = quantity if (quantity is not None and quantity > 0) else self.buyer_target_quantity
        if q and q > 0:
            return total_price / q
        return total_price

    # ------------------------------------------------------------------
    # Agreement check
    # ------------------------------------------------------------------

    def _check_agreement(self) -> bool:
        """Return True when both unit-price and quantity have converged."""
        if self.state.buyer_price is None or self.state.seller_price is None:
            return False

        bq = self.buyer_quantity if self.buyer_quantity is not None else self.buyer_target_quantity
        sq = self.seller_quantity if self.seller_quantity is not None else self.buyer_target_quantity

        buyer_unit = self._derive_unit_price(self.state.buyer_price, bq)
        seller_unit = self._derive_unit_price(self.state.seller_price, sq)

        qty_diff = abs(bq - sq)
        price_ok = (abs(buyer_unit - seller_unit) <= self.price_tolerance) or (seller_unit <= buyer_unit)
        qty_ok = qty_diff <= self.quantity_tolerance

        return price_ok and qty_ok

    def _get_agreement_reason(self) -> str:
        if self.state.buyer_price is None or self.state.seller_price is None:
            return "Prices not yet specified"

        bq = self.buyer_quantity if self.buyer_quantity is not None else self.buyer_target_quantity
        sq = self.seller_quantity if self.seller_quantity is not None else self.buyer_target_quantity
        buyer_unit = self._derive_unit_price(self.state.buyer_price, bq)
        seller_unit = self._derive_unit_price(self.state.seller_price, sq)

        qty_diff = abs(bq - sq)
        unit_diff = abs(buyer_unit - seller_unit)
        price_ok = (unit_diff <= self.price_tolerance) or (seller_unit <= buyer_unit)
        qty_ok = qty_diff <= self.quantity_tolerance

        if price_ok and qty_ok:
            return (
                f"Unit price converged (${seller_unit:.2f} vs ${buyer_unit:.2f}/unit) "
                f"and quantity converged ({sq} vs {bq})"
            )
        parts = []
        if not price_ok:
            parts.append(
                f"unit price gap ${unit_diff:.2f}/unit > tolerance ${self.price_tolerance:.2f}/unit"
            )
        if not qty_ok:
            parts.append(f"quantity gap {qty_diff} > tolerance {self.quantity_tolerance}")
        return "No agreement: " + "; ".join(parts)

    # ------------------------------------------------------------------
    # Reward calculations  (unit prices derived from totals)
    # ------------------------------------------------------------------

    def _calculate_reward(self) -> float:
        time_cost = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            if self.state.agreed_price is None:
                return time_cost * self.reward_weights["time_cost"]
            qty = self.agreed_quantity or self.buyer_target_quantity
            agreed_unit = self.state.agreed_price / qty if qty > 0 else self.state.agreed_price
            reward = buyer_savings = seller_profit = 0.0
            if self.buyer_max_price is not None:
                buyer_savings = (self.buyer_max_price - agreed_unit) * qty
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            if self.seller_min_price is not None:
                seller_profit = (agreed_unit - self.seller_min_price) * qty
                reward += seller_profit * self.reward_weights["seller_profit"]
            reward += time_cost * self.reward_weights["time_cost"]
            print(
                f"Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f})"
                f" + seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f})"
                f" + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f})"
                f" = {reward:.2f}"
                f" (qty={qty}, unit=${agreed_unit:.2f}, total=${self.state.agreed_price:.2f}, round={self.current_round})"
            )
            return reward
        else:
            wc = time_cost * self.reward_weights["time_cost"]
            print(f"Reward = time_cost = {wc:.2f} (deal not reached)")
            return wc

    def _calculate_seller_reward(self) -> float:
        time_cost = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            if self.state.agreed_price is None:
                return time_cost * self.reward_weights["time_cost"]
            qty = self.agreed_quantity or self.buyer_target_quantity
            agreed_unit = self.state.agreed_price / qty if qty > 0 else self.state.agreed_price
            reward = seller_profit = 0.0
            if self.seller_min_price is not None:
                seller_profit = (agreed_unit - self.seller_min_price) * qty
                reward += seller_profit * self.reward_weights["seller_profit"]
            reward += time_cost * self.reward_weights["time_cost"]
            print(
                f"Seller Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f})"
                f" + time_cost = {reward:.2f}"
                f" (qty={qty}, unit=${agreed_unit:.2f}, seller_min=${self.seller_min_price})"
            )
            return reward
        else:
            wc = time_cost * self.reward_weights["time_cost"]
            print(f"Seller Reward = time_cost = {wc:.2f} (deal not reached)")
            return wc

    def _calculate_buyer_reward(self) -> float:
        time_cost = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            if self.state.agreed_price is None:
                return time_cost * self.reward_weights["time_cost"]
            qty = self.agreed_quantity or self.buyer_target_quantity
            agreed_unit = self.state.agreed_price / qty if qty > 0 else self.state.agreed_price
            reward = buyer_savings = 0.0
            if self.buyer_max_price is not None:
                buyer_savings = (self.buyer_max_price - agreed_unit) * qty
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            reward += time_cost * self.reward_weights["time_cost"]
            print(
                f"Buyer Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f})"
                f" + time_cost = {reward:.2f}"
                f" (qty={qty}, unit=${agreed_unit:.2f}, buyer_max=${self.buyer_max_price})"
            )
            return reward
        else:
            wc = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer Reward = time_cost = {wc:.2f} (deal not reached)")
            return wc

    def _calculate_step_seller_reward(self) -> float:
        round_cost = -self.current_round
        reward = seller_profit = 0.0
        if self.state.seller_price is not None and self.seller_min_price is not None:
            sq = self.seller_quantity or self.buyer_target_quantity
            seller_unit = self._derive_unit_price(self.state.seller_price, sq)
            seller_profit = (seller_unit - self.seller_min_price) * sq
            reward += seller_profit * self.reward_weights["seller_profit"]
        reward += round_cost * self.reward_weights["time_cost"]
        return reward

    def _calculate_step_buyer_reward(self) -> float:
        round_cost = -self.current_round
        reward = buyer_savings = 0.0
        if self.state.buyer_price is not None and self.buyer_max_price is not None:
            bq = self.buyer_quantity or self.buyer_target_quantity
            buyer_unit = self._derive_unit_price(self.state.buyer_price, bq)
            buyer_savings = (self.buyer_max_price - buyer_unit) * bq
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        reward += round_cost * self.reward_weights["time_cost"]
        return reward

    # ------------------------------------------------------------------
    # Score calculations (per-unit price ZOPA, same formula as existing tasks)
    # ------------------------------------------------------------------

    def _get_final_unit_price(self) -> Optional[float]:
        """Return the best available per-unit price.

        Derives unit price from total prices stored in state:
            unit = total / quantity
        """
        if self.state.agreed_price is not None:
            qty = self.agreed_quantity or self.buyer_target_quantity
            return self.state.agreed_price / qty if qty > 0 else self.state.agreed_price

        bq = self.buyer_quantity or self.buyer_target_quantity
        sq = self.seller_quantity or self.buyer_target_quantity

        if self.state.buyer_price is not None and self.state.seller_price is not None:
            bu = self._derive_unit_price(self.state.buyer_price, bq)
            su = self._derive_unit_price(self.state.seller_price, sq)
            return (bu + su) / 2

        if self.state.buyer_price is not None:
            return self._derive_unit_price(self.state.buyer_price, bq)
        if self.state.seller_price is not None:
            return self._derive_unit_price(self.state.seller_price, sq)
        return None

    def _calculate_global_score(self, print_details: bool = True) -> float:
        """GlobalScore on per-unit price ZOPA (same formula as Task1)."""
        if self.buyer_max_price is None or self.seller_min_price is None:
            round_index = max(0, self.current_round - 1)
            discount = self.gamma ** round_index
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  buyer_max_unit_price or seller_min_unit_price is None")
                print(f"  GlobalScore = {fp:.3f}")
            return fp

        Z = self.buyer_max_price - self.seller_min_price
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index

        feasible_deal = (
            self.negotiation_info.status == NegotiationStatus.AGREED
            or self.state.agreed_price is not None
        )

        final_unit = self._get_final_unit_price()
        if final_unit is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[GlobalScore Calculation]  No unit price available → {fp:.3f}")
            return fp

        valid_range = Z > 0 and self.seller_min_price <= final_unit <= self.buyer_max_price

        if feasible_deal and valid_range:
            u_b = (self.buyer_max_price - final_unit) / Z
            u_s = (final_unit - self.seller_min_price) / Z
            Q = 4.0 * u_b * u_s
            deal_score = self.deal_score_weight * discount
            quality_score = self.quality_score_weight * Q * discount
            efficiency_score = self.efficiency_score_weight * discount
            global_score = deal_score + quality_score + efficiency_score
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = buyer_max_unit(${self.buyer_max_price:.2f}) - seller_min_unit(${self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  agreed_unit_price = ${final_unit:.4f}")
                print(f"  feasible_deal = {feasible_deal}, valid_range = {valid_range}")
                print(f"  discount = γ^{round_index} = {discount:.6f}")
                print(f"  u_b = {u_b:.4f}, u_s = {u_s:.4f}, Q = {Q:.4f}")
                print(f"  DealScore = {deal_score:.3f}, QualityScore = {quality_score:.3f}, EfficiencyScore = {efficiency_score:.3f}")
                print(f"  GlobalScore = {global_score:.3f}")
            return global_score
        else:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = {Z:.2f}, agreed_unit_price = ${final_unit:.4f}")
                print(f"  feasible_deal = {feasible_deal}, valid_range = {valid_range}")
                print(f"  FailurePenalty = {fp:.3f}")
            return fp

    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """BuyerScore on per-unit price surplus (same formula as Task1)."""
        if self.buyer_max_price is None or self.seller_min_price is None:
            round_index = max(0, self.current_round - 1)
            discount = self.gamma ** round_index
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[BuyerScore Calculation]  buyer_max or seller_min is None → {bs:.3f}")
            return bs

        Z = self.buyer_max_price - self.seller_min_price
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index
        feasible_deal = (
            self.negotiation_info.status == NegotiationStatus.AGREED
            or self.state.agreed_price is not None
        )
        final_unit = self._get_final_unit_price()
        if final_unit is None:
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[BuyerScore Calculation]  No unit price → {bs:.3f}")
            return bs

        valid_range = Z > 0 and self.seller_min_price <= final_unit <= self.buyer_max_price
        if feasible_deal and valid_range:
            u_b = (self.buyer_max_price - final_unit) / Z
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = {Z:.2f}, agreed_unit = ${final_unit:.4f}, u_b = {u_b:.4f}")
                print(f"  BuyerScore = {buyer_score:.3f}")
            return buyer_score
        else:
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[BuyerScore Calculation]  failure (unit=${final_unit:.4f} out of range) → {bs:.3f}")
            return bs

    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """SellerScore on per-unit price surplus (same formula as Task1)."""
        if self.buyer_max_price is None or self.seller_min_price is None:
            round_index = max(0, self.current_round - 1)
            discount = self.gamma ** round_index
            ss = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[SellerScore Calculation]  buyer_max or seller_min is None → {ss:.3f}")
            return ss

        Z = self.buyer_max_price - self.seller_min_price
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index
        feasible_deal = (
            self.negotiation_info.status == NegotiationStatus.AGREED
            or self.state.agreed_price is not None
        )
        final_unit = self._get_final_unit_price()
        if final_unit is None:
            ss = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[SellerScore Calculation]  No unit price → {ss:.3f}")
            return ss

        valid_range = Z > 0 and self.seller_min_price <= final_unit <= self.buyer_max_price
        if feasible_deal and valid_range:
            u_s = (final_unit - self.seller_min_price) / Z
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = {Z:.2f}, agreed_unit = ${final_unit:.4f}, u_s = {u_s:.4f}")
                print(f"  SellerScore = {seller_score:.3f}")
            return seller_score
        else:
            ss = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[SellerScore Calculation]  failure (unit=${final_unit:.4f} out of range) → {ss:.3f}")
            return ss

    def _print_global_score_details(self):
        self._calculate_global_score(print_details=True)

    def _print_buyer_score_details(self):
        self._calculate_buyer_score(print_details=True)

    def _print_seller_score_details(self):
        self._calculate_seller_score(print_details=True)
