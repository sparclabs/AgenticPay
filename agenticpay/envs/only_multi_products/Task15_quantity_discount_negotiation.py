"""Task15 Multi-Product Quantity/Bulk-Discount Negotiation Environment

Sequential multi-product negotiation where each product has its own negotiable
quantity and per-unit price. The seller holds private tiered pricing per product.
Conversation context is preserved across products.

Price tags:
    BUYER_PRICE / SELLER_PRICE = TOTAL price for the current product order.
Quantity tags:
    ### BUYER_QUANTITY(X) ###   ### BUYER_PRICE($YYY) ###
    ### SELLER_QUANTITY(X) ###  ### SELLER_PRICE($YYY) ###

Per-unit prices are derived internally. Per-product scores are recorded.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class ProductQuantityResult:
    """Single-product negotiation result with quantity info."""

    def __init__(
        self,
        product_name: str,
        product_info: Dict[str, Any],
        status: NegotiationStatus,
        agreed_price: Optional[float] = None,
        agreed_quantity: Optional[int] = None,
        agreed_unit_price: Optional[float] = None,
        total_deal_value: Optional[float] = None,
        rounds: int = 0,
    ):
        self.product_name = product_name
        self.product_info = product_info
        self.status = status
        self.agreed_price = agreed_price
        self.agreed_quantity = agreed_quantity
        self.agreed_unit_price = agreed_unit_price
        self.total_deal_value = total_deal_value
        self.rounds = rounds


class Task15MultiProductQuantityDiscountNegotiation(BaseEnv):
    """Sequential multi-product negotiation with quantity/bulk-discount.

    For each product, both quantity and per-unit price are negotiable.
    Seller tiers may be set globally or overridden per product via reset().
    """

    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds_per_product: int = 20,
        initial_seller_unit_price: float = 10.0,
        buyer_max_unit_price: Optional[float] = None,
        buyer_target_quantity: int = 50,
        seller_min_unit_price: Optional[float] = None,
        seller_tiers: Optional[List[Tuple[int, float]]] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 0.5,
        quantity_tolerance: int = 5,
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
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.max_rounds_per_product = max_rounds_per_product
        self.initial_seller_unit_price = initial_seller_unit_price
        self.buyer_max_price = buyer_max_unit_price   # per-unit alias for scoring
        self.seller_min_price = seller_min_unit_price   # per-unit alias for scoring
        self.default_buyer_target_quantity = buyer_target_quantity
        self.default_seller_tiers = seller_tiers or [(1, initial_seller_unit_price)]
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance
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

        default_weights = {"buyer_savings": 1.0, "seller_profit": 1.0, "time_cost": 0.1}
        if reward_weights:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights

        super().__init__()

        self.memory = ConversationMemory()
        self.current_product_state = NegotiationState()
        self.current_round = 0
        self.current_product_info: Optional[Dict[str, Any]] = None
        self.current_product_name: str = ""
        self.negotiation_info = NegotiationInfo()
        self.product_results: List[ProductQuantityResult] = []
        self.current_product_index = 0

        # Per-product quantity state
        self.buyer_target_quantity: int = buyer_target_quantity
        self.seller_tiers: List[Tuple[int, float]] = self.default_seller_tiers
        self.buyer_quantity: Optional[int] = None
        self.seller_quantity: Optional[int] = None
        self.agreed_quantity: Optional[int] = None

    # ------------------------------------------------------------------
    # Reset (called once per product)
    # ------------------------------------------------------------------

    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        clear_history: bool = False,
        available_products: Optional[List[Dict[str, Any]]] = None,
        seller_tiers: Optional[List[Tuple[int, float]]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset for the next product negotiation.

        Args:
            seller_tiers: Optional per-product tiers; falls back to default.
        """
        if clear_history:
            self.memory.clear()
            self.product_results = []
            self.current_product_index = 0

        self.current_product_state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.buyer_quantity = None
        self.seller_quantity = None
        self.agreed_quantity = None

        product_info = product_info or {}
        self.current_product_info = product_info
        self.current_product_name = product_info.get("name", "Unknown Product")

        # Per-product quantity and tiers
        self.buyer_target_quantity = product_info.get("target_quantity", self.default_buyer_target_quantity)
        self.seller_tiers = seller_tiers or product_info.get("seller_tiers") or self.default_seller_tiers

        qty = self.buyer_target_quantity
        buyer_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- You need approximately {qty} units of {self.current_product_name}.\n"
            f"- In EVERY turn include: ### BUYER_QUANTITY(X) ### and ### BUYER_PRICE($YYY) ###\n"
            f"- BUYER_PRICE is the TOTAL price for the entire order.\n"
            f"- Negotiate both quantity and per-unit rate to get a better deal.\n"
        )
        seller_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- Volume discounts available — larger orders unlock lower per-unit prices.\n"
            f"- In EVERY turn include: ### SELLER_QUANTITY(X) ### and ### SELLER_PRICE($YYY) ###\n"
            f"- SELLER_PRICE is the TOTAL price for the entire order.\n"
        )
        self.buyer_agent.system_prompt_suffix = buyer_suffix
        self.seller_agent.system_prompt_suffix = seller_suffix

        total_budget = self.buyer_max_price * qty if self.buyer_max_price else None
        listed_total = qty * self.initial_seller_unit_price

        self.buyer_agent.initialize({
            "user_requirement": user_requirement,
            "max_price": total_budget,
            "target_quantity": qty,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info,
            "previous_negotiations": self._get_previous_negotiations_summary(),
        })

        tier_lines = [f"  {mq}+ units: ${up:.2f}/unit" for mq, up in sorted(self.seller_tiers)]
        self.seller_agent.initialize({
            "product_info": product_info,
            "available_products": available_products or [],
            "initial_price": listed_total,
            "min_price": self.seller_min_price * qty if self.seller_min_price else None,
            "pricing_tiers": "\n".join(tier_lines),
            "environment_info": self.environment_info,
            "previous_negotiations": self._get_previous_negotiations_summary(),
        })

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        buyer_action: Optional[str] = None,
        seller_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if buyer_action is not None:
            self.memory.add_message("buyer", buyer_action, self.current_round)
            p = self._extract_price(buyer_action)
            if p is not None:
                self.current_product_state.update(buyer_price=p)
                self.negotiation_info.buyer_price = p
                self.negotiation_info.current_price = p
            q = self._extract_quantity(buyer_action, "buyer")
            if q is not None: self.buyer_quantity = q
            elif self.buyer_quantity is None: self.buyer_quantity = self.buyer_target_quantity

        if seller_action is not None:
            self.memory.add_message("seller", seller_action, self.current_round)
            p = self._extract_price(seller_action)
            if p is not None:
                self.current_product_state.update(seller_price=p)
                self.negotiation_info.seller_price = p
                self.negotiation_info.current_price = p
            q = self._extract_quantity(seller_action, "seller")
            if q is not None: self.seller_quantity = q
            elif self.seller_quantity is None: self.seller_quantity = self.buyer_target_quantity

        terminated = truncated = False
        reward = seller_reward = buyer_reward = 0.0

        if self._check_agreement():
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            bq = self.buyer_quantity or self.buyer_target_quantity
            sq = self.seller_quantity or self.buyer_target_quantity
            bu = self._derive_unit_price(self.current_product_state.buyer_price, bq)
            su = self._derive_unit_price(self.current_product_state.seller_price, sq) if self.current_product_state.seller_price else bu
            agreed_unit = su if su <= bu else (bu + su) / 2
            self.agreed_quantity = round((bq + sq) / 2)
            agreed_total = agreed_unit * self.agreed_quantity
            self.current_product_state.update(agreed_price=agreed_total)
            self.negotiation_info.current_price = agreed_total
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            seller_reward = self._calculate_seller_reward()
            buyer_reward = self._calculate_buyer_reward()
            self._save_product_result(agreed_total)
            self.current_product_index += 1
        elif self.current_round >= self.max_rounds_per_product:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            seller_reward = self._calculate_seller_reward()
            buyer_reward = self._calculate_buyer_reward()
            self._save_product_result(None)
            self.current_product_index += 1
        else:
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round

        step_s = self._calculate_step_seller_reward()
        step_b = self._calculate_step_buyer_reward()

        observation = self._get_observation()
        info = self._get_info()
        info["step_seller_reward"] = step_s
        info["step_buyer_reward"] = step_b

        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["seller_reward"] = seller_reward
            info["buyer_reward"] = buyer_reward
            info["global_score"] = self._calculate_global_score(print_details=False)
            info["buyer_score"] = self._calculate_buyer_score(print_details=False)
            info["seller_score"] = self._calculate_seller_score(print_details=False)
            info["agreed_quantity"] = self.agreed_quantity
            aq = self.agreed_quantity or self.buyer_target_quantity
            agreed_total = self.current_product_state.agreed_price
            info["agreed_unit_price"] = round(agreed_total / aq, 4) if agreed_total else None
            info["total_deal_value"] = agreed_total
            info["product_results"] = [
                {
                    "product_name": r.product_name,
                    "status": r.status.value,
                    "agreed_quantity": r.agreed_quantity,
                    "agreed_unit_price": r.agreed_unit_price,
                    "total_deal_value": r.total_deal_value,
                    "rounds": r.rounds,
                }
                for r in self.product_results
            ]

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        lines = []
        history = self.memory.get_history()
        round_to_display = self.current_round - 1 if self.current_round > 0 else 0
        msgs = [m for m in history if m["round"] == round_to_display]
        if msgs:
            lines.append(f"\n{'='*60}")
            lines.append(f"Round {self.current_round} — {self.current_product_name}")
            lines.append(f"{'='*60}")
            for m in msgs:
                lines.append(f"\n[{m['role'].upper()}]: {m['content']}")

        lines.append(f"\n{'-'*60}")
        lines.append(f"Round {self.current_round} Summary — {self.current_product_name}")
        lines.append(f"{'-'*60}")
        bq = self.buyer_quantity or "N/A"
        sq = self.seller_quantity or "N/A"
        lines.append(f"  Buyer Qty: {bq}  |  Seller Qty: {sq}")
        if self.current_product_state.agreed_price:
            aq = self.agreed_quantity or self.buyer_target_quantity
            aup = self.current_product_state.agreed_price / aq
            lines.append(f"  Agreed: {aq} units @ ${aup:.2f}/unit = ${self.current_product_state.agreed_price:.2f}")
        else:
            bp = self.current_product_state.buyer_price
            sp = self.current_product_state.seller_price
            bq_n = self.buyer_quantity or self.buyer_target_quantity
            sq_n = self.seller_quantity or self.buyer_target_quantity
            if bp: lines.append(f"  Buyer: ${bp:.2f} total (${self._derive_unit_price(bp, bq_n):.2f}/unit)")
            if sp: lines.append(f"  Seller: ${sp:.2f} total (${self._derive_unit_price(sp, sq_n):.2f}/unit)")
        lines.append(f"{'='*60}\n")
        output = "\n".join(lines)
        if mode == "human":
            print(output)
            return None
        return output

    def close(self):
        self.memory.clear()
        self.current_product_state = NegotiationState()

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "conversation_history": self.memory.get_history(),
            "current_round": self.current_round,
            "current_product": self.current_product_name,
            "buyer_price": self.current_product_state.buyer_price,
            "seller_price": self.current_product_state.seller_price,
            "buyer_quantity": self.buyer_quantity,
            "seller_quantity": self.seller_quantity,
            "status": self.negotiation_info.status.value,
            "product_results": [(r.product_name, r.status.value) for r in self.product_results],
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "current_product": self.current_product_name,
            "buyer_price": self.current_product_state.buyer_price,
            "seller_price": self.current_product_state.seller_price,
            "agreed_price": self.current_product_state.agreed_price,
            "buyer_quantity": self.buyer_quantity,
            "seller_quantity": self.seller_quantity,
            "negotiation_info": self.negotiation_info,
        }

    def _get_previous_negotiations_summary(self) -> str:
        if not self.product_results:
            return ""
        lines = ["Previous negotiations:"]
        for r in self.product_results:
            if r.status == NegotiationStatus.AGREED and r.agreed_unit_price:
                lines.append(f"  {r.product_name}: {r.agreed_quantity} units @ ${r.agreed_unit_price:.2f}/unit = ${r.total_deal_value:.2f}")
            else:
                lines.append(f"  {r.product_name}: {r.status.value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_price(self, text: str) -> Optional[float]:
        def parse(s):
            try:
                v = float(s.replace(",", ""))
                return v if v > 0 else None
            except ValueError:
                return None
        for pat in [r'###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###',
                    r'###\s*\$([\d,]+\.?\d*)\s*###']:
            m = re.findall(pat, text, re.I)
            if m:
                p = parse(m[-1])
                if p: return p
        for pat in [r'\$([\d,]+\.?\d*)', r'([\d,]+\.?\d*)\s*dollars?', r'([\d,]+\.?\d*)\s*USD',
                    r'price.*?([\d,]+\.?\d*)', r'offer.*?([\d,]+\.?\d*)']:
            m = re.findall(pat, text, re.I)
            if m:
                p = parse(m[-1])
                if p: return p
        return None

    def _extract_quantity(self, text: str, role: str = "buyer") -> Optional[int]:
        tag = role.upper()
        m = re.findall(rf'###\s*{tag}_QUANTITY\s*\(\s*(\d+)\s*\)\s*###', text, re.I)
        if m:
            try: return int(m[-1])
            except ValueError: pass
        m = re.findall(r'###\s*(?:BUYER_QUANTITY|SELLER_QUANTITY|QUANTITY)\s*\(\s*(\d+)\s*\)\s*###', text, re.I)
        if m:
            try: return int(m[-1])
            except ValueError: pass
        return None

    def _derive_unit_price(self, total: Optional[float], qty: Optional[int]) -> Optional[float]:
        if total is None: return None
        q = qty if (qty and qty > 0) else self.buyer_target_quantity
        return total / q if q > 0 else total

    # ------------------------------------------------------------------
    # Agreement check
    # ------------------------------------------------------------------

    def _check_agreement(self) -> bool:
        if self.current_product_state.buyer_price is None or self.current_product_state.seller_price is None:
            return False
        bq = self.buyer_quantity or self.buyer_target_quantity
        sq = self.seller_quantity or self.buyer_target_quantity
        bu = self._derive_unit_price(self.current_product_state.buyer_price, bq)
        su = self._derive_unit_price(self.current_product_state.seller_price, sq)
        price_ok = (abs(bu - su) <= self.price_tolerance) or (su <= bu)
        qty_ok = abs(bq - sq) <= self.quantity_tolerance
        return price_ok and qty_ok

    def _save_product_result(self, agreed_total: Optional[float]):
        aq = self.agreed_quantity or self.buyer_target_quantity
        aup = (agreed_total / aq) if (agreed_total and aq > 0) else None
        self.product_results.append(ProductQuantityResult(
            product_name=self.current_product_name,
            product_info=self.current_product_info or {},
            status=self.negotiation_info.status,
            agreed_price=agreed_total,
            agreed_quantity=self.agreed_quantity if agreed_total else None,
            agreed_unit_price=round(aup, 4) if aup else None,
            total_deal_value=round(agreed_total, 4) if agreed_total else None,
            rounds=self.current_round,
        ))

    # ------------------------------------------------------------------
    # Reward / score calculations (per-unit ZOPA)
    # ------------------------------------------------------------------

    def _get_final_unit_price(self) -> Optional[float]:
        ap = self.current_product_state.agreed_price
        if ap is not None:
            aq = self.agreed_quantity or self.buyer_target_quantity
            return ap / aq if aq > 0 else ap
        bq = self.buyer_quantity or self.buyer_target_quantity
        sq = self.seller_quantity or self.buyer_target_quantity
        bp = self.current_product_state.buyer_price
        sp = self.current_product_state.seller_price
        if bp and sp:
            return (self._derive_unit_price(bp, bq) + self._derive_unit_price(sp, sq)) / 2
        if bp: return self._derive_unit_price(bp, bq)
        if sp: return self._derive_unit_price(sp, sq)
        return None

    def _calculate_reward(self) -> float:
        tc = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            ap = self.current_product_state.agreed_price
            if ap is None: return tc * self.reward_weights["time_cost"]
            aq = self.agreed_quantity or self.buyer_target_quantity
            aunit = ap / aq
            reward = bs = sp = 0.0
            if self.buyer_max_price: bs = (self.buyer_max_price - aunit) * aq; reward += bs * self.reward_weights["buyer_savings"]
            if self.seller_min_price: sp = (aunit - self.seller_min_price) * aq; reward += sp * self.reward_weights["seller_profit"]
            reward += tc * self.reward_weights["time_cost"]
            print(f"Reward ({self.current_product_name}) = bs({bs:.2f}) + sp({sp:.2f}) + tc({tc:.2f}) = {reward:.2f}")
            return reward
        wc = tc * self.reward_weights["time_cost"]
        print(f"Reward ({self.current_product_name}) = time_cost = {wc:.2f}")
        return wc

    def _calculate_seller_reward(self) -> float:
        tc = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            ap = self.current_product_state.agreed_price
            if ap is None: return tc * self.reward_weights["time_cost"]
            aq = self.agreed_quantity or self.buyer_target_quantity
            aunit = ap / aq
            reward = sp = 0.0
            if self.seller_min_price: sp = (aunit - self.seller_min_price) * aq; reward += sp * self.reward_weights["seller_profit"]
            return reward + tc * self.reward_weights["time_cost"]
        return tc * self.reward_weights["time_cost"]

    def _calculate_buyer_reward(self) -> float:
        tc = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            ap = self.current_product_state.agreed_price
            if ap is None: return tc * self.reward_weights["time_cost"]
            aq = self.agreed_quantity or self.buyer_target_quantity
            aunit = ap / aq
            reward = bs = 0.0
            if self.buyer_max_price: bs = (self.buyer_max_price - aunit) * aq; reward += bs * self.reward_weights["buyer_savings"]
            return reward + tc * self.reward_weights["time_cost"]
        return tc * self.reward_weights["time_cost"]

    def _calculate_step_seller_reward(self) -> float:
        rc = -self.current_round
        reward = 0.0
        sp = self.current_product_state.seller_price
        sq = self.seller_quantity or self.buyer_target_quantity
        if sp and self.seller_min_price:
            su = self._derive_unit_price(sp, sq)
            reward += (su - self.seller_min_price) * sq * self.reward_weights["seller_profit"]
        return reward + rc * self.reward_weights["time_cost"]

    def _calculate_step_buyer_reward(self) -> float:
        rc = -self.current_round
        reward = 0.0
        bp = self.current_product_state.buyer_price
        bq = self.buyer_quantity or self.buyer_target_quantity
        if bp and self.buyer_max_price:
            bu = self._derive_unit_price(bp, bq)
            reward += (self.buyer_max_price - bu) * bq * self.reward_weights["buyer_savings"]
        return reward + rc * self.reward_weights["time_cost"]

    def _calculate_global_score(self, print_details: bool = True) -> float:
        bmax = self.buyer_max_price
        smin = self.seller_min_price
        ri = max(0, self.current_round - 1)
        discount = self.gamma ** ri
        if bmax is None or smin is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[GlobalScore] buyer_max or seller_min None → {fp:.3f}")
            return fp
        Z = bmax - smin
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED
                    or self.current_product_state.agreed_price is not None)
        fu = self._get_final_unit_price()
        if fu is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[GlobalScore] No unit price → {fp:.3f}")
            return fp
        valid = Z > 0 and smin <= fu <= bmax
        if feasible and valid:
            u_b = (bmax - fu) / Z; u_s = (fu - smin) / Z
            Q = 4.0 * u_b * u_s
            gs = (self.deal_score_weight + self.quality_score_weight * Q + self.efficiency_score_weight) * discount
            if print_details:
                print(f"\n[GlobalScore] Z={Z:.2f}, unit=${fu:.4f}, Q={Q:.4f}, discount={discount:.6f}, GlobalScore={gs:.3f}")
            return gs
        fp = -self.failure_penalty_weight * (1.0 - discount)
        if print_details: print(f"\n[GlobalScore] feasible={feasible}, valid={valid} → {fp:.3f}")
        return fp

    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        bmax = self.buyer_max_price
        smin = self.seller_min_price
        ri = max(0, self.current_round - 1)
        discount = self.gamma ** ri
        if bmax is None or smin is None:
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[BuyerScore] → {bs:.3f}")
            return bs
        Z = bmax - smin
        fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED
                    or self.current_product_state.agreed_price is not None)
        if fu is None or not (Z > 0 and smin <= fu <= bmax and feasible):
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[BuyerScore] → {bs:.3f}")
            return bs
        u_b = (bmax - fu) / Z
        bs = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
        if print_details: print(f"\n[BuyerScore] u_b={u_b:.4f} → {bs:.3f}")
        return bs

    def _calculate_seller_score(self, print_details: bool = True) -> float:
        bmax = self.buyer_max_price
        smin = self.seller_min_price
        ri = max(0, self.current_round - 1)
        discount = self.gamma ** ri
        if bmax is None or smin is None:
            ss = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[SellerScore] → {ss:.3f}")
            return ss
        Z = bmax - smin
        fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED
                    or self.current_product_state.agreed_price is not None)
        if fu is None or not (Z > 0 and smin <= fu <= bmax and feasible):
            ss = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[SellerScore] → {ss:.3f}")
            return ss
        u_s = (fu - smin) / Z
        ss = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
        if print_details: print(f"\n[SellerScore] u_s={u_s:.4f} → {ss:.3f}")
        return ss

    def _print_global_score_details(self): self._calculate_global_score(print_details=True)
    def _print_buyer_score_details(self): self._calculate_buyer_score(print_details=True)
    def _print_seller_score_details(self): self._calculate_seller_score(print_details=True)
