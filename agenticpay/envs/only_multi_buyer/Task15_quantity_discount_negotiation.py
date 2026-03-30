"""Task15 Parallel Two-Buyer Quantity/Bulk-Discount Negotiation Environment

Two buyers compete in parallel for the same product. The seller holds private
tiered pricing (unit price drops with volume). Both quantity and per-unit price
are negotiable. The seller picks the buyer who offers the higher unit price.

Price tags follow the benchmark convention:
    BUYER_PRICE = TOTAL price for the order.
    Quantity tags are new:
        ### BUYER_QUANTITY(X) ###   ### BUYER_PRICE($YYY) ###
        ### SELLER_QUANTITY(X) ###  ### SELLER_PRICE($YYY) ###

Per-unit prices are derived internally: unit = total / quantity.
Scoring uses per-unit ZOPA so results are comparable across all tasks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task15ParallelTwoBuyerQuantityDiscountNegotiation(BaseEnv):
    """Two buyers compete; seller picks buyer with higher unit-price offer.

    Quantity and per-unit price are jointly negotiable. Seller holds private
    tiered pricing; buyer budgets are per-unit and kept private.
    """

    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_unit_price: float = 10.0,
        buyer1_max_unit_price: Optional[float] = None,
        buyer2_max_unit_price: Optional[float] = None,
        buyer_target_quantity: int = 50,
        seller_min_unit_price: Optional[float] = None,
        seller_tiers: Optional[List[Tuple[int, float]]] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 0.5,
        quantity_tolerance: int = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        buyer_reward_aggregation: str = "average",
        seller_reward_aggregation: str = "average",
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
        self.buyer1_agent = buyer1_agent
        self.buyer2_agent = buyer2_agent
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_unit_price = initial_seller_unit_price
        self.buyer1_max_unit_price = buyer1_max_unit_price
        self.buyer2_max_unit_price = buyer2_max_unit_price
        self.buyer_target_quantity = buyer_target_quantity
        self.seller_min_unit_price = seller_min_unit_price
        # seller_min_price alias (per-unit) used by scoring helpers
        self.seller_min_price = seller_min_unit_price
        self.seller_tiers = seller_tiers or [(1, initial_seller_unit_price)]
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance
        self.quantity_tolerance = quantity_tolerance
        self.buyer_reward_aggregation = buyer_reward_aggregation
        self.seller_reward_aggregation = seller_reward_aggregation

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

        self.memory_buyer1 = ConversationMemory()
        self.memory_buyer2 = ConversationMemory()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()

        # Quantity tracking
        self.buyer1_quantity: Optional[int] = None
        self.buyer2_quantity: Optional[int] = None
        self.seller_qty_buyer1: Optional[int] = None
        self.seller_qty_buyer2: Optional[int] = None
        self.agreed_quantity: Optional[int] = None
        self.selected_buyer: Optional[int] = None
        self.final_deal_price: Optional[float] = None   # total deal price

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
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.buyer1_quantity = None
        self.buyer2_quantity = None
        self.seller_qty_buyer1 = None
        self.seller_qty_buyer2 = None
        self.agreed_quantity = None
        self.selected_buyer = None
        self.final_deal_price = None

        product_info = product_info or {}
        qty = self.buyer_target_quantity

        buyer_qty_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- You need approximately {qty} units of this product.\n"
            f"- In EVERY turn include: ### BUYER_QUANTITY(X) ### and ### BUYER_PRICE($YYY) ###\n"
            f"- BUYER_PRICE is the TOTAL price for the entire order (quantity × unit rate).\n"
            f"- You may negotiate both quantity and per-unit rate to get a better deal.\n"
        )
        seller_qty_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- Volume discounts available — larger orders unlock lower per-unit prices.\n"
            f"- In EVERY turn include: ### SELLER_QUANTITY(X) ### and ### SELLER_PRICE($YYY) ###\n"
            f"- SELLER_PRICE is the TOTAL price for the entire order.\n"
            f"- You are negotiating with TWO buyers simultaneously. Choose the better deal.\n"
        )
        self.buyer1_agent.system_prompt_suffix = buyer_qty_suffix
        self.buyer2_agent.system_prompt_suffix = buyer_qty_suffix
        self.seller_agent.system_prompt_suffix = seller_qty_suffix

        b1_total_budget = self.buyer1_max_unit_price * qty if self.buyer1_max_unit_price else None
        b2_total_budget = self.buyer2_max_unit_price * qty if self.buyer2_max_unit_price else None
        listed_total = qty * self.initial_seller_unit_price

        self.buyer1_agent.initialize({
            "user_requirement": user_requirement,
            "max_price": b1_total_budget,
            "target_quantity": qty,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info,
            "buyer_id": 1,
        })
        self.buyer2_agent.initialize({
            "user_requirement": user_requirement,
            "max_price": b2_total_budget,
            "target_quantity": qty,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info,
            "buyer_id": 2,
        })

        tier_lines = [f"  {mq}+ units: ${up:.2f}/unit" for mq, up in sorted(self.seller_tiers)]
        tier_desc = "\n".join(tier_lines)
        self.seller_agent.initialize({
            "product_info": product_info,
            "initial_price": listed_total,
            "min_price": self.seller_min_unit_price * qty if self.seller_min_unit_price else None,
            "pricing_tiers": tier_desc,
            "environment_info": self.environment_info,
            "num_buyers": 2,
        })

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        buyer1_action: Optional[str] = None,
        buyer2_action: Optional[str] = None,
        seller_action_buyer1: Optional[str] = None,
        seller_action_buyer2: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # Process buyer1 actions
        if buyer1_action is not None:
            self.memory_buyer1.add_message("buyer", buyer1_action, self.current_round)
            p = self._extract_price(buyer1_action)
            if p is not None:
                self.state_buyer1.update(buyer_price=p)
            q = self._extract_quantity(buyer1_action, "buyer")
            if q is not None:
                self.buyer1_quantity = q
            elif self.buyer1_quantity is None:
                self.buyer1_quantity = self.buyer_target_quantity

        # Process buyer2 actions
        if buyer2_action is not None:
            self.memory_buyer2.add_message("buyer", buyer2_action, self.current_round)
            p = self._extract_price(buyer2_action)
            if p is not None:
                self.state_buyer2.update(buyer_price=p)
            q = self._extract_quantity(buyer2_action, "buyer")
            if q is not None:
                self.buyer2_quantity = q
            elif self.buyer2_quantity is None:
                self.buyer2_quantity = self.buyer_target_quantity

        # Process seller→buyer1 action
        if seller_action_buyer1 is not None:
            self.memory_buyer1.add_message("seller", seller_action_buyer1, self.current_round)
            p = self._extract_price(seller_action_buyer1)
            if p is not None:
                self.state_buyer1.update(seller_price=p)
            q = self._extract_quantity(seller_action_buyer1, "seller")
            if q is not None:
                self.seller_qty_buyer1 = q
            elif self.seller_qty_buyer1 is None:
                self.seller_qty_buyer1 = self.buyer_target_quantity

        # Process seller→buyer2 action
        if seller_action_buyer2 is not None:
            self.memory_buyer2.add_message("seller", seller_action_buyer2, self.current_round)
            p = self._extract_price(seller_action_buyer2)
            if p is not None:
                self.state_buyer2.update(seller_price=p)
            q = self._extract_quantity(seller_action_buyer2, "seller")
            if q is not None:
                self.seller_qty_buyer2 = q
            elif self.seller_qty_buyer2 is None:
                self.seller_qty_buyer2 = self.buyer_target_quantity

        # Check agreement per buyer track
        can_deal_b1 = self._check_agreement_track(
            self.state_buyer1, self.buyer1_quantity, self.seller_qty_buyer1
        ) if buyer1_action is not None else False
        can_deal_b2 = self._check_agreement_track(
            self.state_buyer2, self.buyer2_quantity, self.seller_qty_buyer2
        ) if buyer2_action is not None else False

        if can_deal_b1 or can_deal_b2:
            unit1 = self._effective_unit_price(self.state_buyer1, self.buyer1_quantity, self.seller_qty_buyer1) if can_deal_b1 else None
            unit2 = self._effective_unit_price(self.state_buyer2, self.buyer2_quantity, self.seller_qty_buyer2) if can_deal_b2 else None

            if unit1 is not None and unit2 is not None:
                chosen = 1 if unit1 >= unit2 else 2
            elif unit1 is not None:
                chosen = 1
            else:
                chosen = 2

            self.selected_buyer = chosen
            if chosen == 1:
                self.final_deal_price, self.agreed_quantity = self._settle_track(
                    self.state_buyer1, self.buyer1_quantity, self.seller_qty_buyer1
                )
            else:
                self.final_deal_price, self.agreed_quantity = self._settle_track(
                    self.state_buyer2, self.buyer2_quantity, self.seller_qty_buyer2
                )

        terminated = False
        truncated = False
        reward = buyer1_reward = buyer2_reward = seller_reward = 0.0

        if self.selected_buyer is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller_reward = self._calculate_seller_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller_reward = self._calculate_seller_reward()
        else:
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round

        step_b1 = self._calculate_step_buyer_reward(1)
        step_b2 = self._calculate_step_buyer_reward(2)
        step_s = self._calculate_step_seller_reward()

        observation = self._get_observation()
        info = self._get_info()
        info["step_buyer1_reward"] = step_b1
        info["step_buyer2_reward"] = step_b2
        info["step_seller_reward"] = step_s

        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["seller_reward"] = seller_reward
            info["global_score"] = self._calculate_global_score(print_details=False)
            info["buyer_score"] = self._calculate_buyer_score(print_details=False)
            info["seller_score"] = self._calculate_seller_score(print_details=False)
            if terminated:
                info["selected_buyer"] = self.selected_buyer
                info["final_deal_price"] = self.final_deal_price
                info["agreed_quantity"] = self.agreed_quantity
                aq = self.agreed_quantity or self.buyer_target_quantity
                info["agreed_unit_price"] = round(self.final_deal_price / aq, 4) if self.final_deal_price else None
                info["total_deal_value"] = self.final_deal_price

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        lines = []
        round_to_display = self.current_round - 1 if self.current_round > 0 else 0
        lines.append(f"\n{'='*60}")
        lines.append(f"Round {self.current_round} - Parallel Two-Buyer Negotiation")
        lines.append(f"{'='*60}")

        for bid, mem in [(1, self.memory_buyer1), (2, self.memory_buyer2)]:
            hist = mem.get_history()
            msgs = [m for m in hist if m["round"] == round_to_display]
            if msgs:
                lines.append(f"\n[BUYER {bid} Conversation]:")
                for m in msgs:
                    lines.append(f"  [{m['role'].upper()}]: {m['content']}")

        lines.append(f"\n{'-'*60}")
        lines.append(f"Round {self.current_round} Summary:")
        lines.append(f"{'-'*60}")

        for bid, state, bq, sq in [
            (1, self.state_buyer1, self.buyer1_quantity, self.seller_qty_buyer1),
            (2, self.state_buyer2, self.buyer2_quantity, self.seller_qty_buyer2),
        ]:
            lines.append(f"\nBuyer {bid}:")
            bq_d = bq or self.buyer_target_quantity
            sq_d = sq or self.buyer_target_quantity
            if state.buyer_price is not None:
                bu = self._derive_unit_price(state.buyer_price, bq)
                lines.append(f"  Buyer Price: ${state.buyer_price:.2f} total  (${bu:.2f}/unit @ qty {bq_d})")
            if state.seller_price is not None:
                su = self._derive_unit_price(state.seller_price, sq)
                lines.append(f"  Seller Price: ${state.seller_price:.2f} total  (${su:.2f}/unit @ qty {sq_d})")

        if self.selected_buyer is not None and self.final_deal_price is not None:
            aq = self.agreed_quantity or self.buyer_target_quantity
            aup = self.final_deal_price / aq
            lines.append(f"\n  ✓ DEAL with Buyer {self.selected_buyer}: {aq} units @ ${aup:.2f}/unit = ${self.final_deal_price:.2f} total")
        else:
            lines.append(f"\n  ✗ NO DEAL YET")

        status_map = {
            NegotiationStatus.ONGOING: "Ongoing",
            NegotiationStatus.AGREED: "Agreed",
            NegotiationStatus.FAILED: "Failed",
            NegotiationStatus.TIMEOUT: "Timeout",
        }
        lines.append(f"  Status: {status_map.get(self.negotiation_info.status, 'Unknown')}")
        lines.append(f"{'='*60}\n")

        output = "\n".join(lines)
        if mode == "human":
            print(output)
            return None
        return output

    def close(self):
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "conversation_history_buyer1": self.memory_buyer1.get_history(),
            "conversation_history_buyer2": self.memory_buyer2.get_history(),
            "current_round": self.current_round,
            "buyer1_price": self.state_buyer1.buyer_price,
            "seller_price_buyer1": self.state_buyer1.seller_price,
            "buyer2_price": self.state_buyer2.buyer_price,
            "seller_price_buyer2": self.state_buyer2.seller_price,
            "buyer1_quantity": self.buyer1_quantity,
            "buyer2_quantity": self.buyer2_quantity,
            "status": self.negotiation_info.status.value,
            "selected_buyer": self.selected_buyer,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "buyer1_price": self.state_buyer1.buyer_price,
            "buyer2_price": self.state_buyer2.buyer_price,
            "selected_buyer": self.selected_buyer,
            "final_deal_price": self.final_deal_price,
            "agreed_quantity": self.agreed_quantity,
            "negotiation_info": self.negotiation_info,
        }

    # ------------------------------------------------------------------
    # Price / quantity extraction
    # ------------------------------------------------------------------

    def _extract_price(self, text: str) -> Optional[float]:
        def parse(s):
            try:
                v = float(s.replace(",", ""))
                return v if v > 0 else None
            except ValueError:
                return None

        m = re.findall(r'###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###', text, re.I)
        if m:
            p = parse(m[-1])
            if p: return p
        m = re.findall(r'###\s*\$([\d,]+\.?\d*)\s*###', text, re.I)
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
        if total is None:
            return None
        q = qty if (qty is not None and qty > 0) else self.buyer_target_quantity
        return total / q if q > 0 else total

    # ------------------------------------------------------------------
    # Agreement helpers
    # ------------------------------------------------------------------

    def _check_agreement_track(
        self,
        state: NegotiationState,
        bq: Optional[int],
        sq: Optional[int],
    ) -> bool:
        if state.buyer_price is None or state.seller_price is None:
            return False
        bq = bq or self.buyer_target_quantity
        sq = sq or self.buyer_target_quantity
        buyer_unit = self._derive_unit_price(state.buyer_price, bq)
        seller_unit = self._derive_unit_price(state.seller_price, sq)
        price_ok = (abs(buyer_unit - seller_unit) <= self.price_tolerance) or (seller_unit <= buyer_unit)
        qty_ok = abs(bq - sq) <= self.quantity_tolerance
        return price_ok and qty_ok

    def _effective_unit_price(
        self,
        state: NegotiationState,
        bq: Optional[int],
        sq: Optional[int],
    ) -> Optional[float]:
        """Return the effective unit price that would be agreed for this track."""
        bq = bq or self.buyer_target_quantity
        sq = sq or self.buyer_target_quantity
        if state.buyer_price is None:
            return None
        buyer_unit = self._derive_unit_price(state.buyer_price, bq)
        if state.seller_price is not None:
            seller_unit = self._derive_unit_price(state.seller_price, sq)
            if seller_unit <= buyer_unit:
                return seller_unit
            return (buyer_unit + seller_unit) / 2
        return buyer_unit

    def _settle_track(
        self,
        state: NegotiationState,
        bq: Optional[int],
        sq: Optional[int],
    ) -> Tuple[float, int]:
        """Return (agreed_total, agreed_quantity) for an agreed track."""
        bq = bq or self.buyer_target_quantity
        sq = sq or self.buyer_target_quantity
        agreed_qty = round((bq + sq) / 2)
        buyer_unit = self._derive_unit_price(state.buyer_price, bq)
        if state.seller_price is not None:
            seller_unit = self._derive_unit_price(state.seller_price, sq)
            agreed_unit = seller_unit if seller_unit <= buyer_unit else (buyer_unit + seller_unit) / 2
        else:
            agreed_unit = buyer_unit
        return agreed_unit * agreed_qty, agreed_qty

    # ------------------------------------------------------------------
    # Reward calculations (per-unit ZOPA)
    # ------------------------------------------------------------------

    def _get_selected_buyer_max_unit_price(self) -> Optional[float]:
        if self.selected_buyer == 1:
            return self.buyer1_max_unit_price
        if self.selected_buyer == 2:
            return self.buyer2_max_unit_price
        return None

    def _get_final_unit_price(self) -> Optional[float]:
        if self.final_deal_price is not None and self.agreed_quantity:
            return self.final_deal_price / self.agreed_quantity
        return None

    def _calculate_reward(self) -> float:
        time_cost = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_deal_price is not None:
            aq = self.agreed_quantity or self.buyer_target_quantity
            agreed_unit = self.final_deal_price / aq
            reward = bs = sp = 0.0
            bmax = self._get_selected_buyer_max_unit_price()
            if bmax is not None:
                bs = (bmax - agreed_unit) * aq
                reward += bs * self.reward_weights["buyer_savings"]
            if self.seller_min_unit_price is not None:
                sp = (agreed_unit - self.seller_min_unit_price) * aq
                reward += sp * self.reward_weights["seller_profit"]
            reward += time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer{self.selected_buyer}_savings({bs:.2f}) + seller_profit({sp:.2f}) + time_cost({time_cost:.2f}) = {reward:.2f} (unit=${agreed_unit:.2f}, qty={aq})")
            return reward
        wc = time_cost * self.reward_weights["time_cost"]
        print(f"Global Reward = time_cost = {wc:.2f} (deal not reached)")
        return wc

    def _calculate_buyer_reward(self, buyer_id: int) -> float:
        time_cost = -self.current_round
        if (self.negotiation_info.status == NegotiationStatus.AGREED
                and self.selected_buyer == buyer_id
                and self.final_deal_price is not None):
            aq = self.agreed_quantity or self.buyer_target_quantity
            agreed_unit = self.final_deal_price / aq
            bmax = self.buyer1_max_unit_price if buyer_id == 1 else self.buyer2_max_unit_price
            reward = bs = 0.0
            if bmax is not None:
                bs = (bmax - agreed_unit) * aq
                reward += bs * self.reward_weights["buyer_savings"]
            reward += time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = savings({bs:.2f}) + time_cost({time_cost:.2f}) = {reward:.2f}")
            return reward
        wc = time_cost * self.reward_weights["time_cost"]
        print(f"Buyer{buyer_id} Reward = time_cost = {wc:.2f}")
        return wc

    def _calculate_seller_reward(self) -> float:
        time_cost = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_deal_price is not None:
            aq = self.agreed_quantity or self.buyer_target_quantity
            agreed_unit = self.final_deal_price / aq
            reward = sp = 0.0
            if self.seller_min_unit_price is not None:
                sp = (agreed_unit - self.seller_min_unit_price) * aq
                reward += sp * self.reward_weights["seller_profit"]
            reward += time_cost * self.reward_weights["time_cost"]
            print(f"Seller Reward = profit({sp:.2f}) + time_cost({time_cost:.2f}) = {reward:.2f}")
            return reward
        wc = time_cost * self.reward_weights["time_cost"]
        print(f"Seller Reward = time_cost = {wc:.2f}")
        return wc

    def _calculate_step_buyer_reward(self, buyer_id: int) -> float:
        round_cost = -self.current_round
        state = self.state_buyer1 if buyer_id == 1 else self.state_buyer2
        bq = (self.buyer1_quantity if buyer_id == 1 else self.buyer2_quantity) or self.buyer_target_quantity
        bmax = self.buyer1_max_unit_price if buyer_id == 1 else self.buyer2_max_unit_price
        reward = 0.0
        if state.buyer_price is not None and bmax is not None:
            buyer_unit = self._derive_unit_price(state.buyer_price, bq)
            reward += (bmax - buyer_unit) * bq * self.reward_weights["buyer_savings"]
        return reward + round_cost * self.reward_weights["time_cost"]

    def _calculate_step_seller_reward(self) -> float:
        round_cost = -self.current_round
        rewards = []
        for state, sq in [
            (self.state_buyer1, self.seller_qty_buyer1 or self.buyer_target_quantity),
            (self.state_buyer2, self.seller_qty_buyer2 or self.buyer_target_quantity),
        ]:
            if state.seller_price is not None and self.seller_min_unit_price is not None:
                su = self._derive_unit_price(state.seller_price, sq)
                rewards.append((su - self.seller_min_unit_price) * sq * self.reward_weights["seller_profit"])
        agg = sum(rewards) / len(rewards) if rewards else 0.0
        return agg + round_cost * self.reward_weights["time_cost"]

    # ------------------------------------------------------------------
    # Score calculations (per-unit ZOPA)
    # ------------------------------------------------------------------

    def _calculate_global_score(self, print_details: bool = True) -> float:
        bmax = self._get_selected_buyer_max_unit_price()
        smin = self.seller_min_unit_price
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index

        if bmax is None or smin is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[GlobalScore] buyer_max or seller_min is None → {fp:.3f}")
            return fp

        Z = bmax - smin
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        final_unit = self._get_final_unit_price()
        if final_unit is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[GlobalScore] No unit price → {fp:.3f}")
            return fp

        valid = Z > 0 and smin <= final_unit <= bmax
        if feasible and valid:
            u_b = (bmax - final_unit) / Z
            u_s = (final_unit - smin) / Z
            Q = 4.0 * u_b * u_s
            gs = (self.deal_score_weight + self.quality_score_weight * Q + self.efficiency_score_weight) * discount
            if print_details:
                print(f"\n[GlobalScore] Z={Z:.2f}, unit=${final_unit:.4f}, u_b={u_b:.4f}, u_s={u_s:.4f}, Q={Q:.4f}, discount={discount:.6f}, GlobalScore={gs:.3f}")
            return gs
        fp = -self.failure_penalty_weight * (1.0 - discount)
        if print_details: print(f"\n[GlobalScore] feasible={feasible}, valid={valid} → {fp:.3f}")
        return fp

    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        bmax = self._get_selected_buyer_max_unit_price()
        smin = self.seller_min_unit_price
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index

        if bmax is None or smin is None:
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[BuyerScore] buyer_max or seller_min is None → {bs:.3f}")
            return bs

        Z = bmax - smin
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        final_unit = self._get_final_unit_price()
        if final_unit is None:
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[BuyerScore] No unit price → {bs:.3f}")
            return bs

        valid = Z > 0 and smin <= final_unit <= bmax
        if feasible and valid:
            u_b = (bmax - final_unit) / Z
            bs = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            if print_details: print(f"\n[BuyerScore] u_b={u_b:.4f}, BuyerScore={bs:.3f}")
            return bs
        bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
        if print_details: print(f"\n[BuyerScore] → {bs:.3f}")
        return bs

    def _calculate_seller_score(self, print_details: bool = True) -> float:
        bmax = self._get_selected_buyer_max_unit_price()
        smin = self.seller_min_unit_price
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index

        if bmax is None or smin is None:
            ss = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[SellerScore] buyer_max or seller_min is None → {ss:.3f}")
            return ss

        Z = bmax - smin
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        final_unit = self._get_final_unit_price()
        if final_unit is None:
            ss = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[SellerScore] No unit price → {ss:.3f}")
            return ss

        valid = Z > 0 and smin <= final_unit <= bmax
        if feasible and valid:
            u_s = (final_unit - smin) / Z
            ss = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            if print_details: print(f"\n[SellerScore] u_s={u_s:.4f}, SellerScore={ss:.3f}")
            return ss
        ss = -self.seller_failure_penalty_weight * (1.0 - discount)
        if print_details: print(f"\n[SellerScore] → {ss:.3f}")
        return ss

    def _print_global_score_details(self):
        self._calculate_global_score(print_details=True)

    def _print_buyer_score_details(self):
        self._calculate_buyer_score(print_details=True)

    def _print_seller_score_details(self):
        self._calculate_seller_score(print_details=True)
