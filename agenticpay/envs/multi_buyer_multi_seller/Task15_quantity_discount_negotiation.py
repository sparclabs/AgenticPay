"""Task15 Parallel Two-Buyer Two-Seller Quantity/Bulk-Discount Negotiation

Two buyers negotiate in parallel with two sellers. Each seller holds private
tiered pricing. Both quantity and per-unit price are negotiable. The best deal
(highest combined buyer + seller unit surplus) is selected as the final outcome.

Price tags: BUYER_PRICE / SELLER_PRICE = TOTAL price for the order.
Quantity tags: ### BUYER_QUANTITY(X) ### / ### SELLER_QUANTITY(X) ###
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task15TwoBuyerTwoSellerQuantityDiscountNegotiation(BaseEnv):
    """2B × 2S quantity/bulk-discount negotiation.

    4 conversation tracks (b1s1, b1s2, b2s1, b2s2). Agreement requires
    both unit-price and quantity convergence per track. The track with
    the highest combined unit surplus is selected as the final deal.
    """

    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_unit_price: float = 10.0,
        initial_seller2_unit_price: float = 10.5,
        buyer1_max_unit_price: Optional[float] = None,
        buyer2_max_unit_price: Optional[float] = None,
        buyer_target_quantity: int = 50,
        seller1_min_unit_price: Optional[float] = None,
        seller2_min_unit_price: Optional[float] = None,
        seller1_tiers: Optional[List[Tuple[int, float]]] = None,
        seller2_tiers: Optional[List[Tuple[int, float]]] = None,
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
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.max_rounds = max_rounds
        self.initial_seller1_unit_price = initial_seller1_unit_price
        self.initial_seller2_unit_price = initial_seller2_unit_price
        self.buyer1_max_unit_price = buyer1_max_unit_price
        self.buyer2_max_unit_price = buyer2_max_unit_price
        self.buyer_target_quantity = buyer_target_quantity
        self.seller1_min_unit_price = seller1_min_unit_price
        self.seller2_min_unit_price = seller2_min_unit_price
        self.seller1_tiers = seller1_tiers or [(1, initial_seller1_unit_price)]
        self.seller2_tiers = seller2_tiers or [(1, initial_seller2_unit_price)]
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

        # 4 memory tracks and states
        self.memory_b1s1 = ConversationMemory()
        self.memory_b1s2 = ConversationMemory()
        self.memory_b2s1 = ConversationMemory()
        self.memory_b2s2 = ConversationMemory()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()

        # Quantity tracking per track
        self._init_qty_state()

        # Final deal info
        self.selected_buyer: Optional[int] = None
        self.selected_seller: Optional[int] = None
        self.final_deal_price: Optional[float] = None
        self.agreed_quantity: Optional[int] = None

    def _init_qty_state(self):
        self.bq_b1s1 = self.bq_b1s2 = self.bq_b2s1 = self.bq_b2s2 = None
        self.sq_b1s1 = self.sq_b1s2 = self.sq_b2s1 = self.sq_b2s2 = None

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
        for mem in [self.memory_b1s1, self.memory_b1s2, self.memory_b2s1, self.memory_b2s2]:
            mem.clear()
        self.state_b1s1 = self.state_b1s2 = self.state_b2s1 = self.state_b2s2 = NegotiationState()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self._init_qty_state()
        self.selected_buyer = self.selected_seller = None
        self.final_deal_price = self.agreed_quantity = None

        product_info = product_info or {}
        qty = self.buyer_target_quantity

        buyer_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- You need approximately {qty} units. You are comparing TWO suppliers.\n"
            f"- In EVERY turn include: ### BUYER_QUANTITY(X) ### and ### BUYER_PRICE($YYY) ###\n"
            f"- BUYER_PRICE is TOTAL price for the entire order.\n"
        )
        seller_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- Volume discounts available. You negotiate with TWO buyers.\n"
            f"- In EVERY turn include: ### SELLER_QUANTITY(X) ### and ### SELLER_PRICE($YYY) ###\n"
            f"- SELLER_PRICE is TOTAL price for the entire order.\n"
        )
        for agent in [self.buyer1_agent, self.buyer2_agent]:
            agent.system_prompt_suffix = buyer_suffix
        for agent in [self.seller1_agent, self.seller2_agent]:
            agent.system_prompt_suffix = seller_suffix

        def tier_desc(tiers):
            return "\n".join(f"  {mq}+ units: ${up:.2f}/unit" for mq, up in sorted(tiers))

        self.buyer1_agent.initialize({"user_requirement": user_requirement, "max_price": self.buyer1_max_unit_price * qty if self.buyer1_max_unit_price else None, "target_quantity": qty, "user_profile": user_profile, "environment_info": self.environment_info, "product_info": product_info, "buyer_id": 1, "num_sellers": 2})
        self.buyer2_agent.initialize({"user_requirement": user_requirement, "max_price": self.buyer2_max_unit_price * qty if self.buyer2_max_unit_price else None, "target_quantity": qty, "user_profile": user_profile, "environment_info": self.environment_info, "product_info": product_info, "buyer_id": 2, "num_sellers": 2})
        self.seller1_agent.initialize({"product_info": product_info, "initial_price": self.initial_seller1_unit_price * qty, "min_price": self.seller1_min_unit_price * qty if self.seller1_min_unit_price else None, "pricing_tiers": tier_desc(self.seller1_tiers), "environment_info": self.environment_info, "seller_id": 1, "num_buyers": 2})
        self.seller2_agent.initialize({"product_info": product_info, "initial_price": self.initial_seller2_unit_price * qty, "min_price": self.seller2_min_unit_price * qty if self.seller2_min_unit_price else None, "pricing_tiers": tier_desc(self.seller2_tiers), "environment_info": self.environment_info, "seller_id": 2, "num_buyers": 2})

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _upd(self, mem, state, role, action, bq_attr, sq_attr):
        """Helper: update memory + state + quantity for one message."""
        if action is None:
            return
        mem.add_message(role, action, self.current_round)
        p = self._extract_price(action)
        if p:
            if role == "buyer":
                state.update(buyer_price=p)
            else:
                state.update(seller_price=p)
        q = self._extract_quantity(action, role)
        if q is not None:
            setattr(self, bq_attr if role == "buyer" else sq_attr, q)
        elif getattr(self, bq_attr if role == "buyer" else sq_attr) is None:
            setattr(self, bq_attr if role == "buyer" else sq_attr, self.buyer_target_quantity)

    def step(
        self,
        buyer1_action_seller1: Optional[str] = None,
        buyer1_action_seller2: Optional[str] = None,
        buyer2_action_seller1: Optional[str] = None,
        buyer2_action_seller2: Optional[str] = None,
        seller1_action_buyer1: Optional[str] = None,
        seller1_action_buyer2: Optional[str] = None,
        seller2_action_buyer1: Optional[str] = None,
        seller2_action_buyer2: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # Update buyer actions first, then seller
        self._upd(self.memory_b1s1, self.state_b1s1, "buyer", buyer1_action_seller1, "bq_b1s1", "sq_b1s1")
        self._upd(self.memory_b1s2, self.state_b1s2, "buyer", buyer1_action_seller2, "bq_b1s2", "sq_b1s2")
        self._upd(self.memory_b2s1, self.state_b2s1, "buyer", buyer2_action_seller1, "bq_b2s1", "sq_b2s1")
        self._upd(self.memory_b2s2, self.state_b2s2, "buyer", buyer2_action_seller2, "bq_b2s2", "sq_b2s2")
        self._upd(self.memory_b1s1, self.state_b1s1, "seller", seller1_action_buyer1, "bq_b1s1", "sq_b1s1")
        self._upd(self.memory_b1s2, self.state_b1s2, "seller", seller2_action_buyer1, "bq_b1s2", "sq_b1s2")
        self._upd(self.memory_b2s1, self.state_b2s1, "seller", seller1_action_buyer2, "bq_b2s1", "sq_b2s1")
        self._upd(self.memory_b2s2, self.state_b2s2, "seller", seller2_action_buyer2, "bq_b2s2", "sq_b2s2")

        # Check each track's agreement and score the best one
        tracks = [
            (1, 1, self.state_b1s1, self.bq_b1s1, self.sq_b1s1, self.buyer1_max_unit_price, self.seller1_min_unit_price),
            (1, 2, self.state_b1s2, self.bq_b1s2, self.sq_b1s2, self.buyer1_max_unit_price, self.seller2_min_unit_price),
            (2, 1, self.state_b2s1, self.bq_b2s1, self.sq_b2s1, self.buyer2_max_unit_price, self.seller1_min_unit_price),
            (2, 2, self.state_b2s2, self.bq_b2s2, self.sq_b2s2, self.buyer2_max_unit_price, self.seller2_min_unit_price),
        ]

        best_surplus = None
        for bid, sid, state, bq, sq, bmax, smin in tracks:
            if not self._check_agreement_track(state, bq, sq):
                continue
            eu = self._effective_unit_price(state, bq, sq)
            if eu is None:
                continue
            surplus = 0.0
            if bmax: surplus += bmax - eu
            if smin: surplus += eu - smin
            if best_surplus is None or surplus > best_surplus:
                best_surplus = surplus
                self.selected_buyer = bid
                self.selected_seller = sid
                self.final_deal_price, self.agreed_quantity = self._settle_track(state, bq, sq)

        terminated = truncated = False
        reward = b1r = b2r = s1r = s2r = 0.0

        if self.selected_buyer is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
        else:
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round

        observation = self._get_observation()
        info = self._get_info()

        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["global_score"] = self._calculate_global_score(print_details=False)
            info["buyer_score"] = self._calculate_buyer_score(print_details=False)
            info["seller_score"] = self._calculate_seller_score(print_details=False)
            if terminated:
                info["selected_buyer"] = self.selected_buyer
                info["selected_seller"] = self.selected_seller
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
        rd = self.current_round - 1 if self.current_round > 0 else 0
        lines.append(f"\n{'='*60}")
        lines.append(f"Round {self.current_round} — 2B×2S Negotiation")
        lines.append(f"{'='*60}")
        for label, mem, state, bq, sq in [
            ("B1-S1", self.memory_b1s1, self.state_b1s1, self.bq_b1s1, self.sq_b1s1),
            ("B1-S2", self.memory_b1s2, self.state_b1s2, self.bq_b1s2, self.sq_b1s2),
            ("B2-S1", self.memory_b2s1, self.state_b2s1, self.bq_b2s1, self.sq_b2s1),
            ("B2-S2", self.memory_b2s2, self.state_b2s2, self.bq_b2s2, self.sq_b2s2),
        ]:
            msgs = [m for m in mem.get_history() if m["round"] == rd]
            if msgs:
                lines.append(f"\n[Track {label}]:")
                for m in msgs:
                    lines.append(f"  [{m['role'].upper()}]: {m['content']}")
            if state.buyer_price:
                bu = self._derive_unit_price(state.buyer_price, bq)
                lines.append(f"  {label} Buyer: ${state.buyer_price:.2f} (${bu:.2f}/unit @ {bq or self.buyer_target_quantity})")
            if state.seller_price:
                su = self._derive_unit_price(state.seller_price, sq)
                lines.append(f"  {label} Seller: ${state.seller_price:.2f} (${su:.2f}/unit @ {sq or self.buyer_target_quantity})")
        if self.selected_buyer and self.final_deal_price:
            aq = self.agreed_quantity or self.buyer_target_quantity
            lines.append(f"\n  ✓ DEAL: B{self.selected_buyer}×S{self.selected_seller}, {aq} units @ ${self.final_deal_price/aq:.2f}/unit = ${self.final_deal_price:.2f}")
        lines.append(f"{'='*60}\n")
        output = "\n".join(lines)
        if mode == "human":
            print(output)
            return None
        return output

    def close(self):
        for mem in [self.memory_b1s1, self.memory_b1s2, self.memory_b2s1, self.memory_b2s2]:
            mem.clear()

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "conversation_history_b1s1": self.memory_b1s1.get_history(),
            "conversation_history_b1s2": self.memory_b1s2.get_history(),
            "conversation_history_b2s1": self.memory_b2s1.get_history(),
            "conversation_history_b2s2": self.memory_b2s2.get_history(),
            "current_round": self.current_round,
            "selected_buyer": self.selected_buyer,
            "selected_seller": self.selected_seller,
            "status": self.negotiation_info.status.value,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "selected_buyer": self.selected_buyer,
            "selected_seller": self.selected_seller,
            "final_deal_price": self.final_deal_price,
            "agreed_quantity": self.agreed_quantity,
            "negotiation_info": self.negotiation_info,
        }

    # ------------------------------------------------------------------
    # Extraction / agreement helpers (identical to multi_buyer)
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

    def _check_agreement_track(self, state, bq, sq) -> bool:
        if state.buyer_price is None or state.seller_price is None: return False
        bq = bq or self.buyer_target_quantity; sq = sq or self.buyer_target_quantity
        bu = self._derive_unit_price(state.buyer_price, bq)
        su = self._derive_unit_price(state.seller_price, sq)
        return ((abs(bu - su) <= self.price_tolerance) or (su <= bu)) and (abs(bq - sq) <= self.quantity_tolerance)

    def _effective_unit_price(self, state, bq, sq) -> Optional[float]:
        bq = bq or self.buyer_target_quantity; sq = sq or self.buyer_target_quantity
        if state.buyer_price is None: return None
        bu = self._derive_unit_price(state.buyer_price, bq)
        if state.seller_price is not None:
            su = self._derive_unit_price(state.seller_price, sq)
            return su if su <= bu else (bu + su) / 2
        return bu

    def _settle_track(self, state, bq, sq) -> Tuple[float, int]:
        bq = bq or self.buyer_target_quantity; sq = sq or self.buyer_target_quantity
        agreed_qty = round((bq + sq) / 2)
        bu = self._derive_unit_price(state.buyer_price, bq)
        if state.seller_price:
            su = self._derive_unit_price(state.seller_price, sq)
            aunit = su if su <= bu else (bu + su) / 2
        else:
            aunit = bu
        return aunit * agreed_qty, agreed_qty

    # ------------------------------------------------------------------
    # Reward / Score (per-unit ZOPA, selected buyer+seller ZOPA)
    # ------------------------------------------------------------------

    def _get_selected_max_unit_price(self) -> Optional[float]:
        if self.selected_buyer == 1: return self.buyer1_max_unit_price
        if self.selected_buyer == 2: return self.buyer2_max_unit_price
        return None

    def _get_selected_min_unit_price(self) -> Optional[float]:
        if self.selected_seller == 1: return self.seller1_min_unit_price
        if self.selected_seller == 2: return self.seller2_min_unit_price
        return None

    def _get_final_unit_price(self) -> Optional[float]:
        if self.final_deal_price and self.agreed_quantity:
            return self.final_deal_price / self.agreed_quantity
        return None

    def _calculate_reward(self) -> float:
        tc = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_deal_price:
            aq = self.agreed_quantity or self.buyer_target_quantity
            aunit = self.final_deal_price / aq
            reward = bs = sp = 0.0
            bmax = self._get_selected_max_unit_price()
            smin = self._get_selected_min_unit_price()
            if bmax: bs = (bmax - aunit) * aq; reward += bs * self.reward_weights["buyer_savings"]
            if smin: sp = (aunit - smin) * aq; reward += sp * self.reward_weights["seller_profit"]
            reward += tc * self.reward_weights["time_cost"]
            print(f"Global Reward = bs({bs:.2f}) + sp({sp:.2f}) + tc({tc:.2f}) = {reward:.2f} (B{self.selected_buyer}×S{self.selected_seller}, unit=${aunit:.2f}, qty={aq})")
            return reward
        wc = tc * self.reward_weights["time_cost"]
        print(f"Global Reward = time_cost = {wc:.2f}")
        return wc

    def _calculate_global_score(self, print_details: bool = True) -> float:
        bmax = self._get_selected_max_unit_price()
        smin = self._get_selected_min_unit_price()
        ri = max(0, self.current_round)
        discount = self.gamma ** ri
        if bmax is None or smin is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[GlobalScore] → {fp:.3f}")
            return fp
        Z = bmax - smin
        fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        if fu is None or not (Z > 0 and smin <= fu <= bmax and feasible):
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[GlobalScore] → {fp:.3f}")
            return fp
        u_b = (bmax - fu) / Z; u_s = (fu - smin) / Z
        gs = (self.deal_score_weight + self.quality_score_weight * 4 * u_b * u_s + self.efficiency_score_weight) * discount
        if print_details: print(f"\n[GlobalScore] Z={Z:.2f}, unit=${fu:.4f}, Q={4*u_b*u_s:.4f}, gs={gs:.3f}")
        return gs

    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        bmax = self._get_selected_max_unit_price()
        smin = self._get_selected_min_unit_price()
        ri = max(0, self.current_round)
        discount = self.gamma ** ri
        if bmax is None or smin is None:
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[BuyerScore] → {bs:.3f}")
            return bs
        Z = bmax - smin
        fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        if fu is None or not (Z > 0 and smin <= fu <= bmax and feasible):
            bs = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[BuyerScore] → {bs:.3f}")
            return bs
        u_b = (bmax - fu) / Z
        bs = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
        if print_details: print(f"\n[BuyerScore] u_b={u_b:.4f} → {bs:.3f}")
        return bs

    def _calculate_seller_score(self, print_details: bool = True) -> float:
        bmax = self._get_selected_max_unit_price()
        smin = self._get_selected_min_unit_price()
        ri = max(0, self.current_round)
        discount = self.gamma ** ri
        if bmax is None or smin is None:
            ss = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[SellerScore] → {ss:.3f}")
            return ss
        Z = bmax - smin
        fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
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
