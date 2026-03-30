"""Task15 Multi-Product Two-Seller Quantity/Bulk-Discount Negotiation Environment

One buyer negotiates in parallel with two sellers, each selling a DIFFERENT product.
Each seller-track has its own product info, tiered pricing, quantity, and ZOPA.
The buyer picks the seller with the better effective unit-price deal relative to
that product's budget ceiling. Both quantity and per-unit price are negotiable.

Price tags follow the benchmark convention:
    BUYER_PRICE / SELLER_PRICE = TOTAL price for the order.
Quantity tags:
    ### BUYER_QUANTITY(X) ###   ### BUYER_PRICE($YYY) ###
    ### SELLER_QUANTITY(X) ###  ### SELLER_PRICE($YYY) ###

Per-unit prices are derived internally: unit = total / quantity.
Scoring uses the per-unit ZOPA of the selected seller's product.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task15MultiProductTwoSellerQuantityDiscountNegotiation(BaseEnv):
    """One buyer vs two sellers each with a different product.

    Each track has its own ZOPA (product-specific buyer_max / seller_min),
    tiered pricing, and target quantity. Buyer picks the track with the best
    surplus relative to that product's ceiling price.
    """

    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_unit_price: float = 10.0,
        initial_seller2_unit_price: float = 40.0,
        buyer_max_unit_price_s1: Optional[float] = None,
        buyer_max_unit_price_s2: Optional[float] = None,
        buyer_target_quantity_s1: int = 50,
        buyer_target_quantity_s2: int = 5,
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
        self.buyer_agent = buyer_agent
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.max_rounds = max_rounds
        self.initial_seller1_unit_price = initial_seller1_unit_price
        self.initial_seller2_unit_price = initial_seller2_unit_price
        self.buyer_max_unit_price_s1 = buyer_max_unit_price_s1
        self.buyer_max_unit_price_s2 = buyer_max_unit_price_s2
        self.buyer_target_quantity_s1 = buyer_target_quantity_s1
        self.buyer_target_quantity_s2 = buyer_target_quantity_s2
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

        self.memory_seller1 = ConversationMemory()
        self.memory_seller2 = ConversationMemory()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()

        # Per-track quantity state
        self.buyer_qty_s1: Optional[int] = None
        self.buyer_qty_s2: Optional[int] = None
        self.seller1_quantity: Optional[int] = None
        self.seller2_quantity: Optional[int] = None
        self.agreed_quantity: Optional[int] = None
        self.selected_seller: Optional[int] = None
        self.final_deal_price: Optional[float] = None

        # Product info per seller
        self.seller1_product_info: Dict[str, Any] = {}
        self.seller2_product_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        user_requirement: str = "",
        seller1_product_info: Optional[Dict[str, Any]] = None,
        seller2_product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.memory_seller1.clear()
        self.memory_seller2.clear()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.buyer_qty_s1 = None
        self.buyer_qty_s2 = None
        self.seller1_quantity = None
        self.seller2_quantity = None
        self.agreed_quantity = None
        self.selected_seller = None
        self.final_deal_price = None

        self.seller1_product_info = seller1_product_info or {}
        self.seller2_product_info = seller2_product_info or {}

        qty1 = self.buyer_target_quantity_s1
        qty2 = self.buyer_target_quantity_s2
        budget1 = self.buyer_max_unit_price_s1 * qty1 if self.buyer_max_unit_price_s1 else None
        budget2 = self.buyer_max_unit_price_s2 * qty2 if self.buyer_max_unit_price_s2 else None

        buyer_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- You are comparing TWO suppliers each selling a DIFFERENT product.\n"
            f"- Seller 1 sells: {self.seller1_product_info.get('name', 'Product 1')} "
            f"(need ~{qty1} units).\n"
            f"- Seller 2 sells: {self.seller2_product_info.get('name', 'Product 2')} "
            f"(need ~{qty2} units).\n"
            f"- In EVERY turn include: ### BUYER_QUANTITY(X) ### and ### BUYER_PRICE($YYY) ###\n"
            f"- BUYER_PRICE is the TOTAL price for that product order.\n"
        )
        seller_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- Volume discounts available — larger orders unlock lower per-unit prices.\n"
            f"- In EVERY turn include: ### SELLER_QUANTITY(X) ### and ### SELLER_PRICE($YYY) ###\n"
            f"- SELLER_PRICE is the TOTAL price for the entire order.\n"
        )
        self.buyer_agent.system_prompt_suffix = buyer_suffix
        self.seller1_agent.system_prompt_suffix = seller_suffix
        self.seller2_agent.system_prompt_suffix = seller_suffix

        def tier_desc(tiers):
            return "\n".join(f"  {mq}+ units: ${up:.2f}/unit" for mq, up in sorted(tiers))

        self.buyer_agent.initialize({
            "user_requirement": user_requirement,
            "max_price": (budget1 or 0) + (budget2 or 0),
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
            "num_sellers": 2,
        })
        self.seller1_agent.initialize({
            "product_info": self.seller1_product_info,
            "initial_price": self.initial_seller1_unit_price * qty1,
            "min_price": self.seller1_min_unit_price * qty1 if self.seller1_min_unit_price else None,
            "pricing_tiers": tier_desc(self.seller1_tiers),
            "environment_info": self.environment_info,
            "seller_id": 1,
        })
        self.seller2_agent.initialize({
            "product_info": self.seller2_product_info,
            "initial_price": self.initial_seller2_unit_price * qty2,
            "min_price": self.seller2_min_unit_price * qty2 if self.seller2_min_unit_price else None,
            "pricing_tiers": tier_desc(self.seller2_tiers),
            "environment_info": self.environment_info,
            "seller_id": 2,
        })

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        buyer_action_seller1: Optional[str] = None,
        buyer_action_seller2: Optional[str] = None,
        seller1_action: Optional[str] = None,
        seller2_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if buyer_action_seller1 is not None:
            self.memory_seller1.add_message("buyer", buyer_action_seller1, self.current_round)
            p = self._extract_price(buyer_action_seller1)
            if p: self.state_seller1.update(buyer_price=p)
            q = self._extract_quantity(buyer_action_seller1, "buyer")
            if q: self.buyer_qty_s1 = q
            elif self.buyer_qty_s1 is None: self.buyer_qty_s1 = self.buyer_target_quantity_s1

        if buyer_action_seller2 is not None:
            self.memory_seller2.add_message("buyer", buyer_action_seller2, self.current_round)
            p = self._extract_price(buyer_action_seller2)
            if p: self.state_seller2.update(buyer_price=p)
            q = self._extract_quantity(buyer_action_seller2, "buyer")
            if q: self.buyer_qty_s2 = q
            elif self.buyer_qty_s2 is None: self.buyer_qty_s2 = self.buyer_target_quantity_s2

        if seller1_action is not None:
            self.memory_seller1.add_message("seller", seller1_action, self.current_round)
            p = self._extract_price(seller1_action)
            if p: self.state_seller1.update(seller_price=p)
            q = self._extract_quantity(seller1_action, "seller")
            if q: self.seller1_quantity = q
            elif self.seller1_quantity is None: self.seller1_quantity = self.buyer_target_quantity_s1

        if seller2_action is not None:
            self.memory_seller2.add_message("seller", seller2_action, self.current_round)
            p = self._extract_price(seller2_action)
            if p: self.state_seller2.update(seller_price=p)
            q = self._extract_quantity(seller2_action, "seller")
            if q: self.seller2_quantity = q
            elif self.seller2_quantity is None: self.seller2_quantity = self.buyer_target_quantity_s2

        can_s1 = self._check_agreement_track(
            self.state_seller1, self.buyer_qty_s1, self.seller1_quantity,
            self.buyer_target_quantity_s1
        ) if buyer_action_seller1 is not None else False
        can_s2 = self._check_agreement_track(
            self.state_seller2, self.buyer_qty_s2, self.seller2_quantity,
            self.buyer_target_quantity_s2
        ) if buyer_action_seller2 is not None else False

        if can_s1 or can_s2:
            # Choose seller with better surplus relative to product budget
            sur1 = self._track_surplus(
                self.state_seller1, self.buyer_qty_s1, self.seller1_quantity,
                self.buyer_max_unit_price_s1, self.seller1_min_unit_price,
                self.buyer_target_quantity_s1
            ) if can_s1 else None
            sur2 = self._track_surplus(
                self.state_seller2, self.buyer_qty_s2, self.seller2_quantity,
                self.buyer_max_unit_price_s2, self.seller2_min_unit_price,
                self.buyer_target_quantity_s2
            ) if can_s2 else None

            if sur1 is not None and sur2 is not None:
                chosen = 1 if sur1 >= sur2 else 2
            elif sur1 is not None:
                chosen = 1
            else:
                chosen = 2

            self.selected_seller = chosen
            if chosen == 1:
                self.final_deal_price, self.agreed_quantity = self._settle_track(
                    self.state_seller1, self.buyer_qty_s1, self.seller1_quantity,
                    self.buyer_target_quantity_s1
                )
                self.buyer_max_price = self.buyer_max_unit_price_s1
                self.seller_min_price = self.seller1_min_unit_price
            else:
                self.final_deal_price, self.agreed_quantity = self._settle_track(
                    self.state_seller2, self.buyer_qty_s2, self.seller2_quantity,
                    self.buyer_target_quantity_s2
                )
                self.buyer_max_price = self.buyer_max_unit_price_s2
                self.seller_min_price = self.seller2_min_unit_price

        terminated = truncated = False
        reward = seller1_reward = seller2_reward = buyer_reward = 0.0

        if self.selected_seller is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
            buyer_reward = self._calculate_buyer_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
            buyer_reward = self._calculate_buyer_reward()
        else:
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round

        observation = self._get_observation()
        info = self._get_info()

        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["buyer_reward"] = buyer_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
            info["global_score"] = self._calculate_global_score(print_details=False)
            info["buyer_score"] = self._calculate_buyer_score(print_details=False)
            info["seller_score"] = self._calculate_seller_score(print_details=False)
            if terminated:
                info["selected_seller"] = self.selected_seller
                info["final_deal_price"] = self.final_deal_price
                info["agreed_quantity"] = self.agreed_quantity
                tgt = self.buyer_target_quantity_s1 if self.selected_seller == 1 else self.buyer_target_quantity_s2
                aq = self.agreed_quantity or tgt
                info["agreed_unit_price"] = round(self.final_deal_price / aq, 4) if self.final_deal_price else None
                info["total_deal_value"] = self.final_deal_price
                info["selected_product"] = (
                    self.seller1_product_info.get("name") if self.selected_seller == 1
                    else self.seller2_product_info.get("name")
                )

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        lines = []
        round_to_display = self.current_round - 1 if self.current_round > 0 else 0
        lines.append(f"\n{'='*60}")
        lines.append(f"Round {self.current_round} - Multi-Product Two-Seller Negotiation")
        lines.append(f"{'='*60}")
        for sid, mem, pinfo in [
            (1, self.memory_seller1, self.seller1_product_info),
            (2, self.memory_seller2, self.seller2_product_info),
        ]:
            hist = [m for m in mem.get_history() if m["round"] == round_to_display]
            if hist:
                lines.append(f"\n[SELLER {sid} — {pinfo.get('name', 'Product')}]:")
                for m in hist:
                    lines.append(f"  [{m['role'].upper()}]: {m['content']}")
        lines.append(f"\n{'-'*60}")
        for sid, state, bq, sq, tgt in [
            (1, self.state_seller1, self.buyer_qty_s1, self.seller1_quantity, self.buyer_target_quantity_s1),
            (2, self.state_seller2, self.buyer_qty_s2, self.seller2_quantity, self.buyer_target_quantity_s2),
        ]:
            lines.append(f"\nSeller {sid} ({self.seller1_product_info.get('name') if sid==1 else self.seller2_product_info.get('name')}):")
            if state.buyer_price:
                bu = self._derive_unit_price(state.buyer_price, bq, tgt)
                lines.append(f"  Buyer: ${state.buyer_price:.2f} total  (${bu:.2f}/unit @ qty {bq or tgt})")
            if state.seller_price:
                su = self._derive_unit_price(state.seller_price, sq, tgt)
                lines.append(f"  Seller: ${state.seller_price:.2f} total  (${su:.2f}/unit @ qty {sq or tgt})")
        if self.selected_seller and self.final_deal_price:
            tgt = self.buyer_target_quantity_s1 if self.selected_seller == 1 else self.buyer_target_quantity_s2
            aq = self.agreed_quantity or tgt
            aup = self.final_deal_price / aq
            lines.append(f"\n  ✓ DEAL with Seller {self.selected_seller}: {aq} units @ ${aup:.2f}/unit = ${self.final_deal_price:.2f}")
        else:
            lines.append(f"\n  ✗ NO DEAL YET")
        lines.append(f"{'='*60}\n")
        output = "\n".join(lines)
        if mode == "human":
            print(output)
            return None
        return output

    def close(self):
        self.memory_seller1.clear()
        self.memory_seller2.clear()

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "conversation_history_seller1": self.memory_seller1.get_history(),
            "conversation_history_seller2": self.memory_seller2.get_history(),
            "current_round": self.current_round,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_qty_s1": self.buyer_qty_s1,
            "buyer_qty_s2": self.buyer_qty_s2,
            "status": self.negotiation_info.status.value,
            "selected_seller": self.selected_seller,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "selected_seller": self.selected_seller,
            "final_deal_price": self.final_deal_price,
            "agreed_quantity": self.agreed_quantity,
            "negotiation_info": self.negotiation_info,
        }

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
        for pat in [
            r'###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###',
            r'###\s*\$([\d,]+\.?\d*)\s*###',
        ]:
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

    def _derive_unit_price(self, total: Optional[float], qty: Optional[int], default_qty: int) -> Optional[float]:
        if total is None: return None
        q = qty if (qty and qty > 0) else default_qty
        return total / q if q > 0 else total

    # ------------------------------------------------------------------
    # Agreement / selection helpers
    # ------------------------------------------------------------------

    def _check_agreement_track(self, state, bq, sq, default_qty) -> bool:
        if state.buyer_price is None or state.seller_price is None: return False
        bq = bq or default_qty
        sq = sq or default_qty
        bu = self._derive_unit_price(state.buyer_price, bq, default_qty)
        su = self._derive_unit_price(state.seller_price, sq, default_qty)
        return ((abs(bu - su) <= self.price_tolerance) or (su <= bu)) and (abs(bq - sq) <= self.quantity_tolerance)

    def _track_surplus(self, state, bq, sq, bmax, smin, default_qty) -> Optional[float]:
        bq = bq or default_qty
        sq = sq or default_qty
        if state.buyer_price is None: return None
        bu = self._derive_unit_price(state.buyer_price, bq, default_qty)
        if state.seller_price is not None:
            su = self._derive_unit_price(state.seller_price, sq, default_qty)
            eff = su if su <= bu else (bu + su) / 2
        else:
            eff = bu
        b_sur = (bmax - eff) if bmax else 0.0
        s_sur = (eff - smin) if smin else 0.0
        return b_sur + s_sur

    def _settle_track(self, state, bq, sq, default_qty) -> Tuple[float, int]:
        bq = bq or default_qty
        sq = sq or default_qty
        agreed_qty = round((bq + sq) / 2)
        bu = self._derive_unit_price(state.buyer_price, bq, default_qty)
        if state.seller_price is not None:
            su = self._derive_unit_price(state.seller_price, sq, default_qty)
            agreed_unit = su if su <= bu else (bu + su) / 2
        else:
            agreed_unit = bu
        return agreed_unit * agreed_qty, agreed_qty

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_final_unit_price(self) -> Optional[float]:
        if self.final_deal_price and self.agreed_quantity:
            return self.final_deal_price / self.agreed_quantity
        return None

    def _get_selected_bmax(self) -> Optional[float]:
        if self.selected_seller == 1: return self.buyer_max_unit_price_s1
        if self.selected_seller == 2: return self.buyer_max_unit_price_s2
        return None

    def _get_selected_smin(self) -> Optional[float]:
        if self.selected_seller == 1: return self.seller1_min_unit_price
        if self.selected_seller == 2: return self.seller2_min_unit_price
        return None

    def _get_selected_default_qty(self) -> int:
        if self.selected_seller == 1: return self.buyer_target_quantity_s1
        return self.buyer_target_quantity_s2

    def _calculate_reward(self) -> float:
        tc = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_deal_price:
            aq = self.agreed_quantity or self._get_selected_default_qty()
            aunit = self.final_deal_price / aq
            reward = bs = sp = 0.0
            bmax = self._get_selected_bmax()
            smin = self._get_selected_smin()
            if bmax: bs = (bmax - aunit) * aq; reward += bs * self.reward_weights["buyer_savings"]
            if smin: sp = (aunit - smin) * aq; reward += sp * self.reward_weights["seller_profit"]
            reward += tc * self.reward_weights["time_cost"]
            print(f"Global Reward = bs({bs:.2f}) + sp({sp:.2f}) + tc({tc:.2f}) = {reward:.2f} (unit=${aunit:.2f}, qty={aq})")
            return reward
        wc = tc * self.reward_weights["time_cost"]
        print(f"Global Reward = time_cost = {wc:.2f}")
        return wc

    def _calculate_buyer_reward(self) -> float:
        tc = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_deal_price:
            aq = self.agreed_quantity or self._get_selected_default_qty()
            aunit = self.final_deal_price / aq
            bmax = self._get_selected_bmax()
            reward = 0.0
            if bmax: reward += (bmax - aunit) * aq * self.reward_weights["buyer_savings"]
            reward += tc * self.reward_weights["time_cost"]
            return reward
        return tc * self.reward_weights["time_cost"]

    def _calculate_seller_reward(self, seller_id: int) -> float:
        tc = -self.current_round
        if (self.negotiation_info.status == NegotiationStatus.AGREED
                and self.selected_seller == seller_id
                and self.final_deal_price):
            aq = self.agreed_quantity or self._get_selected_default_qty()
            aunit = self.final_deal_price / aq
            smin = self.seller1_min_unit_price if seller_id == 1 else self.seller2_min_unit_price
            reward = 0.0
            if smin: reward += (aunit - smin) * aq * self.reward_weights["seller_profit"]
            reward += tc * self.reward_weights["time_cost"]
            return reward
        return tc * self.reward_weights["time_cost"]

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def _calculate_global_score(self, print_details: bool = True) -> float:
        bmax = self._get_selected_bmax()
        smin = self._get_selected_smin()
        ri = max(0, self.current_round)
        discount = self.gamma ** ri
        if bmax is None or smin is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[GlobalScore] bmax or smin None → {fp:.3f}")
            return fp
        Z = bmax - smin
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        fu = self._get_final_unit_price()
        if fu is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if print_details: print(f"\n[GlobalScore] No unit price → {fp:.3f}")
            return fp
        valid = Z > 0 and smin <= fu <= bmax
        if feasible and valid:
            u_b = (bmax - fu) / Z; u_s = (fu - smin) / Z
            gs = (self.deal_score_weight + self.quality_score_weight * 4 * u_b * u_s + self.efficiency_score_weight) * discount
            if print_details: print(f"\n[GlobalScore] Z={Z:.2f}, unit=${fu:.4f}, Q={4*u_b*u_s:.4f}, gs={gs:.3f}")
            return gs
        fp = -self.failure_penalty_weight * (1.0 - discount)
        if print_details: print(f"\n[GlobalScore] feasible={feasible}, valid={valid} → {fp:.3f}")
        return fp

    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        bmax = self._get_selected_bmax()
        smin = self._get_selected_smin()
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
        bmax = self._get_selected_bmax()
        smin = self._get_selected_smin()
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
