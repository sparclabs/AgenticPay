"""Task15 Parallel Two-Buyer Multi-Product Quantity/Bulk-Discount Negotiation

Two buyers compete for a product bundle. Prices are TOTAL for the bundle.
Quantity negotiation uses a per-product breakdown; scoring uses per-unit ZOPA
based on an effective unit price (total / total_units_in_bundle).

Price tags: BUYER_PRICE / SELLER_PRICE = TOTAL price for the entire bundle.
Quantity tags: ### BUYER_QUANTITY(X) ### = total bundle units (across all products).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task15TwoBuyerMultiProductQuantityDiscountNegotiation(BaseEnv):
    """Two buyers compete on a product bundle; seller picks buyer with higher price.

    Both quantity and per-unit price are negotiable. Seller holds private
    tiered pricing. Scoring uses per-unit ZOPA of the selected buyer.
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
        self.seller_min_price = seller_min_unit_price  # scoring alias
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
        if reward_weights: default_weights.update(reward_weights)
        self.reward_weights = default_weights

        super().__init__()

        self.memory_buyer1 = ConversationMemory()
        self.memory_buyer2 = ConversationMemory()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.buyer1_quantity: Optional[int] = None
        self.buyer2_quantity: Optional[int] = None
        self.seller_qty_buyer1: Optional[int] = None
        self.seller_qty_buyer2: Optional[int] = None
        self.agreed_quantity: Optional[int] = None
        self.selected_buyer: Optional[int] = None
        self.final_deal_price: Optional[float] = None

    # ---- Reset ----

    def reset(self, user_requirement: str = "", product_info: Optional[Dict[str, Any]] = None,
              user_profile: Optional[Any] = None, **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.memory_buyer1.clear(); self.memory_buyer2.clear()
        self.state_buyer1 = NegotiationState(); self.state_buyer2 = NegotiationState()
        self.current_round = 0; self.negotiation_info = NegotiationInfo()
        self.buyer1_quantity = self.buyer2_quantity = None
        self.seller_qty_buyer1 = self.seller_qty_buyer2 = None
        self.agreed_quantity = self.selected_buyer = self.final_deal_price = None

        product_info = product_info or {}
        qty = self.buyer_target_quantity

        buyer_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- You need approximately {qty} units of this product bundle.\n"
            f"- In EVERY turn include: ### BUYER_QUANTITY(X) ### and ### BUYER_PRICE($YYY) ###\n"
            f"- BUYER_PRICE is TOTAL price for the entire order.\n"
        )
        seller_suffix = (
            f"\nQUANTITY NEGOTIATION INSTRUCTIONS:\n"
            f"- Volume discounts available. Negotiating with TWO buyers.\n"
            f"- In EVERY turn include: ### SELLER_QUANTITY(X) ### and ### SELLER_PRICE($YYY) ###\n"
            f"- SELLER_PRICE is TOTAL price for the entire order.\n"
        )
        self.buyer1_agent.system_prompt_suffix = buyer_suffix
        self.buyer2_agent.system_prompt_suffix = buyer_suffix
        self.seller_agent.system_prompt_suffix = seller_suffix

        tier_lines = "\n".join(f"  {mq}+ units: ${up:.2f}/unit" for mq, up in sorted(self.seller_tiers))
        self.buyer1_agent.initialize({"user_requirement": user_requirement, "max_price": self.buyer1_max_unit_price * qty if self.buyer1_max_unit_price else None, "target_quantity": qty, "user_profile": user_profile, "environment_info": self.environment_info, "product_info": product_info, "buyer_id": 1})
        self.buyer2_agent.initialize({"user_requirement": user_requirement, "max_price": self.buyer2_max_unit_price * qty if self.buyer2_max_unit_price else None, "target_quantity": qty, "user_profile": user_profile, "environment_info": self.environment_info, "product_info": product_info, "buyer_id": 2})
        self.seller_agent.initialize({"product_info": product_info, "initial_price": self.initial_seller_unit_price * qty, "min_price": self.seller_min_unit_price * qty if self.seller_min_unit_price else None, "pricing_tiers": tier_lines, "environment_info": self.environment_info, "num_buyers": 2})
        return self._get_observation(), self._get_info()

    # ---- Step ----

    def step(self, buyer1_action: Optional[str] = None, buyer2_action: Optional[str] = None,
             seller_action_buyer1: Optional[str] = None, seller_action_buyer2: Optional[str] = None
             ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        def proc(mem, state, role, action, bqa, sqa):
            if action is None: return
            mem.add_message(role, action, self.current_round)
            p = self._extract_price(action)
            if p: state.update(**{"buyer_price": p} if role == "buyer" else {"seller_price": p})
            q = self._extract_quantity(action, role)
            attr = bqa if role == "buyer" else sqa
            if q: setattr(self, attr, q)
            elif getattr(self, attr) is None: setattr(self, attr, self.buyer_target_quantity)

        proc(self.memory_buyer1, self.state_buyer1, "buyer", buyer1_action, "buyer1_quantity", "seller_qty_buyer1")
        proc(self.memory_buyer2, self.state_buyer2, "buyer", buyer2_action, "buyer2_quantity", "seller_qty_buyer2")
        proc(self.memory_buyer1, self.state_buyer1, "seller", seller_action_buyer1, "buyer1_quantity", "seller_qty_buyer1")
        proc(self.memory_buyer2, self.state_buyer2, "seller", seller_action_buyer2, "buyer2_quantity", "seller_qty_buyer2")

        can_b1 = self._check_track(self.state_buyer1, self.buyer1_quantity, self.seller_qty_buyer1) if buyer1_action else False
        can_b2 = self._check_track(self.state_buyer2, self.buyer2_quantity, self.seller_qty_buyer2) if buyer2_action else False

        if can_b1 or can_b2:
            u1 = self._eff_unit(self.state_buyer1, self.buyer1_quantity, self.seller_qty_buyer1) if can_b1 else None
            u2 = self._eff_unit(self.state_buyer2, self.buyer2_quantity, self.seller_qty_buyer2) if can_b2 else None
            chosen = 1 if (u1 is not None and (u2 is None or u1 >= u2)) else 2
            self.selected_buyer = chosen
            state = self.state_buyer1 if chosen == 1 else self.state_buyer2
            bq = (self.buyer1_quantity if chosen == 1 else self.buyer2_quantity)
            sq = (self.seller_qty_buyer1 if chosen == 1 else self.seller_qty_buyer2)
            self.final_deal_price, self.agreed_quantity = self._settle(state, bq, sq)

        terminated = truncated = False
        reward = 0.0
        if self.selected_buyer is not None and self.final_deal_price is not None:
            terminated = True; self.negotiation_info.status = NegotiationStatus.AGREED
            self.current_round += 1; self.negotiation_info.round_count = self.current_round
            reward = self._calc_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True; self.negotiation_info.status = NegotiationStatus.TIMEOUT
            self.current_round += 1; self.negotiation_info.round_count = self.current_round
            reward = self._calc_reward()
        else:
            self.current_round += 1; self.negotiation_info.round_count = self.current_round

        info = self._get_info()
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["global_score"] = self._gs(False); info["buyer_score"] = self._bs(False); info["seller_score"] = self._ss(False)
            if terminated:
                info.update({"selected_buyer": self.selected_buyer, "agreed_quantity": self.agreed_quantity, "final_deal_price": self.final_deal_price})
                aq = self.agreed_quantity or self.buyer_target_quantity
                info["agreed_unit_price"] = round(self.final_deal_price / aq, 4) if self.final_deal_price else None
                info["total_deal_value"] = self.final_deal_price
        return self._get_observation(), reward, terminated, truncated, info

    def render(self, mode="human") -> Optional[str]:
        rd = self.current_round - 1 if self.current_round > 0 else 0
        lines = [f"\n{'='*60}", f"Round {self.current_round} — 2-Buyer Multi-Product", f"{'='*60}"]
        for bid, mem, state, bq, sq in [(1, self.memory_buyer1, self.state_buyer1, self.buyer1_quantity, self.seller_qty_buyer1),
                                         (2, self.memory_buyer2, self.state_buyer2, self.buyer2_quantity, self.seller_qty_buyer2)]:
            msgs = [m for m in mem.get_history() if m["round"] == rd]
            if msgs:
                lines.append(f"\n[BUYER {bid}]:")
                for m in msgs: lines.append(f"  [{m['role'].upper()}]: {m['content']}")
            if state.buyer_price: lines.append(f"  B{bid} offer: ${state.buyer_price:.2f} (${self._du(state.buyer_price, bq):.2f}/unit @ {bq or self.buyer_target_quantity})")
            if state.seller_price: lines.append(f"  S→B{bid}: ${state.seller_price:.2f} (${self._du(state.seller_price, sq):.2f}/unit @ {sq or self.buyer_target_quantity})")
        if self.selected_buyer and self.final_deal_price:
            aq = self.agreed_quantity or self.buyer_target_quantity
            lines.append(f"\n  ✓ DEAL with Buyer {self.selected_buyer}: {aq} units @ ${self.final_deal_price/aq:.2f}/unit = ${self.final_deal_price:.2f}")
        lines.append(f"{'='*60}\n")
        output = "\n".join(lines)
        if mode == "human": print(output); return None
        return output

    def close(self): self.memory_buyer1.clear(); self.memory_buyer2.clear()

    def _get_observation(self):
        return {"conversation_history_buyer1": self.memory_buyer1.get_history(),
                "conversation_history_buyer2": self.memory_buyer2.get_history(),
                "current_round": self.current_round,
                "buyer1_quantity": self.buyer1_quantity, "buyer2_quantity": self.buyer2_quantity,
                "status": self.negotiation_info.status.value, "selected_buyer": self.selected_buyer}

    def _get_info(self):
        return {"round": self.current_round, "status": self.negotiation_info.status.value,
                "selected_buyer": self.selected_buyer, "final_deal_price": self.final_deal_price,
                "agreed_quantity": self.agreed_quantity, "negotiation_info": self.negotiation_info}

    def _extract_price(self, text):
        def parse(s):
            try: v = float(s.replace(",", "")); return v if v > 0 else None
            except ValueError: return None
        for pat in [r'###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###', r'###\s*\$([\d,]+\.?\d*)\s*###']:
            m = re.findall(pat, text, re.I)
            if m:
                p = parse(m[-1])
                if p: return p
        for pat in [r'\$([\d,]+\.?\d*)', r'([\d,]+\.?\d*)\s*dollars?', r'([\d,]+\.?\d*)\s*USD', r'price.*?([\d,]+\.?\d*)', r'offer.*?([\d,]+\.?\d*)']:
            m = re.findall(pat, text, re.I)
            if m:
                p = parse(m[-1])
                if p: return p
        return None

    def _extract_quantity(self, text, role="buyer"):
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

    def _du(self, total, qty):
        if total is None: return 0.0
        q = qty if (qty and qty > 0) else self.buyer_target_quantity
        return total / q if q > 0 else total

    def _derive_unit_price(self, total, qty): return self._du(total, qty)

    def _check_track(self, state, bq, sq):
        if state.buyer_price is None or state.seller_price is None: return False
        bq = bq or self.buyer_target_quantity; sq = sq or self.buyer_target_quantity
        bu = self._du(state.buyer_price, bq); su = self._du(state.seller_price, sq)
        return ((abs(bu - su) <= self.price_tolerance) or (su <= bu)) and (abs(bq - sq) <= self.quantity_tolerance)

    def _eff_unit(self, state, bq, sq):
        bq = bq or self.buyer_target_quantity; sq = sq or self.buyer_target_quantity
        if state.buyer_price is None: return None
        bu = self._du(state.buyer_price, bq)
        if state.seller_price:
            su = self._du(state.seller_price, sq); return su if su <= bu else (bu + su) / 2
        return bu

    def _settle(self, state, bq, sq):
        bq = bq or self.buyer_target_quantity; sq = sq or self.buyer_target_quantity
        aq = round((bq + sq) / 2)
        bu = self._du(state.buyer_price, bq)
        if state.seller_price:
            su = self._du(state.seller_price, sq); aunit = su if su <= bu else (bu + su) / 2
        else: aunit = bu
        return aunit * aq, aq

    def _get_final_unit_price(self):
        if self.final_deal_price and self.agreed_quantity: return self.final_deal_price / self.agreed_quantity
        return None

    def _get_sel_max(self):
        if self.selected_buyer == 1: return self.buyer1_max_unit_price
        if self.selected_buyer == 2: return self.buyer2_max_unit_price
        return None

    def _calc_reward(self):
        tc = -self.current_round
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_deal_price:
            aq = self.agreed_quantity or self.buyer_target_quantity; aunit = self.final_deal_price / aq
            reward = bs = sp = 0.0
            bmax = self._get_sel_max()
            if bmax: bs = (bmax - aunit) * aq; reward += bs * self.reward_weights["buyer_savings"]
            if self.seller_min_unit_price: sp = (aunit - self.seller_min_unit_price) * aq; reward += sp * self.reward_weights["seller_profit"]
            reward += tc * self.reward_weights["time_cost"]
            print(f"Reward = bs({bs:.2f}) + sp({sp:.2f}) + tc({tc:.2f}) = {reward:.2f}")
            return reward
        wc = tc * self.reward_weights["time_cost"]
        print(f"Reward = tc = {wc:.2f}"); return wc

    def _score(self, bmax, smin, fp_w, deal_w, util_w, eff_w, util_fn):
        ri = max(0, self.current_round); discount = self.gamma ** ri
        if bmax is None or smin is None: return -fp_w * (1.0 - discount)
        Z = bmax - smin; fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        if fu is None or not (Z > 0 and smin <= fu <= bmax and feasible): return -fp_w * (1.0 - discount)
        u = util_fn(fu, bmax, smin, Z)
        return discount * (deal_w + util_w * u + eff_w)

    def _gs(self, pd=True):
        bmax = self._get_sel_max(); smin = self.seller_min_unit_price
        ri = max(0, self.current_round); discount = self.gamma ** ri
        if bmax is None or smin is None:
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if pd: print(f"\n[GlobalScore] → {fp:.3f}")
            return fp
        Z = bmax - smin; fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        if fu is None or not (Z > 0 and smin <= fu <= bmax and feasible):
            fp = -self.failure_penalty_weight * (1.0 - discount)
            if pd: print(f"\n[GlobalScore] → {fp:.3f}")
            return fp
        u_b = (bmax - fu) / Z; u_s = (fu - smin) / Z
        gs = (self.deal_score_weight + self.quality_score_weight * 4 * u_b * u_s + self.efficiency_score_weight) * discount
        if pd: print(f"\n[GlobalScore] Z={Z:.2f}, unit=${fu:.4f}, gs={gs:.3f}")
        return gs

    def _bs(self, pd=True):
        bmax = self._get_sel_max(); smin = self.seller_min_unit_price
        ri = max(0, self.current_round); discount = self.gamma ** ri
        if bmax is None or smin is None:
            s = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if pd: print(f"\n[BuyerScore] → {s:.3f}")
            return s
        Z = bmax - smin; fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        if fu is None or not (Z > 0 and smin <= fu <= bmax and feasible):
            s = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if pd: print(f"\n[BuyerScore] → {s:.3f}")
            return s
        u_b = (bmax - fu) / Z
        s = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
        if pd: print(f"\n[BuyerScore] u_b={u_b:.4f} → {s:.3f}")
        return s

    def _ss(self, pd=True):
        bmax = self._get_sel_max(); smin = self.seller_min_unit_price
        ri = max(0, self.current_round); discount = self.gamma ** ri
        if bmax is None or smin is None:
            s = -self.seller_failure_penalty_weight * (1.0 - discount)
            if pd: print(f"\n[SellerScore] → {s:.3f}")
            return s
        Z = bmax - smin; fu = self._get_final_unit_price()
        feasible = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        if fu is None or not (Z > 0 and smin <= fu <= bmax and feasible):
            s = -self.seller_failure_penalty_weight * (1.0 - discount)
            if pd: print(f"\n[SellerScore] → {s:.3f}")
            return s
        u_s = (fu - smin) / Z
        s = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
        if pd: print(f"\n[SellerScore] u_s={u_s:.4f} → {s:.3f}")
        return s

    def _calculate_global_score(self, print_details=True): return self._gs(print_details)
    def _calculate_buyer_score(self, print_details=True): return self._bs(print_details)
    def _calculate_seller_score(self, print_details=True): return self._ss(print_details)
    def _print_global_score_details(self): self._gs(True)
    def _print_buyer_score_details(self): self._bs(True)
    def _print_seller_score_details(self): self._ss(True)
