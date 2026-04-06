"""Multi-Parameter Seller Agent

Seller agent that negotiates on multiple parameters simultaneously:
price, quality, delivery timeline, warranty period, and payment schedule.
The active subset is controlled by `active_params`.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from agenticpay.agents.base_agent import BaseAgent
from agenticpay.models.base_llm import BaseLLM
from agenticpay.models.base_vlm import BaseVLM
from agenticpay.envs.multi_param.base_multi_param_negotiation import ParamPreferences


# Ordinal encodings (must match env)
QUALITY_ORDER  = {"Standard": 0, "Premium": 1, "Luxury": 2}
PAYMENT_ORDER  = {"upfront": 0, "30-day": 1, "installments": 2}

# Human-readable labels for the guidance block
_PARAM_LABELS = {
    "price":            "Price",
    "quality":          "Quality",
    "delivery_days":    "Delivery",
    "warranty_months":  "Warranty",
    "payment":          "Payment",
}

def _seller_opening_hint(param: str, prefs: "ParamPreferences") -> str:
    """One-line description of seller's opening value for a non-price parameter."""
    if param == "quality":
        return f"{prefs.min_quality} — do NOT offer higher quality unless buyer raises price"
    elif param == "delivery_days":
        return f"{prefs.max_delivery_days} days — your standard lead time; faster delivery requires a price premium"
    elif param == "warranty_months":
        return f"{prefs.min_warranty_months} months — your standard warranty; longer coverage requires a higher price"
    elif param == "payment":
        return f"{prefs.preferred_payment} — only concede to less favorable terms for a meaningful price gain"
    return ""


# Textual direction for each param (seller perspective)
_SELLER_DIRECTION = {
    "price":            "higher is better (charge more); if you offer higher quality, demand a higher price",
    "quality":          "Standard is cheapest to deliver; Premium and Luxury cost you more — only offer them if the price justifies it",
    "delivery_days":    "more days is better (standard lead time, less rush cost); rush delivery requires a price premium",
    "warranty_months":  "fewer months is better (less support liability); longer warranty requires a higher price",
    "payment":          "upfront > 30-day > installments (immediate cash is better)",
}



class MultiParamSellerAgent(BaseAgent):
    """Seller agent for multi-parameter negotiation tasks.

    Symmetric counterpart to MultiParamBuyerAgent. Emits SELLER_OFFER tags
    with only the parameters active in this task.
    """

    def __init__(
        self,
        model: Union[BaseLLM, BaseVLM],
        preferences: ParamPreferences,
        active_params: List[str],
        name: str = "Seller",
        role_description: str = (
            "You are a seller aiming to close a profitable deal. "
            "You are professional, flexible, and willing to trade off "
            "one parameter to protect another."
        ),
        system_prompt_suffix: Optional[str] = None,
    ):
        """
        Args:
            model: LLM or VLM interface.
            preferences: Seller's reservation values and utility weights (confidential).
            active_params: Which parameters are negotiated in this task.
            name: Agent display name.
            role_description: Role description used in system prompt.
            system_prompt_suffix: Optional personality text appended to guidance.
        """
        super().__init__(model, role_description, name)
        self.preferences   = preferences
        self.active_params = active_params
        self.system_prompt_suffix = system_prompt_suffix
        self.prompt_log: List[Dict[str, Any]] = []

    def respond(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        """Generate seller's negotiation response."""
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")

        prompt = self._build_prompt(conversation_history, current_state)
        guidance = self._build_guidance(current_state)
        full_prompt = prompt + guidance
        self.prompt_log.append({
            "round": current_state.get("current_round", 0),
            "role": "seller",
            "prompt": full_prompt,
        })

        images = None
        if self.is_vlm:
            images = current_state.get("images") or current_state.get("product_images")
            if images is None:
                images = self.context.get("images") or self.context.get("product_images")

        if self.is_vlm and images is not None:
            response = self.model.generate(full_prompt, images=images, temperature=0.0, max_tokens=1024)
        else:
            response = self.model.generate(full_prompt, temperature=0.0, max_tokens=1024)

        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()

    # ------------------------------------------------------------------
    # Utility function (seller perspective)
    # ------------------------------------------------------------------

    def _compute_utility(self, offer_dict: Optional[Dict]) -> Tuple[float, Dict[str, float]]:
        """Compute seller utility U = Σ (w_k/Σw) × u_k(x_k) for an offer.

        Returns (total_utility, per_param_utility).
        All per-param utilities are in [0, 1] where 1 = ideal for the seller.
        Seller formulas are mirrors of buyer formulas.
        """
        if not offer_dict:
            return 0.0, {}

        neg_params = self.context.get("negotiation_params", {})
        p_range = neg_params.get("price_range", (18000, 28000))
        d_range = neg_params.get("delivery_range", (1, 60))
        w_range = neg_params.get("warranty_range", (12, 48))

        raw_weights = {p: self.preferences.weight_for(p) for p in self.active_params}
        total_weight = sum(raw_weights.values())
        if total_weight == 0:
            return 0.0, {}

        per_param: Dict[str, float] = {}
        for p in self.active_params:
            val = offer_dict.get(p)
            if val is None:
                per_param[p] = 0.0
                continue
            if p == "price":
                # Seller wants high price
                p_min, p_max = p_range
                span = p_max - p_min
                per_param[p] = max(0.0, min(1.0, (val - p_min) / span)) if span > 0 else 0.0
            elif p == "quality":
                # Seller wants low quality (cheapest to deliver)
                per_param[p] = 1.0 - QUALITY_ORDER.get(str(val), 0) / 2.0
            elif p == "delivery_days":
                # Seller wants more days (standard lead time)
                d_min, d_max = d_range
                span = d_max - d_min
                per_param[p] = max(0.0, min(1.0, (val - d_min) / span)) if span > 0 else 0.0
            elif p == "warranty_months":
                # Seller wants fewer months (less service liability)
                w_min, w_max = w_range
                span = w_max - w_min
                per_param[p] = max(0.0, min(1.0, (w_max - val) / span)) if span > 0 else 0.0
            elif p == "payment":
                # Seller wants upfront
                per_param[p] = 1.0 - PAYMENT_ORDER.get(str(val), 0) / 2.0
            else:
                per_param[p] = 0.0

        total = sum((raw_weights[p] / total_weight) * per_param[p] for p in self.active_params)
        return round(total, 4), {k: round(v, 4) for k, v in per_param.items()}

    def _compute_reservation_utility(self) -> Tuple[float, Dict[str, float]]:
        """Utility at the seller's own reservation values (minimum acceptable floor)."""
        prefs = self.preferences
        reservation: Dict[str, Any] = {}
        for p in self.active_params:
            if p == "price":
                reservation[p] = prefs.price_limit
            elif p == "quality":
                reservation[p] = prefs.min_quality
            elif p == "delivery_days":
                reservation[p] = prefs.max_delivery_days
            elif p == "warranty_months":
                reservation[p] = prefs.min_warranty_months
            elif p == "payment":
                reservation[p] = prefs.preferred_payment
        return self._compute_utility(reservation)

    def _compute_marginal_utility(self) -> Dict[str, float]:
        """Marginal utility per unit change per parameter (normalized weights).

        Returns a dict mapping param → Δu per unit change (per $1 for price,
        per tier step for quality, per day for delivery_days, etc.).
        """
        neg_params = self.context.get("negotiation_params", {})
        p_range = neg_params.get("price_range", (18000, 28000))
        d_range = neg_params.get("delivery_range", (1, 60))
        w_range = neg_params.get("warranty_range", (12, 48))

        raw_weights = {p: self.preferences.weight_for(p) for p in self.active_params}
        total_weight = sum(raw_weights.values())
        if total_weight == 0:
            return {}

        marginal: Dict[str, float] = {}
        for p in self.active_params:
            nw = raw_weights[p] / total_weight
            if p == "price":
                span = p_range[1] - p_range[0]
                marginal[p] = nw / span if span > 0 else 0.0   # Δu per $1 higher price
            elif p == "quality":
                marginal[p] = nw * 0.5                          # Δu per tier step lower
            elif p == "delivery_days":
                span = d_range[1] - d_range[0]
                marginal[p] = nw / span if span > 0 else 0.0   # Δu per 1 extra day
            elif p == "warranty_months":
                span = w_range[1] - w_range[0]
                marginal[p] = nw / span if span > 0 else 0.0   # Δu per 1 month shorter
            elif p == "payment":
                marginal[p] = nw * 0.5                          # Δu per step toward upfront
            else:
                marginal[p] = 0.0
        return marginal

    # ------------------------------------------------------------------
    # Guidance block construction
    # ------------------------------------------------------------------

    def _build_guidance(self, current_state: Dict[str, Any]) -> str:
        prefs = self.preferences
        env_info = self.context.get("environment_info", {})
        product_info = self.context.get("product_info", {})
        neg_params = self.context.get("negotiation_params", {})

        personality = f"\n{self.system_prompt_suffix}\n" if self.system_prompt_suffix else ""

        reservation_lines = self._reservation_lines()

        directions = []
        for p in self.active_params:
            directions.append(f"  - {_PARAM_LABELS[p]}: {_SELLER_DIRECTION[p]}")

        tag_example = self._build_tag_example()

        valid_values = []
        if "quality" in self.active_params:
            opts = neg_params.get("quality_options", ["Standard", "Premium", "Luxury"])
            valid_values.append(f"  - Quality: {' | '.join(opts)}")
        if "payment" in self.active_params:
            opts = neg_params.get("payment_options", ["upfront", "30-day", "installments"])
            valid_values.append(f"  - Payment: {' | '.join(opts)}")
        if "delivery_days" in self.active_params:
            d_range = neg_params.get("delivery_range", (1, 60))
            valid_values.append(f"  - Delivery: {d_range[0]}–{d_range[1]} days (integer)")
        if "warranty_months" in self.active_params:
            w_range = neg_params.get("warranty_range", (1, 48))
            valid_values.append(f"  - Warranty: {w_range[0]}–{w_range[1]} months (integer)")

        valid_block = "\n".join(valid_values) if valid_values else "  (no categorical constraints)"

        # Utility analysis block
        utility_block = self._utility_analysis_block(current_state)

        guidance = f"""
IMPORTANT REMINDERS:
- Current product: {product_info}
- Consider the environment: {env_info}
{personality}
YOUR CONFIDENTIAL RESERVATION VALUES (do NOT reveal to the buyer):
{chr(10).join(reservation_lines)}

NEGOTIATION DIRECTIONS (you want → buyer wants the opposite):
{chr(10).join(directions)}

TRADE-OFF STRATEGY:
- Parameters are linked — adjust them together, not in isolation.
  Examples:
    * If buyer demands Premium quality, raise your price offer to compensate.
    * If buyer demands faster delivery, raise your price to cover rush costs.
    * If buyer wants a longer warranty, increase your price to offset support costs.
- Never accept a deal where price < ${prefs.price_limit:,.2f} (your hard floor).
- Make gradual concessions — never give multiple parameters away in the same round.

OPENING STRATEGY:
- Your first offer MUST use your own reservation defaults for every non-price parameter:
{chr(10).join(f"    * {_PARAM_LABELS[p]}: {_seller_opening_hint(p, prefs)}" for p in self.active_params if p != "price")}
- Only move a non-price parameter after the buyer has moved on price to compensate.
- Do NOT pre-emptively match what the buyer asks for — make them pay for each concession.

{utility_block}
CRITICAL — FORMAT REQUIREMENT:
Every turn you MUST output exactly one offer using this exact format:
  {tag_example}

VALID VALUES:
{valid_block}

DEAL AGREEMENT:
- If you accept the buyer's most recent offer in full, include "MAKE_DEAL" in your response.
- Example: "Agreed, let's proceed. MAKE_DEAL"
- Only use MAKE_DEAL when ALL parameters meet or exceed your reservation values.

Keep your response concise (150 words or less).
Now, respond as {self.name}:
"""
        return guidance

    def _utility_analysis_block(self, current_state: Dict[str, Any]) -> str:
        """Build the UTILITY ANALYSIS section for the guidance block."""
        neg_params = self.context.get("negotiation_params", {})
        p_range  = neg_params.get("price_range",   (18000, 28000))
        d_range  = neg_params.get("delivery_range", (1, 60))
        w_range  = neg_params.get("warranty_range", (12, 48))

        raw_weights   = {p: self.preferences.weight_for(p) for p in self.active_params}
        total_weight  = sum(raw_weights.values())
        if total_weight == 0:
            return ""

        norm_weights  = {p: raw_weights[p] / total_weight for p in self.active_params}
        marginal      = self._compute_marginal_utility()
        u_floor, res_breakdown = self._compute_reservation_utility()

        # --- Weight line ---
        weight_str = ", ".join(
            f"{_PARAM_LABELS[p]}={norm_weights[p]*100:.1f}%"
            for p in self.active_params
        )

        # --- Formula lines (seller perspective) ---
        formula_lines = []
        for p in self.active_params:
            if p == "price":
                formula_lines.append(
                    f"  u_price(P)         = (P − {p_range[0]:,}) / {p_range[1]-p_range[0]:,}"
                    f"   [higher P → higher utility]"
                )
            elif p == "quality":
                formula_lines.append(
                    "  u_quality(Q)       = (2 − QUALITY_ORDER[Q]) / 2"
                    "   [Standard=1.00, Premium=0.50, Luxury=0.00]"
                )
            elif p == "delivery_days":
                formula_lines.append(
                    f"  u_delivery(T)      = (T − {d_range[0]}) / {d_range[1]-d_range[0]}"
                    f"   [more days → higher utility]"
                )
            elif p == "warranty_months":
                formula_lines.append(
                    f"  u_warranty(W)      = ({w_range[1]} − W) / {w_range[1]-w_range[0]}"
                    f"   [fewer months → higher utility]"
                )
            elif p == "payment":
                formula_lines.append(
                    "  u_payment(Pay)     = (2 − PAYMENT_ORDER[Pay]) / 2"
                    "   [upfront=1.00, 30-day=0.50, installments=0.00]"
                )

        # --- Current opponent offer evaluation ---
        opponent_offer = current_state.get("buyer_offer")
        if opponent_offer and any(v is not None for v in opponent_offer.values()):
            u_current, cur_breakdown = self._compute_utility(opponent_offer)

            # Format opponent offer string
            offer_parts = []
            for p in self.active_params:
                v = opponent_offer.get(p, "?")
                if p == "price":
                    offer_parts.append(f"PRICE(${v:,})" if isinstance(v, (int, float)) else f"PRICE({v})")
                elif p == "quality":
                    offer_parts.append(f"QUALITY({v})")
                elif p == "delivery_days":
                    offer_parts.append(f"DELIVERY({v} days)")
                elif p == "warranty_months":
                    offer_parts.append(f"WARRANTY({v} months)")
                elif p == "payment":
                    offer_parts.append(f"PAYMENT({v})")
            offer_str = " | ".join(offer_parts)

            breakdown_str = ", ".join(
                f"{_PARAM_LABELS[p]}={cur_breakdown.get(p, 0):.3f}"
                for p in self.active_params
            )
            weighted_terms = " + ".join(
                f"{norm_weights[p]:.2f}×{cur_breakdown.get(p, 0):.3f}"
                for p in self.active_params
            )
            eval_block = (
                f"  Buyer's last offer: {offer_str}\n"
                f"  Per-param utilities: {breakdown_str}\n"
                f"  U = {weighted_terms} = {u_current:.3f}"
            )
        else:
            u_current = None
            eval_block = "  No offer from buyer yet."

        # --- Reservation floor ---
        res_parts = []
        prefs = self.preferences
        for p in self.active_params:
            if p == "price":
                res_parts.append(f"PRICE(${prefs.price_limit:,.0f})")
            elif p == "quality":
                res_parts.append(f"QUALITY({prefs.min_quality})")
            elif p == "delivery_days":
                res_parts.append(f"DELIVERY({prefs.max_delivery_days} days)")
            elif p == "warranty_months":
                res_parts.append(f"WARRANTY({prefs.min_warranty_months} months)")
            elif p == "payment":
                res_parts.append(f"PAYMENT({prefs.preferred_payment})")
        res_str = " | ".join(res_parts)
        res_breakdown_str = ", ".join(
            f"{_PARAM_LABELS[p]}={res_breakdown.get(p, 0):.3f}"
            for p in self.active_params
        )

        # --- Marginal utility lines ---
        marginal_lines = []
        for p in self.active_params:
            m = marginal.get(p, 0)
            if p == "price":
                marginal_lines.append(f"  Price:          ±{m*1000:.4f} utility per $1,000 higher price")
            elif p == "quality":
                marginal_lines.append(f"  Quality:        ±{m:.4f} utility per tier step lower")
            elif p == "delivery_days":
                marginal_lines.append(f"  Delivery:       ±{m:.4f} utility per 1 extra day")
            elif p == "warranty_months":
                marginal_lines.append(f"  Warranty:       ±{m:.4f} utility per 1 month shorter")
            elif p == "payment":
                marginal_lines.append(f"  Payment:        ±{m:.4f} utility per step toward upfront")

        # Add a cross-param trade-off example (price ↔ quality if both active)
        if "price" in self.active_params and "quality" in self.active_params:
            m_q = marginal.get("quality", 0)
            m_p = marginal.get("price", 0)
            if m_p > 0 and m_q > 0:
                dollars = m_q / m_p
                marginal_lines.append(
                    f"  → Conceding Standard→Premium costs {m_q:.4f} utility"
                    f" — demand ${dollars:,.0f} price increase to stay utility-neutral"
                )
        elif "price" in self.active_params and "delivery_days" in self.active_params:
            m_d = marginal.get("delivery_days", 0)
            m_p = marginal.get("price", 0)
            if m_p > 0 and m_d > 0:
                dollars_per_day = m_d / m_p
                marginal_lines.append(
                    f"  → Each day faster delivery costs {m_d:.4f} utility"
                    f" — demand ${dollars_per_day:,.0f} price increase per day to compensate"
                )

        lines = [
            "UTILITY ANALYSIS:",
            f"  U(offer) = Σ (w_k/Σw) × u_k(x_k)   [0=worst for you, 1=ideal]",
            f"  Normalized weights: {weight_str}",
            "",
            "  Formulas:",
        ] + formula_lines + [
            "",
            "  CURRENT EVALUATION:",
        ] + eval_block.splitlines() + [
            "",
            "  RESERVATION FLOOR (your worst acceptable offer):",
            f"  At {res_str}",
            f"  Per-param: {res_breakdown_str}",
            f"  U_floor = {u_floor:.3f}",
            "",
            "  MARGINAL UTILITY (use to price concessions):",
        ] + marginal_lines + [""]

        return "\n".join(lines) + "\n"

    def _reservation_lines(self) -> List[str]:
        prefs = self.preferences
        lines = []
        if "price" in self.active_params:
            lines.append(f"  * Minimum acceptable price: ${prefs.price_limit:,.2f}")
        if "quality" in self.active_params:
            lines.append(
                f"  * Default quality tier: {prefs.min_quality} (lowest cost to you). "
                f"You CAN offer Premium or Luxury, but only if the buyer agrees to a higher price — "
                f"use quality concessions as a trade-off tool, not a free gift."
            )
        if "delivery_days" in self.active_params:
            lines.append(
                f"  * Standard lead time: {prefs.max_delivery_days} days. "
                f"You CAN deliver faster, but faster delivery requires a higher price to offset rush costs."
            )
        if "warranty_months" in self.active_params:
            lines.append(
                f"  * Standard warranty: {prefs.min_warranty_months} months. "
                f"You CAN offer longer warranty, but it must be paired with a higher price."
            )
        if "payment" in self.active_params:
            lines.append(f"  * Preferred payment: {prefs.preferred_payment}")
        return lines

    def _build_tag_example(self) -> str:
        """Build the format-tag example using seller's own reservation values."""
        prefs = self.preferences
        parts = []
        for p in self.active_params:
            if p == "price":
                parts.append("PRICE($X)")
            elif p == "quality":
                parts.append(f"QUALITY({prefs.min_quality})")
            elif p == "delivery_days":
                parts.append(f"DELIVERY({prefs.max_delivery_days} days)")
            elif p == "warranty_months":
                parts.append(f"WARRANTY({prefs.min_warranty_months} months)")
            elif p == "payment":
                parts.append(f"PAYMENT({prefs.preferred_payment})")
        return f"### SELLER_OFFER: {' | '.join(parts)} ###"
