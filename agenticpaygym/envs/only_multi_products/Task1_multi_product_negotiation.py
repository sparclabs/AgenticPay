"""Task1 Multi-Product Negotiation Environment Implementation

Supports continuous negotiation for multiple products while preserving context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, List

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class ProductNegotiationResult:
    """Single product negotiation result"""
    
    def __init__(
        self,
        product_name: str,
        product_info: Dict[str, Any],
        status: NegotiationStatus,
        agreed_price: Optional[float] = None,
        rounds: int = 0,
    ):
        self.product_name = product_name
        self.product_info = product_info
        self.status = status
        self.agreed_price = agreed_price
        self.rounds = rounds


class Task1MultiProductNegotiation(BaseEnv):
    """Task1 Multi-Product Negotiation Environment
    
    Manages continuous negotiation process for multiple products between buyer and seller.
    Preserves conversation context across different products.
    """
    
    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds_per_product: int = 20,
        initial_seller_price: float = 100.0,
        buyer_max_price: Optional[float] = None,
        seller_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
    ):
        """Initialize multi-product negotiation environment
        
        Args:
            buyer_agent: Buyer Agent
            seller_agent: Seller Agent
            max_rounds_per_product: Maximum number of negotiation rounds per product
            initial_seller_price: Initial price offered by seller (default for new products)
            buyer_max_price: Maximum acceptable price for buyer (confidential)
            seller_min_price: Minimum acceptable price for seller (confidential)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
        """
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.max_rounds_per_product = max_rounds_per_product
        self.initial_seller_price = initial_seller_price
        self.buyer_max_price = buyer_max_price
        self.seller_min_price = seller_min_price
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance
        
        # Call parent class initialization
        super().__init__()
        
        # State management - shared across all products
        self.memory = ConversationMemory()
        self.current_product_state = NegotiationState()
        self.current_round = 0
        self.current_product_info: Optional[Dict[str, Any]] = None
        self.current_product_name: str = ""
        self.negotiation_info = NegotiationInfo()
        
        # Track all product negotiations
        self.product_results: List[ProductNegotiationResult] = []
        self.current_product_index = 0
    
    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        clear_history: bool = False,
        available_products: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new product negotiation
        
        Args:
            user_requirement: User requirement description
            product_info: Product information for current negotiation
            user_profile: User profile
            clear_history: Whether to clear conversation history (default: False, preserves context)
            available_products: List of all available products (seller can see all products)
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Clear history if requested (for completely new session)
        if clear_history:
            self.memory.clear()
            self.product_results = []
            self.current_product_index = 0
        
        # Reset current product state
        self.current_product_state = NegotiationState()
        self.current_round = 0
        self.current_product_info = product_info or {}
        self.current_product_name = product_info.get("name", "Unknown Product") if product_info else "Unknown Product"
        self.negotiation_info = NegotiationInfo()
        
        # Initialize Agents with context (preserve previous context)
        buyer_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "previous_negotiations": self._get_previous_negotiations_summary(),
            "product_info": product_info or {},  # Buyer can now see product information
        }
        self.buyer_agent.initialize(buyer_context)
        
        seller_context = {
            "product_info": product_info or {},
            "available_products": available_products or [],  # All available products
            "initial_price": self.initial_seller_price,
            "min_price": self.seller_min_price,
            "environment_info": self.environment_info,
            "previous_negotiations": self._get_previous_negotiations_summary(),
        }
        self.seller_agent.initialize(seller_context)
        
        # Seller gives initial offer
        initial_message = f"I'm offering this {self.current_product_name} for ${self.initial_seller_price:.2f}."
        self.memory.add_message("seller", initial_message, self.current_round)
        self.current_product_state.update(seller_price=self.initial_seller_price)
        self.negotiation_info.current_price = self.initial_seller_price
        self.negotiation_info.seller_price = self.initial_seller_price
        
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        buyer_action: Optional[str] = None, 
        seller_action: Optional[str] = None
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, both buyer and seller make their responses, then check if agreement is reached.
        
        Args:
            buyer_action: Buyer's response text (optional, can be None if buyer doesn't respond)
            seller_action: Seller's response text (optional, can be None if seller doesn't respond)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Add messages to memory (both agents respond in the same round)
        if seller_action is not None:
            self.memory.add_message("seller", seller_action, self.current_round)
            # Extract seller price
            seller_price = self._extract_price(seller_action)
            if seller_price is not None:
                self.current_product_state.update(seller_price=seller_price)
                self.negotiation_info.seller_price = seller_price
                self.negotiation_info.current_price = seller_price
        
        if buyer_action is not None:
            self.memory.add_message("buyer", buyer_action, self.current_round)
            # Extract buyer price
            buyer_price = self._extract_price(buyer_action)
            if buyer_price is not None:
                self.current_product_state.update(buyer_price=buyer_price)
                self.negotiation_info.buyer_price = buyer_price
                self.negotiation_info.current_price = buyer_price
        
        # Check if agreement is reached (after both agents have responded)
        terminated = False
        truncated = False
        reward = 0.0
        
        if self._check_agreement():
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            agreed_price = (self.current_product_state.buyer_price + self.current_product_state.seller_price) / 2
            self.current_product_state.update(agreed_price=agreed_price)
            self.negotiation_info.current_price = agreed_price
            reward = self._calculate_reward()
            
            # Save product result
            self._save_product_result(agreed_price)
            
        elif self.current_round >= self.max_rounds_per_product:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            reward = self._calculate_reward()
            
            # Save product result
            self._save_product_result(None)
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["product_name"] = self.current_product_name
            info["can_continue"] = True  # Can continue with next product
        
        return observation, reward, terminated, truncated, info
    
    def continue_with_new_product(
        self,
        user_requirement: str,
        product_info: Dict[str, Any],
        user_profile: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Continue negotiation with a new product, preserving context
        
        Args:
            user_requirement: User requirement for new product
            product_info: New product information
            user_profile: User profile
            
        Returns:
            (observation, info) Initial observation and info for new product
        """
        self.current_product_index += 1
        return self.reset(
            user_requirement=user_requirement,
            product_info=product_info,
            user_profile=user_profile,
            clear_history=False,  # Preserve history
        )
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays buyer and seller outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
        # Display current product info
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Product: {self.current_product_name}")
        output_lines.append(f"Product Index: {self.current_product_index + 1}")
        if self.product_results:
            output_lines.append(f"Previous Products Negotiated: {len(self.product_results)}")
        output_lines.append(f"{'='*60}")
        
        # Get messages from current round (both buyer and seller)
        history = self.memory.get_history()
        if history:
            # Get messages from the current round
            current_round_messages = [
                msg for msg in history if msg["round"] == self.current_round
            ]
            
            if current_round_messages:
                output_lines.append(f"\n{'='*60}")
                output_lines.append(f"Round {self.current_round} - Negotiation Output")
                output_lines.append(f"{'='*60}")
                
                # Display seller message first (if exists)
                seller_msg = next(
                    (msg for msg in current_round_messages if msg["role"] == "seller"), 
                    None
                )
                if seller_msg:
                    output_lines.append(f"\n[SELLER Output]:")
                    output_lines.append(f"  {seller_msg['content']}")
                
                # Display buyer message (if exists)
                buyer_msg = next(
                    (msg for msg in current_round_messages if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg:
                    output_lines.append(f"\n[BUYER Output]:")
                    output_lines.append(f"  {buyer_msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display buyer price
        if self.current_product_state.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.current_product_state.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        
        # Display seller price
        if self.current_product_state.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.current_product_state.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Check agreement status and provide reason
        is_agreed = self._check_agreement()
        agreement_reason = self._get_agreement_reason()
        
        output_lines.append(f"  Agreement Status: {'✓ AGREED' if is_agreed else '✗ NOT AGREED'}")
        output_lines.append(f"  Reason: {agreement_reason}")
        
        # Display agreed price if agreement is reached
        if self.current_product_state.agreed_price is not None:
            output_lines.append(f"  Agreed Price: ${self.current_product_state.agreed_price:.2f}")
        
        # Display negotiation status
        status_display = {
            NegotiationStatus.ONGOING: "Ongoing",
            NegotiationStatus.AGREED: "Agreed",
            NegotiationStatus.FAILED: "Failed",
            NegotiationStatus.TIMEOUT: "Timeout"
        }
        output_lines.append(f"  Negotiation Status: {status_display.get(self.negotiation_info.status, 'Unknown')}")
        
        # Display previous products summary
        if self.product_results:
            output_lines.append(f"\n  Previous Products:")
            for i, result in enumerate(self.product_results, 1):
                status_str = "✓ Agreed" if result.status == NegotiationStatus.AGREED else "✗ Timeout"
                price_str = f"${result.agreed_price:.2f}" if result.agreed_price else "N/A"
                output_lines.append(f"    {i}. {result.product_name}: {status_str} @ {price_str}")
        
        output_lines.append(f"{'='*60}\n")
        
        output = "\n".join(output_lines)
        
        if mode == "human":
            print(output)
            return None
        else:
            return output
    
    def _get_previous_negotiations_summary(self) -> str:
        """Get summary of previous negotiations"""
        if not self.product_results:
            return "No previous negotiations."
        
        summary = f"Previous negotiations ({len(self.product_results)} products):\n"
        for i, result in enumerate(self.product_results, 1):
            status_str = "Agreed" if result.status == NegotiationStatus.AGREED else "Timeout"
            price_str = f"${result.agreed_price:.2f}" if result.agreed_price else "N/A"
            summary += f"  {i}. {result.product_name}: {status_str} @ {price_str}\n"
        
        return summary
    
    def _save_product_result(self, agreed_price: Optional[float]):
        """Save current product negotiation result"""
        result = ProductNegotiationResult(
            product_name=self.current_product_name,
            product_info=self.current_product_info or {},
            status=self.negotiation_info.status,
            agreed_price=agreed_price,
            rounds=self.current_round,
        )
        self.product_results.append(result)
    
    def _get_agreement_reason(self) -> str:
        """Get reason for agreement or non-agreement"""
        if self.current_product_state.buyer_price is None or self.current_product_state.seller_price is None:
            if self.current_product_state.buyer_price is None and self.current_product_state.seller_price is None:
                return "Both buyer and seller prices are not specified yet"
            elif self.current_product_state.buyer_price is None:
                return "Buyer price is not specified yet"
            else:
                return "Seller price is not specified yet"
        
        price_diff = abs(self.current_product_state.buyer_price - self.current_product_state.seller_price)
        
        if price_diff <= self.price_tolerance:
            return f"Price difference (${price_diff:.2f}) is within tolerance (${self.price_tolerance:.2f})"
        else:
            return f"Price difference (${price_diff:.2f}) exceeds tolerance (${self.price_tolerance:.2f})"
    
    def close(self):
        """Close environment, cleanup resources"""
        self.memory.clear()
        self.current_product_state = NegotiationState()
        self.product_results = []
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history": self.memory.get_history(),
            "current_round": self.current_round,
            "seller_price": self.current_product_state.seller_price,
            "buyer_price": self.current_product_state.buyer_price,
            "status": self.negotiation_info.status.value,
            "current_product": self.current_product_name,
            "previous_products_count": len(self.product_results),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "seller_price": self.current_product_state.seller_price,
            "buyer_price": self.current_product_state.buyer_price,
            "agreed_price": self.current_product_state.agreed_price,
            "negotiation_info": self.negotiation_info,
            "current_product": self.current_product_name,
            "product_results": [
                {
                    "product_name": r.product_name,
                    "status": r.status.value,
                    "agreed_price": r.agreed_price,
                    "rounds": r.rounds,
                }
                for r in self.product_results
            ],
        }
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text
        
        Args:
            text: Text containing price
            
        Returns:
            Extracted price, returns None if not found
        """
        # Match $XX.XX or $XX format
        patterns = [
            r'\$(\d+\.?\d*)',  # $100.50 or $100
            r'(\d+\.?\d*)\s*dollars?',  # 100.50 dollars
            r'(\d+\.?\d*)\s*USD',  # 100.50 USD
            r'price.*?(\d+\.?\d*)',  # price 100.50
            r'offer.*?(\d+\.?\d*)',  # offer 100.50
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    price = float(matches[-1])  # Take the last match
                    if price > 0:
                        return price
                except ValueError:
                    continue
        
        return None
    
    def _check_agreement(self) -> bool:
        """Check if agreement is reached
        
        When the prices of both buyer and seller are within the tolerance range, an agreement is considered reached.
        
        Returns:
            Whether agreement is reached
        """
        if self.current_product_state.buyer_price is None or self.current_product_state.seller_price is None:
            return False
        
        price_diff = abs(self.current_product_state.buyer_price - self.current_product_state.seller_price)
        return price_diff <= self.price_tolerance
    
    def _calculate_reward(self) -> float:
        """Calculate reward
        
        Calculate reward value based on negotiation result.
        
        If deal is reached:
            reward = buyer savings + seller profit + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            # Deal reached: buyer savings + seller profit + time cost
            if self.current_product_state.agreed_price is None:
                print(f"Reward = time_cost = {time_cost:.2f} (round={self.current_round})")
                return time_cost
            
            deal_price = self.current_product_state.agreed_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings
            
            # Calculate seller profit: deal_price - seller_min_price
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit
            
            # Add time cost (negative penalty)
            reward += time_cost
            
            print(f"Reward = buyer_savings({buyer_savings:.2f}) + seller_profit({seller_profit:.2f}) + time_cost({time_cost:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            print(f"Reward = time_cost = {time_cost:.2f} (round={self.current_round}, deal not reached)")
            return time_cost

