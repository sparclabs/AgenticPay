"""Task2 Parallel Three-Buyer Negotiation Environment Implementation

Supports parallel negotiation between three buyers and one seller for the same product.
Seller can choose to make a deal with the buyer offering the higher price.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task2ParallelThreeBuyerNegotiation(BaseEnv):
    """Task2 Parallel Three-Buyer Negotiation Environment
    
    Manages parallel negotiation process between three buyers and one seller for the same product.
    Seller negotiates with all three buyers simultaneously and automatically chooses the buyer with the higher price.
    """
    
    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        buyer3_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_price: float = 100.0,
        buyer1_max_price: Optional[float] = None,
        buyer2_max_price: Optional[float] = None,
        buyer3_max_price: Optional[float] = None,
        seller_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
    ):
        """Initialize multi-buyer negotiation environment
        
        Args:
            buyer1_agent: First Buyer Agent
            buyer2_agent: Second Buyer Agent
            buyer3_agent: Third Buyer Agent
            seller_agent: Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller_price: Initial price offered by seller
            buyer1_max_price: Maximum acceptable price for buyer1 (confidential)
            buyer2_max_price: Maximum acceptable price for buyer2 (confidential)
            buyer3_max_price: Maximum acceptable price for buyer3 (confidential)
            seller_min_price: Minimum acceptable price for seller (confidential)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
        """
        self.buyer1_agent = buyer1_agent
        self.buyer2_agent = buyer2_agent
        self.buyer3_agent = buyer3_agent
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_price = initial_seller_price
        self.buyer1_max_price = buyer1_max_price
        self.buyer2_max_price = buyer2_max_price
        self.buyer3_max_price = buyer3_max_price
        self.seller_min_price = seller_min_price
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance
        
        # Call parent class initialization
        super().__init__()
        
        # State management - separate for each buyer
        self.memory_buyer1 = ConversationMemory()
        self.memory_buyer2 = ConversationMemory()
        self.memory_buyer3 = ConversationMemory()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.state_buyer3 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        
        # Track which buyer was chosen for the deal
        self.selected_buyer: Optional[int] = None  # 1, 2, or 3
        self.final_deal_price: Optional[float] = None
    
    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new negotiation
        
        Args:
            user_requirement: User requirement description
            product_info: Product information
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.memory_buyer3.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.state_buyer3 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.selected_buyer = None
        self.final_deal_price = None
        
        # Initialize Buyer1 Agent
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info or {},
            "buyer_id": 1,  # Identify as buyer 1
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info or {},
            "buyer_id": 2,  # Identify as buyer 2
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Buyer3 Agent
        buyer3_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer3_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info or {},
            "buyer_id": 3,  # Identify as buyer 3
        }
        self.buyer3_agent.initialize(buyer3_context)
        
        # Initialize Seller Agent (seller knows about all three buyers)
        seller_context = {
            "product_info": product_info or {},
            "initial_price": self.initial_seller_price,
            "min_price": self.seller_min_price,
            "environment_info": self.environment_info,
            "num_buyers": 3,  # Inform seller there are 3 buyers
        }
        self.seller_agent.initialize(seller_context)
        
        # Seller gives initial offers to all buyers
        initial_message = f"I'm offering this product for ${self.initial_seller_price:.2f}."
        self.memory_buyer1.add_message("seller", initial_message, self.current_round)
        self.state_buyer1.update(seller_price=self.initial_seller_price)
        
        self.memory_buyer2.add_message("seller", initial_message, self.current_round)
        self.state_buyer2.update(seller_price=self.initial_seller_price)
        
        self.memory_buyer3.add_message("seller", initial_message, self.current_round)
        self.state_buyer3.update(seller_price=self.initial_seller_price)
        
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        buyer1_action: Optional[str] = None,
        buyer2_action: Optional[str] = None,
        buyer3_action: Optional[str] = None,
        seller_action_buyer1: Optional[str] = None,
        seller_action_buyer2: Optional[str] = None,
        seller_action_buyer3: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, all three buyers respond first, then seller responds to all three buyers.
        Order: buyer -> seller
        Seller automatically chooses the buyer with the higher price when multiple buyers make deals.
        
        Args:
            buyer1_action: Buyer1's response (optional)
            buyer2_action: Buyer2's response (optional)
            buyer3_action: Buyer3's response (optional)
            seller_action_buyer1: Seller's response to buyer1 (optional)
            seller_action_buyer2: Seller's response to buyer2 (optional)
            seller_action_buyer3: Seller's response to buyer3 (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Add messages to memory in order: buyer -> seller
        # Process buyer1 actions first
        if buyer1_action is not None:
            self.memory_buyer1.add_message("buyer", buyer1_action, self.current_round)
            buyer1_price = self._extract_price(buyer1_action)
            if buyer1_price is not None:
                self.state_buyer1.update(buyer_price=buyer1_price)
        
        # Process buyer2 actions
        if buyer2_action is not None:
            self.memory_buyer2.add_message("buyer", buyer2_action, self.current_round)
            buyer2_price = self._extract_price(buyer2_action)
            if buyer2_price is not None:
                self.state_buyer2.update(buyer_price=buyer2_price)
        
        # Process buyer3 actions
        if buyer3_action is not None:
            self.memory_buyer3.add_message("buyer", buyer3_action, self.current_round)
            buyer3_price = self._extract_price(buyer3_action)
            if buyer3_price is not None:
                self.state_buyer3.update(buyer_price=buyer3_price)
        
        # Process seller actions after buyers (buyer1 conversation)
        if seller_action_buyer1 is not None:
            self.memory_buyer1.add_message("seller", seller_action_buyer1, self.current_round)
            seller_price_buyer1 = self._extract_price(seller_action_buyer1)
            if seller_price_buyer1 is not None:
                self.state_buyer1.update(seller_price=seller_price_buyer1)
        
        # Process seller actions after buyers (buyer2 conversation)
        if seller_action_buyer2 is not None:
            self.memory_buyer2.add_message("seller", seller_action_buyer2, self.current_round)
            seller_price_buyer2 = self._extract_price(seller_action_buyer2)
            if seller_price_buyer2 is not None:
                self.state_buyer2.update(seller_price=seller_price_buyer2)
        
        # Process seller actions after buyers (buyer3 conversation)
        if seller_action_buyer3 is not None:
            self.memory_buyer3.add_message("seller", seller_action_buyer3, self.current_round)
            seller_price_buyer3 = self._extract_price(seller_action_buyer3)
            if seller_price_buyer3 is not None:
                self.state_buyer3.update(seller_price=seller_price_buyer3)
        
        # After processing all actions, check if buyers want to make deals
        # Buyer must explicitly express make deal intent AND price_tolerance condition must be satisfied
        # Check if buyer1 can make deal
        can_make_deal_buyer1 = False
        if (buyer1_action is not None and 
            self._check_make_deal(buyer1_action) and
            self.state_buyer1.buyer_price is not None and 
            self.state_buyer1.seller_price is not None):
            price_diff = abs(self.state_buyer1.buyer_price - self.state_buyer1.seller_price)
            if price_diff <= self.price_tolerance:
                can_make_deal_buyer1 = True
        
        # Check if buyer2 can make deal
        can_make_deal_buyer2 = False
        if (buyer2_action is not None and 
            self._check_make_deal(buyer2_action) and
            self.state_buyer2.buyer_price is not None and 
            self.state_buyer2.seller_price is not None):
            price_diff = abs(self.state_buyer2.buyer_price - self.state_buyer2.seller_price)
            if price_diff <= self.price_tolerance:
                can_make_deal_buyer2 = True
        
        # Check if buyer3 can make deal
        can_make_deal_buyer3 = False
        if (buyer3_action is not None and 
            self._check_make_deal(buyer3_action) and
            self.state_buyer3.buyer_price is not None and 
            self.state_buyer3.seller_price is not None):
            price_diff = abs(self.state_buyer3.buyer_price - self.state_buyer3.seller_price)
            if price_diff <= self.price_tolerance:
                can_make_deal_buyer3 = True
        
        if can_make_deal_buyer1 or can_make_deal_buyer2 or can_make_deal_buyer3:
            # Multiple buyers want to make deals and price tolerance is satisfied, choose the buyer with higher price
            price1 = self._get_effective_price_buyer1() if can_make_deal_buyer1 else None
            price2 = self._get_effective_price_buyer2() if can_make_deal_buyer2 else None
            price3 = self._get_effective_price_buyer3() if can_make_deal_buyer3 else None
            
            # Find the buyer with the highest price
            candidates = []
            if price1 is not None:
                candidates.append((1, price1))
            if price2 is not None:
                candidates.append((2, price2))
            if price3 is not None:
                candidates.append((3, price3))
            
            if candidates:
                # Sort by price (descending) and select the highest
                candidates.sort(key=lambda x: x[1], reverse=True)
                selected_buyer_id, _ = candidates[0]
                self.selected_buyer = selected_buyer_id
                
                # Set final deal price based on selected buyer
                if selected_buyer_id == 1:
                    self.final_deal_price = (self.state_buyer1.buyer_price + self.state_buyer1.seller_price) / 2
                elif selected_buyer_id == 2:
                    self.final_deal_price = (self.state_buyer2.buyer_price + self.state_buyer2.seller_price) / 2
                elif selected_buyer_id == 3:
                    self.final_deal_price = (self.state_buyer3.buyer_price + self.state_buyer3.seller_price) / 2
        
        # Check if deal is made (seller chose a buyer)
        terminated = False
        truncated = False
        reward = 0.0
        
        if self.selected_buyer is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            reward = self._calculate_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            reward = self._calculate_reward()
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.selected_buyer
                info["final_deal_price"] = self.final_deal_price
        
        return observation, reward, terminated, truncated, info
    
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
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Round {self.current_round} - Parallel Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display Buyer1 conversation
        output_lines.append(f"\n[BUYER 1 Conversation]:")
        history_buyer1 = self.memory_buyer1.get_history()
        if history_buyer1:
            current_round_messages_b1 = [
                msg for msg in history_buyer1 if msg["round"] == self.current_round
            ]
            for msg in current_round_messages_b1:
                role = msg["role"].upper()
                output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer2 conversation
        output_lines.append(f"\n[BUYER 2 Conversation]:")
        history_buyer2 = self.memory_buyer2.get_history()
        if history_buyer2:
            current_round_messages_b2 = [
                msg for msg in history_buyer2 if msg["round"] == self.current_round
            ]
            for msg in current_round_messages_b2:
                role = msg["role"].upper()
                output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer3 conversation
        output_lines.append(f"\n[BUYER 3 Conversation]:")
        history_buyer3 = self.memory_buyer3.get_history()
        if history_buyer3:
            current_round_messages_b3 = [
                msg for msg in history_buyer3 if msg["round"] == self.current_round
            ]
            for msg in current_round_messages_b3:
                role = msg["role"].upper()
                output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Buyer1 prices
        output_lines.append(f"\nBuyer 1:")
        if self.state_buyer1.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_buyer1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_buyer1.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_buyer1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display Buyer2 prices
        output_lines.append(f"\nBuyer 2:")
        if self.state_buyer2.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_buyer2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_buyer2.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_buyer2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display Buyer3 prices
        output_lines.append(f"\nBuyer 3:")
        if self.state_buyer3.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_buyer3.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_buyer3.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_buyer3.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display deal status
        if self.selected_buyer is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Buyer {self.selected_buyer}")
            if self.final_deal_price is not None:
                output_lines.append(f"  Final Deal Price: ${self.final_deal_price:.2f}")
        else:
            output_lines.append(f"\n  ✗ NO DEAL YET")
        
        # Display negotiation status
        status_display = {
            NegotiationStatus.ONGOING: "Ongoing",
            NegotiationStatus.AGREED: "Agreed",
            NegotiationStatus.FAILED: "Failed",
            NegotiationStatus.TIMEOUT: "Timeout"
        }
        output_lines.append(f"  Negotiation Status: {status_display.get(self.negotiation_info.status, 'Unknown')}")
        
        output_lines.append(f"{'='*60}\n")
        
        output = "\n".join(output_lines)
        
        if mode == "human":
            print(output)
            return None
        else:
            return output
    
    def close(self):
        """Close environment, cleanup resources"""
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.memory_buyer3.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.state_buyer3 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history_buyer1": self.memory_buyer1.get_history(),
            "conversation_history_buyer2": self.memory_buyer2.get_history(),
            "conversation_history_buyer3": self.memory_buyer3.get_history(),
            "current_round": self.current_round,
            "buyer1_price": self.state_buyer1.buyer_price,
            "seller_price_buyer1": self.state_buyer1.seller_price,
            "buyer2_price": self.state_buyer2.buyer_price,
            "seller_price_buyer2": self.state_buyer2.seller_price,
            "buyer3_price": self.state_buyer3.buyer_price,
            "seller_price_buyer3": self.state_buyer3.seller_price,
            "status": self.negotiation_info.status.value,
            "selected_buyer": self.selected_buyer,
            "final_deal_price": self.final_deal_price,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "buyer1_price": self.state_buyer1.buyer_price,
            "seller_price_buyer1": self.state_buyer1.seller_price,
            "buyer2_price": self.state_buyer2.buyer_price,
            "seller_price_buyer2": self.state_buyer2.seller_price,
            "buyer3_price": self.state_buyer3.buyer_price,
            "seller_price_buyer3": self.state_buyer3.seller_price,
            "selected_buyer": self.selected_buyer,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
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
    
    def _check_make_deal(self, text: str) -> bool:
        """Check if buyer wants to make a deal
        
        Args:
            text: Buyer's response text
            
        Returns:
            Whether buyer wants to make a deal
        """
        # First check for the fixed format "MAKE_DEAL"
        if 'MAKE_DEAL' in text.upper():
            return True
        
        # Also check for other make deal patterns
        make_deal_patterns = [
            r'make\s+deal',
            r'accept',
            r'agree',
            r'deal',
            r'let\'?s\s+do\s+it',
            r'i\'?ll\s+take\s+it',
            r'we\s+have\s+a\s+deal',
        ]
        
        text_lower = text.lower()
        for pattern in make_deal_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _get_effective_price_buyer1(self) -> Optional[float]:
        """Get effective price for buyer1 (agreed price if available, otherwise buyer price)
        
        Returns:
            Effective price for buyer1
        """
        if self.state_buyer1.buyer_price is not None and self.state_buyer1.seller_price is not None:
            price_diff = abs(self.state_buyer1.buyer_price - self.state_buyer1.seller_price)
            if price_diff <= self.price_tolerance:
                return (self.state_buyer1.buyer_price + self.state_buyer1.seller_price) / 2
        return self.state_buyer1.buyer_price
    
    def _get_effective_price_buyer2(self) -> Optional[float]:
        """Get effective price for buyer2 (agreed price if available, otherwise buyer price)
        
        Returns:
            Effective price for buyer2
        """
        if self.state_buyer2.buyer_price is not None and self.state_buyer2.seller_price is not None:
            price_diff = abs(self.state_buyer2.buyer_price - self.state_buyer2.seller_price)
            if price_diff <= self.price_tolerance:
                return (self.state_buyer2.buyer_price + self.state_buyer2.seller_price) / 2
        return self.state_buyer2.buyer_price
    
    def _get_effective_price_buyer3(self) -> Optional[float]:
        """Get effective price for buyer3 (agreed price if available, otherwise buyer price)
        
        Returns:
            Effective price for buyer3
        """
        if self.state_buyer3.buyer_price is not None and self.state_buyer3.seller_price is not None:
            price_diff = abs(self.state_buyer3.buyer_price - self.state_buyer3.seller_price)
            if price_diff <= self.price_tolerance:
                return (self.state_buyer3.buyer_price + self.state_buyer3.seller_price) / 2
        return self.state_buyer3.buyer_price
    
    def _calculate_reward(self) -> float:
        """Calculate reward
        
        Calculate reward value based on negotiation result.
        If deal is reached with a buyer, use that buyer's max_price for calculation.
        Reward is based on the higher price deal if seller made deals with multiple buyers.
        
        If deal is reached:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Note: Since seller wants to maximize profit, we focus on seller profit.
        The reward is primarily based on seller profit from the higher price deal.
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.selected_buyer is not None and self.final_deal_price is not None:
            # Deal reached: seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            # Calculate seller profit: deal_price - seller_min_price
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit
            
            # Add time cost (negative penalty)
            reward += time_cost
            
            # Get the selected buyer's max_price for display
            selected_buyer_max_price = None
            if self.selected_buyer == 1:
                selected_buyer_max_price = self.buyer1_max_price
            elif self.selected_buyer == 2:
                selected_buyer_max_price = self.buyer2_max_price
            elif self.selected_buyer == 3:
                selected_buyer_max_price = self.buyer3_max_price
            
            print(f"Reward = seller_profit({seller_profit:.2f}) + time_cost({time_cost:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, buyer{self.selected_buyer}_max={selected_buyer_max_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            print(f"Reward = time_cost = {time_cost:.2f} (round={self.current_round}, deal not reached)")
            return time_cost

