"""Task1 Multi-Seller Negotiation Environment Implementation

Supports parallel negotiation between one buyer and two sellers for the same product.
Buyer can choose to make a deal with the seller offering the lower price.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task1MultiSellerNegotiation(BaseEnv):
    """Task1 Multi-Seller Negotiation Environment
    
    Manages parallel negotiation process between one buyer and two sellers for the same product.
    Buyer negotiates with both sellers simultaneously and can choose to make a deal with the lower price.
    """
    
    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_price: float = 100.0,
        initial_seller2_price: float = 110.0,
        buyer_max_price: Optional[float] = None,
        seller1_min_price: Optional[float] = None,
        seller2_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
    ):
        """Initialize multi-seller negotiation environment
        
        Args:
            buyer_agent: Buyer Agent
            seller1_agent: First Seller Agent
            seller2_agent: Second Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller1_price: Initial price offered by seller1
            initial_seller2_price: Initial price offered by seller2
            buyer_max_price: Maximum acceptable price for buyer (confidential)
            seller1_min_price: Minimum acceptable price for seller1 (confidential)
            seller2_min_price: Minimum acceptable price for seller2 (confidential)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
        """
        self.buyer_agent = buyer_agent
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.max_rounds = max_rounds
        self.initial_seller1_price = initial_seller1_price
        self.initial_seller2_price = initial_seller2_price
        self.buyer_max_price = buyer_max_price
        self.seller1_min_price = seller1_min_price
        self.seller2_min_price = seller2_min_price
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance
        
        # Call parent class initialization
        super().__init__()
        
        # State management - separate for each seller
        self.memory_seller1 = ConversationMemory()
        self.memory_seller2 = ConversationMemory()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        
        # Track which seller was chosen for the deal
        self.selected_seller: Optional[int] = None  # 1 or 2
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
        self.memory_seller1.clear()
        self.memory_seller2.clear()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.selected_seller = None
        self.final_deal_price = None
        
        # Initialize Buyer Agent (buyer knows about both sellers)
        buyer_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": product_info or {},
            "num_sellers": 2,  # Inform buyer there are 2 sellers
        }
        self.buyer_agent.initialize(buyer_context)
        
        # Initialize Seller1 Agent
        seller1_context = {
            "product_info": product_info or {},
            "initial_price": self.initial_seller1_price,
            "min_price": self.seller1_min_price,
            "environment_info": self.environment_info,
            "seller_id": 1,  # Identify as seller 1
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent
        seller2_context = {
            "product_info": product_info or {},
            "initial_price": self.initial_seller2_price,
            "min_price": self.seller2_min_price,
            "environment_info": self.environment_info,
            "seller_id": 2,  # Identify as seller 2
        }
        self.seller2_agent.initialize(seller2_context)
        
        # Sellers give initial offers
        initial_message_seller1 = f"I'm offering this product for ${self.initial_seller1_price:.2f}."
        self.memory_seller1.add_message("seller", initial_message_seller1, self.current_round)
        self.state_seller1.update(seller_price=self.initial_seller1_price)
        
        initial_message_seller2 = f"I'm offering this product for ${self.initial_seller2_price:.2f}."
        self.memory_seller2.add_message("seller", initial_message_seller2, self.current_round)
        self.state_seller2.update(seller_price=self.initial_seller2_price)
        
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        buyer_action_seller1: Optional[str] = None,
        buyer_action_seller2: Optional[str] = None,
        seller1_action: Optional[str] = None,
        seller2_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, buyer responds to both sellers first, then both sellers respond.
        Order: buyer -> seller
        Buyer can choose to make a deal with either seller.
        
        Args:
            buyer_action_seller1: Buyer's response to seller1 (optional)
            buyer_action_seller2: Buyer's response to seller2 (optional)
            seller1_action: Seller1's response (optional)
            seller2_action: Seller2's response (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Add messages to memory in order: buyer -> seller
        # Process buyer actions first (seller1 conversation)
        if buyer_action_seller1 is not None:
            self.memory_seller1.add_message("buyer", buyer_action_seller1, self.current_round)
            buyer_price_seller1 = self._extract_price(buyer_action_seller1)
            if buyer_price_seller1 is not None:
                self.state_seller1.update(buyer_price=buyer_price_seller1)
        
        # Process buyer actions (seller2 conversation)
        if buyer_action_seller2 is not None:
            self.memory_seller2.add_message("buyer", buyer_action_seller2, self.current_round)
            buyer_price_seller2 = self._extract_price(buyer_action_seller2)
            if buyer_price_seller2 is not None:
                self.state_seller2.update(buyer_price=buyer_price_seller2)
        
        # Process seller actions after buyer (seller1 conversation)
        if seller1_action is not None:
            self.memory_seller1.add_message("seller", seller1_action, self.current_round)
            seller1_price = self._extract_price(seller1_action)
            if seller1_price is not None:
                self.state_seller1.update(seller_price=seller1_price)
        
        # Process seller actions after buyer (seller2 conversation)
        if seller2_action is not None:
            self.memory_seller2.add_message("seller", seller2_action, self.current_round)
            seller2_price = self._extract_price(seller2_action)
            if seller2_price is not None:
                self.state_seller2.update(seller_price=seller2_price)
        
        # After processing all actions, check if buyer wants to make a deal
        # Compare both sellers' prices and choose the one with lower price
        make_deal_seller1 = buyer_action_seller1 is not None and self._check_make_deal(buyer_action_seller1)
        make_deal_seller2 = buyer_action_seller2 is not None and self._check_make_deal(buyer_action_seller2)
        
        if make_deal_seller1 or make_deal_seller2:
            # Buyer wants to make a deal, choose the seller with lower price
            price1 = self._get_effective_price_seller1()
            price2 = self._get_effective_price_seller2()
            
            if price1 is not None and price2 is not None:
                # Both prices available, choose the lower one
                if price1 <= price2:
                    self.selected_seller = 1
                    if self.state_seller1.buyer_price is not None and self.state_seller1.seller_price is not None:
                        self.final_deal_price = (self.state_seller1.buyer_price + self.state_seller1.seller_price) / 2
                    else:
                        self.final_deal_price = price1
                else:
                    self.selected_seller = 2
                    if self.state_seller2.buyer_price is not None and self.state_seller2.seller_price is not None:
                        self.final_deal_price = (self.state_seller2.buyer_price + self.state_seller2.seller_price) / 2
                    else:
                        self.final_deal_price = price2
            elif price1 is not None:
                # Only seller1 price available
                self.selected_seller = 1
                if self.state_seller1.buyer_price is not None and self.state_seller1.seller_price is not None:
                    self.final_deal_price = (self.state_seller1.buyer_price + self.state_seller1.seller_price) / 2
                else:
                    self.final_deal_price = price1
            elif price2 is not None:
                # Only seller2 price available
                self.selected_seller = 2
                if self.state_seller2.buyer_price is not None and self.state_seller2.seller_price is not None:
                    self.final_deal_price = (self.state_seller2.buyer_price + self.state_seller2.seller_price) / 2
                else:
                    self.final_deal_price = price2
        
        # Check if deal is made (buyer chose a seller)
        terminated = False
        truncated = False
        reward = 0.0
        
        if self.selected_seller is not None and self.final_deal_price is not None:
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
                info["selected_seller"] = self.selected_seller
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
        
        # Display Seller1 conversation
        output_lines.append(f"\n[SELLER 1 Conversation]:")
        history_seller1 = self.memory_seller1.get_history()
        if history_seller1:
            current_round_messages_s1 = [
                msg for msg in history_seller1 if msg["round"] == self.current_round
            ]
            for msg in current_round_messages_s1:
                role = msg["role"].upper()
                output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Seller2 conversation
        output_lines.append(f"\n[SELLER 2 Conversation]:")
        history_seller2 = self.memory_seller2.get_history()
        if history_seller2:
            current_round_messages_s2 = [
                msg for msg in history_seller2 if msg["round"] == self.current_round
            ]
            for msg in current_round_messages_s2:
                role = msg["role"].upper()
                output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Seller1 prices
        output_lines.append(f"\nSeller 1:")
        if self.state_seller1.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_seller1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_seller1.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_seller1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display Seller2 prices
        output_lines.append(f"\nSeller 2:")
        if self.state_seller2.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_seller2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_seller2.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_seller2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display deal status
        if self.selected_seller is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Seller {self.selected_seller}")
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
        self.memory_seller1.clear()
        self.memory_seller2.clear()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history_seller1": self.memory_seller1.get_history(),
            "conversation_history_seller2": self.memory_seller2.get_history(),
            "current_round": self.current_round,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "status": self.negotiation_info.status.value,
            "selected_seller": self.selected_seller,
            "final_deal_price": self.final_deal_price,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "selected_seller": self.selected_seller,
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
    
    def _get_effective_price_seller1(self) -> Optional[float]:
        """Get effective price for seller1 (agreed price if available, otherwise seller price)
        
        Returns:
            Effective price for seller1
        """
        if self.state_seller1.buyer_price is not None and self.state_seller1.seller_price is not None:
            price_diff = abs(self.state_seller1.buyer_price - self.state_seller1.seller_price)
            if price_diff <= self.price_tolerance:
                return (self.state_seller1.buyer_price + self.state_seller1.seller_price) / 2
        return self.state_seller1.seller_price
    
    def _get_effective_price_seller2(self) -> Optional[float]:
        """Get effective price for seller2 (agreed price if available, otherwise seller price)
        
        Returns:
            Effective price for seller2
        """
        if self.state_seller2.buyer_price is not None and self.state_seller2.seller_price is not None:
            price_diff = abs(self.state_seller2.buyer_price - self.state_seller2.seller_price)
            if price_diff <= self.price_tolerance:
                return (self.state_seller2.buyer_price + self.state_seller2.seller_price) / 2
        return self.state_seller2.seller_price
    
    def _calculate_reward(self) -> float:
        """Calculate reward
        
        Calculate reward value based on negotiation result.
        If deal is reached with a seller, use that seller's min_price for calculation.
        Reward is based on the lower price deal if buyer made deals with both sellers.
        
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
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected seller's min_price
            selected_seller_min_price = None
            if self.selected_seller == 1:
                selected_seller_min_price = self.seller1_min_price
            elif self.selected_seller == 2:
                selected_seller_min_price = self.seller2_min_price
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings
            
            # Calculate seller profit: deal_price - seller_min_price
            if selected_seller_min_price is not None:
                seller_profit = deal_price - selected_seller_min_price
                reward += seller_profit
            
            # Add time cost (negative penalty)
            reward += time_cost
            
            print(f"Reward = buyer_savings({buyer_savings:.2f}) + seller{self.selected_seller}_profit({seller_profit:.2f}) + time_cost({time_cost:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, seller{self.selected_seller}_min={selected_seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            print(f"Reward = time_cost = {time_cost:.2f} (round={self.current_round}, deal not reached)")
            return time_cost
