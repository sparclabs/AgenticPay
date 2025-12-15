"""Task4 Sequential Three-Buyer Negotiation Environment Implementation

Supports sequential negotiation where seller chooses one buyer per round to negotiate with.
Seller can switch between three buyers and make a deal with any buyer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task4SequentialThreeBuyerNegotiation(BaseEnv):
    """Task4 Sequential Three-Buyer Negotiation Environment
    
    Manages sequential negotiation process where seller chooses one buyer per round to negotiate with.
    Seller can switch between three buyers and make a deal with any buyer.
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
        """Initialize sequential multi-buyer negotiation environment
        
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
        
        # Track which buyer is currently selected and which buyer was chosen for the deal
        self.current_selected_buyer: Optional[int] = None  # 1, 2, or 3, selected for current round
        self.final_selected_buyer: Optional[int] = None  # 1, 2, or 3, chosen for final deal
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
        self.current_selected_buyer = None
        self.final_selected_buyer = None
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
            "negotiation_mode": "sequential",  # Inform seller this is sequential negotiation
        }
        self.seller_agent.initialize(seller_context)
        
        # Seller gives initial offers to all three buyers
        initial_message_buyer1 = f"I'm offering this product for ${self.initial_seller_price:.2f}."
        self.memory_buyer1.add_message("seller", initial_message_buyer1, self.current_round)
        self.state_buyer1.update(seller_price=self.initial_seller_price)
        
        initial_message_buyer2 = f"I'm offering this product for ${self.initial_seller_price:.2f}."
        self.memory_buyer2.add_message("seller", initial_message_buyer2, self.current_round)
        self.state_buyer2.update(seller_price=self.initial_seller_price)
        
        initial_message_buyer3 = f"I'm offering this product for ${self.initial_seller_price:.2f}."
        self.memory_buyer3.add_message("seller", initial_message_buyer3, self.current_round)
        self.state_buyer3.update(seller_price=self.initial_seller_price)
        
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        selected_buyer: int,  # 1, 2, or 3, which buyer seller chooses to negotiate with this round
        seller_action: Optional[str] = None,
        buyer_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, seller chooses one buyer to negotiate with, then seller and that buyer exchange messages.
        Order: seller -> buyer
        
        Args:
            selected_buyer: Which buyer (1, 2, or 3) seller chooses to negotiate with this round
            seller_action: Seller's response (optional)
            buyer_action: Selected buyer's response (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if selected_buyer not in [1, 2, 3]:
            raise ValueError(f"selected_buyer must be 1, 2, or 3, got {selected_buyer}")
        
        self.current_selected_buyer = selected_buyer
        
        # Add messages to memory in order: seller -> buyer
        # Process seller action first
        if seller_action is not None:
            if selected_buyer == 1:
                self.memory_buyer1.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer1.update(seller_price=seller_price)
            elif selected_buyer == 2:
                self.memory_buyer2.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer2.update(seller_price=seller_price)
            else:  # selected_buyer == 3
                self.memory_buyer3.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer3.update(seller_price=seller_price)
        
        # Process buyer action after seller
        if buyer_action is not None:
            if selected_buyer == 1:
                self.memory_buyer1.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer1.update(buyer_price=buyer_price)
            elif selected_buyer == 2:
                self.memory_buyer2.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer2.update(buyer_price=buyer_price)
            else:  # selected_buyer == 3
                self.memory_buyer3.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer3.update(buyer_price=buyer_price)
        
        # Check if deal can be made with the selected buyer
        # Buyer must explicitly express make deal intent AND price_tolerance condition must be satisfied
        if selected_buyer == 1:
            if (buyer_action is not None and 
                self._check_make_deal(buyer_action) and
                self.state_buyer1.buyer_price is not None and 
                self.state_buyer1.seller_price is not None):
                price_diff = abs(self.state_buyer1.buyer_price - self.state_buyer1.seller_price)
                # Only make deal if buyer wants to make deal AND prices are within tolerance
                if price_diff <= self.price_tolerance:
                    self.final_selected_buyer = 1
                    self.final_deal_price = (self.state_buyer1.buyer_price + self.state_buyer1.seller_price) / 2
        elif selected_buyer == 2:
            if (buyer_action is not None and 
                self._check_make_deal(buyer_action) and
                self.state_buyer2.buyer_price is not None and 
                self.state_buyer2.seller_price is not None):
                price_diff = abs(self.state_buyer2.buyer_price - self.state_buyer2.seller_price)
                # Only make deal if buyer wants to make deal AND prices are within tolerance
                if price_diff <= self.price_tolerance:
                    self.final_selected_buyer = 2
                    self.final_deal_price = (self.state_buyer2.buyer_price + self.state_buyer2.seller_price) / 2
        else:  # selected_buyer == 3
            if (buyer_action is not None and 
                self._check_make_deal(buyer_action) and
                self.state_buyer3.buyer_price is not None and 
                self.state_buyer3.seller_price is not None):
                price_diff = abs(self.state_buyer3.buyer_price - self.state_buyer3.seller_price)
                # Only make deal if buyer wants to make deal AND prices are within tolerance
                if price_diff <= self.price_tolerance:
                    self.final_selected_buyer = 3
                    self.final_deal_price = (self.state_buyer3.buyer_price + self.state_buyer3.seller_price) / 2
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        
        if self.final_selected_buyer is not None and self.final_deal_price is not None:
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
                info["selected_buyer"] = self.final_selected_buyer
                info["final_deal_price"] = self.final_deal_price
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays seller and buyer outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Round {self.current_round} - Sequential Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display which buyer was selected this round
        if self.current_selected_buyer is not None:
            output_lines.append(f"\n[Selected Buyer: Buyer {self.current_selected_buyer}]")
        
        # Display Buyer1 conversation (if this round negotiated with buyer1)
        if self.current_selected_buyer == 1:
            output_lines.append(f"\n[BUYER 1 Conversation]:")
            history_buyer1 = self.memory_buyer1.get_history()
            if history_buyer1:
                current_round_messages_b1 = [
                    msg for msg in history_buyer1 if msg["round"] == self.current_round
                ]
                for msg in current_round_messages_b1:
                    role = msg["role"].upper()
                    output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer2 conversation (if this round negotiated with buyer2)
        if self.current_selected_buyer == 2:
            output_lines.append(f"\n[BUYER 2 Conversation]:")
            history_buyer2 = self.memory_buyer2.get_history()
            if history_buyer2:
                current_round_messages_b2 = [
                    msg for msg in history_buyer2 if msg["round"] == self.current_round
                ]
                for msg in current_round_messages_b2:
                    role = msg["role"].upper()
                    output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer3 conversation (if this round negotiated with buyer3)
        if self.current_selected_buyer == 3:
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
        if self.final_selected_buyer is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Buyer {self.final_selected_buyer}")
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
            "current_selected_buyer": self.current_selected_buyer,
            "buyer1_price": self.state_buyer1.buyer_price,
            "seller_price_buyer1": self.state_buyer1.seller_price,
            "buyer2_price": self.state_buyer2.buyer_price,
            "seller_price_buyer2": self.state_buyer2.seller_price,
            "buyer3_price": self.state_buyer3.buyer_price,
            "seller_price_buyer3": self.state_buyer3.seller_price,
            "status": self.negotiation_info.status.value,
            "final_selected_buyer": self.final_selected_buyer,
            "final_deal_price": self.final_deal_price,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "current_selected_buyer": self.current_selected_buyer,
            "buyer1_price": self.state_buyer1.buyer_price,
            "seller_price_buyer1": self.state_buyer1.seller_price,
            "buyer2_price": self.state_buyer2.buyer_price,
            "seller_price_buyer2": self.state_buyer2.seller_price,
            "buyer3_price": self.state_buyer3.buyer_price,
            "seller_price_buyer3": self.state_buyer3.seller_price,
            "final_selected_buyer": self.final_selected_buyer,
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
        
        # More specific patterns to avoid false positives
        # Exclude phrases like "I hope we can reach an agreement" which are just expressions of hope
        make_deal_patterns = [
            r'\bi\s+accept\b',  # "I accept"
            r'\bi\s+agree\b',  # "I agree"
            r'\baccept\s+your\s+offer\b',  # "accept your offer"
            r'\bagree\s+to\s+the\s+deal\b',  # "agree to the deal"
            r'\bmake\s+a\s+deal\b',  # "make a deal"
            r'\bwe\s+have\s+a\s+deal\b',  # "we have a deal"
            r'\blet\'?s\s+do\s+it\b',  # "let's do it"
            r'\bi\'?ll\s+take\s+it\b',  # "I'll take it"
            r'\bi\'?m\s+accepting\b',  # "I'm accepting"
            r'\bi\'?m\s+agreeing\b',  # "I'm agreeing"
            r'\bdeal\s*!',  # "deal!"
            r'\bi\s+accept\s+the\s+price\b',  # "I accept the price"
        ]
        
        text_lower = text.lower()
        
        # Exclude common false positive patterns
        false_positive_patterns = [
            r'hope\s+.*\s+agreement',  # "hope ... agreement"
            r'hoping\s+.*\s+agreement',  # "hoping ... agreement"
            r'would\s+like\s+.*\s+agreement',  # "would like ... agreement"
            r'looking\s+forward\s+.*\s+agreement',  # "looking forward ... agreement"
        ]
        
        # Check for false positives first
        for pattern in false_positive_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for actual deal patterns
        for pattern in make_deal_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _calculate_reward(self) -> float:
        """Calculate reward
        
        Calculate reward value based on negotiation result.
        If deal is reached with a buyer, use that buyer's max_price for calculation.
        
        If deal is reached:
            reward = seller profit + time cost (negative, based on rounds)
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
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_deal_price is not None:
            # Deal reached: seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            # Get the selected buyer's max_price
            selected_buyer_max_price = None
            if self.final_selected_buyer == 1:
                selected_buyer_max_price = self.buyer1_max_price
            elif self.final_selected_buyer == 2:
                selected_buyer_max_price = self.buyer2_max_price
            elif self.final_selected_buyer == 3:
                selected_buyer_max_price = self.buyer3_max_price
            
            # Calculate seller profit: deal_price - seller_min_price
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit
            
            # Add time cost (negative penalty)
            reward += time_cost
            
            print(f"Reward = seller_profit({seller_profit:.2f}) + time_cost({time_cost:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, buyer{self.final_selected_buyer}_max={selected_buyer_max_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            print(f"Reward = time_cost = {time_cost:.2f} (round={self.current_round}, deal not reached)")
            return time_cost

