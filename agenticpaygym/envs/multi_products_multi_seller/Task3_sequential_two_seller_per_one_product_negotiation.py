"""Task3 Sequential Two-Seller Per One Product Negotiation Environment Implementation

Supports sequential negotiation where buyer chooses one seller per round to negotiate with.
Each seller has their own unique product. Buyer can switch between two sellers and make a deal with either seller.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task3SequentialTwoSellerPerOneProductNegotiation(BaseEnv):
    """Task3 Sequential Two-Seller Per One Product Negotiation Environment
    
    Manages sequential negotiation process where buyer chooses one seller per round to negotiate with.
    Each seller has their own unique product. Buyer can switch between two sellers and make a deal with either seller.
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
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize sequential multi-seller negotiation environment
        
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
            reward_weights: Reward weights configuration dict with keys:
                - buyer_savings: weight for buyer savings (default: 1.0)
                - seller_profit: weight for seller profit (default: 1.0)
                - time_cost: weight for time cost (default: 0.1)
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
        
        # Set default reward weights
        default_weights = {
            "buyer_savings": 1.0,      # Buyer savings weight
            "seller_profit": 1.0,      # Seller profit weight
            "time_cost": 0.1,          # Time cost weight (reduced impact)
        }
        if reward_weights is not None:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights
        
        # Call parent class initialization
        super().__init__()
        
        # State management - separate for each seller
        self.memory_seller1 = ConversationMemory()
        self.memory_seller2 = ConversationMemory()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        
        # Track which seller is currently selected and which seller was chosen for the deal
        self.current_selected_seller: Optional[int] = None  # 1 or 2, selected for current round
        self.final_selected_seller: Optional[int] = None  # 1 or 2, chosen for final deal
        self.final_deal_price: Optional[float] = None
        
        # Store product info for each seller
        self.seller1_product_info: Optional[Dict[str, Any]] = None
        self.seller2_product_info: Optional[Dict[str, Any]] = None
    
    def reset(
        self,
        user_requirement: str = "",
        seller1_product_info: Optional[Dict[str, Any]] = None,
        seller2_product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new negotiation
        
        Args:
            user_requirement: User requirement description
            seller1_product_info: Product information for seller1
            seller2_product_info: Product information for seller2
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
        self.current_selected_seller = None
        self.final_selected_seller = None
        self.final_deal_price = None
        
        # Store product info for each seller
        self.seller1_product_info = seller1_product_info or {}
        self.seller2_product_info = seller2_product_info or {}
        
        # Initialize Buyer Agent (buyer knows about both sellers and their products)
        buyer_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
            "num_sellers": 2,  # Inform buyer there are 2 sellers
            "negotiation_mode": "sequential",  # Inform buyer this is sequential negotiation
        }
        self.buyer_agent.initialize(buyer_context)
        
        # Initialize Seller1 Agent with its own product
        seller1_context = {
            "product_info": self.seller1_product_info,
            "initial_price": self.initial_seller1_price,
            "min_price": self.seller1_min_price,
            "environment_info": self.environment_info,
            "seller_id": 1,  # Identify as seller 1
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent with its own product
        seller2_context = {
            "product_info": self.seller2_product_info,
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
        selected_seller: int,  # 1 or 2, which seller buyer chooses to negotiate with this round
        buyer_action: Optional[str] = None,
        seller_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, buyer chooses one seller to negotiate with, then buyer and that seller exchange messages.
        Order: buyer -> seller
        
        Args:
            selected_seller: Which seller (1 or 2) buyer chooses to negotiate with this round
            buyer_action: Buyer's response (optional)
            seller_action: Selected seller's response (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if selected_seller not in [1, 2]:
            raise ValueError(f"selected_seller must be 1 or 2, got {selected_seller}")
        
        self.current_selected_seller = selected_seller
        
        # Add messages to memory in order: buyer -> seller
        # Process buyer action first
        if buyer_action is not None:
            if selected_seller == 1:
                self.memory_seller1.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_seller1.update(buyer_price=buyer_price)
            else:  # selected_seller == 2
                self.memory_seller2.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_seller2.update(buyer_price=buyer_price)
        
        # Process seller action after buyer
        if seller_action is not None:
            if selected_seller == 1:
                self.memory_seller1.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_seller1.update(seller_price=seller_price)
            else:  # selected_seller == 2
                self.memory_seller2.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_seller2.update(seller_price=seller_price)
        
        # Check if deal can be made with the selected seller
        # Buyer must explicitly express make deal intent AND price_tolerance condition must be satisfied
        if selected_seller == 1:
            if (buyer_action is not None and 
                self._check_make_deal(buyer_action) and
                self.state_seller1.buyer_price is not None and 
                self.state_seller1.seller_price is not None):
                price_diff = abs(self.state_seller1.buyer_price - self.state_seller1.seller_price)
                # Only make deal if buyer wants to make deal AND prices are within tolerance
                if price_diff <= self.price_tolerance:
                    self.final_selected_seller = 1
                    self.final_deal_price = (self.state_seller1.buyer_price + self.state_seller1.seller_price) / 2
        else:  # selected_seller == 2
            if (buyer_action is not None and 
                self._check_make_deal(buyer_action) and
                self.state_seller2.buyer_price is not None and 
                self.state_seller2.seller_price is not None):
                price_diff = abs(self.state_seller2.buyer_price - self.state_seller2.seller_price)
                # Only make deal if buyer wants to make deal AND prices are within tolerance
                if price_diff <= self.price_tolerance:
                    self.final_selected_seller = 2
                    self.final_deal_price = (self.state_seller2.buyer_price + self.state_seller2.seller_price) / 2
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer_reward = 0.0
        seller1_reward = 0.0
        seller2_reward = 0.0
        
        if self.final_selected_seller is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            reward = self._calculate_reward()
            buyer_reward = self._calculate_buyer_reward()
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            reward = self._calculate_reward()
            buyer_reward = self._calculate_buyer_reward()
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        # Only calculate for the selected seller in this round (sequential negotiation)
        step_buyer_reward = self._calculate_step_buyer_reward()
        step_seller1_reward = self._calculate_step_seller_reward(1) if selected_seller == 1 else None
        step_seller2_reward = self._calculate_step_seller_reward(2) if selected_seller == 2 else None
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_buyer_reward"] = step_buyer_reward
        if step_seller1_reward is not None:
            info["step_seller1_reward"] = step_seller1_reward
        if step_seller2_reward is not None:
            info["step_seller2_reward"] = step_seller2_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_seller"] = self.final_selected_seller
                info["final_deal_price"] = self.final_deal_price
            info["buyer_reward"] = buyer_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
        
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
        output_lines.append(f"Round {self.current_round} - Sequential Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display which seller was selected this round
        if self.current_selected_seller is not None:
            output_lines.append(f"\n[Selected Seller: Seller {self.current_selected_seller}]")
        
        # Display Seller1 conversation (if this round negotiated with seller1)
        if self.current_selected_seller == 1:
            output_lines.append(f"\n[SELLER 1 Conversation]:")
            history_seller1 = self.memory_seller1.get_history()
            if history_seller1:
                current_round_messages_s1 = [
                    msg for msg in history_seller1 if msg["round"] == self.current_round
                ]
                for msg in current_round_messages_s1:
                    role = msg["role"].upper()
                    output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Seller2 conversation (if this round negotiated with seller2)
        if self.current_selected_seller == 2:
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
        if self.final_selected_seller is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Seller {self.final_selected_seller}")
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
            "current_selected_seller": self.current_selected_seller,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "status": self.negotiation_info.status.value,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "current_selected_seller": self.current_selected_seller,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
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
        """Calculate global reward
        
        Calculate reward value based on negotiation result.
        If deal is reached with a seller, use that seller's min_price for calculation.
        
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
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected seller's min_price
            selected_seller_min_price = None
            if self.final_selected_seller == 1:
                selected_seller_min_price = self.seller1_min_price
            elif self.final_selected_seller == 2:
                selected_seller_min_price = self.seller2_min_price
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price
            if selected_seller_min_price is not None:
                seller_profit = deal_price - selected_seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if self.buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if selected_seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller{self.final_selected_seller}_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, seller{self.final_selected_seller}_min={selected_seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self) -> float:
        """Calculate reward from buyer's perspective
        
        Calculate reward value based on negotiation result from buyer's perspective.
        This reward does not include seller profit.
        
        If deal is reached:
            reward = buyer savings + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from buyer's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if self.buyer_max_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self, seller_id: int) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        
        If deal is reached with this seller:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached or reached with another seller:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Args:
            seller_id: Seller ID (1 or 2)
        
        Returns:
            Reward value from seller's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        # Check if deal was reached with this seller
        deal_reached_with_this_seller = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.final_selected_seller == seller_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_seller:
            # Deal reached with this seller: seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            # Get this seller's min_price
            seller_min_price = None
            if seller_id == 1:
                seller_min_price = self.seller1_min_price
            elif seller_id == 2:
                seller_min_price = self.seller2_min_price
            
            # Calculate seller profit: deal_price - seller_min_price
            if seller_min_price is not None:
                seller_profit = deal_price - seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller{seller_id} Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller{seller_id}_min={seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached or reached with another seller: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller{seller_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this seller)")
            return weighted_time_cost
    
    def _calculate_step_buyer_reward(self) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round with the selected seller.
        This is calculated every round, not just at the end.
        
        reward = buyer savings (from current offer) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        
        # Calculate buyer reward with the selected seller
        reward = 0.0
        buyer_savings = 0.0
        
        # Get buyer price from the selected seller
        if self.current_selected_seller == 1:
            buyer_price = self.state_seller1.buyer_price
        elif self.current_selected_seller == 2:
            buyer_price = self.state_seller2.buyer_price
        else:
            buyer_price = None
        
        # Calculate buyer savings: buyer_max_price - buyer_price
        if buyer_price is not None and self.buyer_max_price is not None:
            buyer_savings = self.buyer_max_price - buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self, seller_id: int) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If seller_price is not specified yet, only round cost is returned.
        
        Args:
            seller_id: Seller ID (1 or 2)
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        # Get seller state
        seller_state = None
        seller_min_price = None
        if seller_id == 1:
            seller_state = self.state_seller1
            seller_min_price = self.seller1_min_price
        elif seller_id == 2:
            seller_state = self.state_seller2
            seller_min_price = self.seller2_min_price
        
        # Calculate seller profit from current offer: seller_price - seller_min_price
        if seller_state is not None and seller_state.seller_price is not None and seller_min_price is not None:
            seller_profit = seller_state.seller_price - seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward

