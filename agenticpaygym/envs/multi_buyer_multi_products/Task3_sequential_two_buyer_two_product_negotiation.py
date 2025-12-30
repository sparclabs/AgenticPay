"""Task3 Sequential Two-Buyer Two-Product Negotiation Environment Implementation

Supports sequential negotiation where seller chooses one buyer per round to negotiate with.
Seller can switch between two buyers and make a deal with either buyer.
Prices represent total price for both products.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task3SequentialTwoBuyerTwoProductNegotiation(BaseEnv):
    """Task3 Sequential Two-Buyer Two-Product Negotiation Environment
    
    Manages sequential negotiation process where seller chooses one buyer per round to negotiate with.
    Seller can switch between two buyers and make a deal with either buyer.
    Prices represent total price for both products.
    """
    
    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_price: float = 100.0,
        buyer1_max_price: Optional[float] = None,
        buyer2_max_price: Optional[float] = None,
        seller_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize sequential multi-buyer multi-product negotiation environment
        
        Args:
            buyer1_agent: First Buyer Agent
            buyer2_agent: Second Buyer Agent
            seller_agent: Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller_price: Initial total price offered by seller for both products
            buyer1_max_price: Maximum acceptable total price for buyer1 (confidential, for both products)
            buyer2_max_price: Maximum acceptable total price for buyer2 (confidential, for both products)
            seller_min_price: Minimum acceptable total price for seller (confidential, for both products)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
            reward_weights: Reward weights configuration dict with keys:
                - buyer_savings: weight for buyer savings (default: 1.0)
                - seller_profit: weight for seller profit (default: 1.0)
                - time_cost: weight for time cost (default: 0.1)
        """
        self.buyer1_agent = buyer1_agent
        self.buyer2_agent = buyer2_agent
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_price = initial_seller_price
        self.buyer1_max_price = buyer1_max_price
        self.buyer2_max_price = buyer2_max_price
        self.seller_min_price = seller_min_price
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
        
        # State management - separate for each buyer
        self.memory_buyer1 = ConversationMemory()
        self.memory_buyer2 = ConversationMemory()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info: Optional[Dict[str, Any]] = None
        
        # Track which buyer is currently selected and which buyer was chosen for the deal
        self.current_selected_buyer: Optional[int] = None  # 1 or 2, selected for current round
        self.final_selected_buyer: Optional[int] = None  # 1 or 2, chosen for final deal
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
            user_requirement: User requirement description (should describe purchasing two products)
            product_info: Product information containing two products and their prices
                Expected format: {
                    "products": [
                        {"name": "Product1", "price": 100.0, ...},
                        {"name": "Product2", "price": 80.0, ...}
                    ]
                }
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.current_selected_buyer = None
        self.final_selected_buyer = None
        self.final_deal_price = None
        self.product_info = product_info or {}
        
        # Extract product information
        products = self.product_info.get("products", [])
        if len(products) < 2:
            raise ValueError("product_info must contain at least 2 products in 'products' list")
        
        # Calculate total price of both products
        total_product_price = sum(p.get("price", 0.0) for p in products)
        
        # Initialize Buyer1 Agent
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 1,  # Identify as buyer 1
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 2,  # Identify as buyer 2
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Seller Agent (seller knows about both buyers and both products)
        seller_context = {
            "product_info": self.product_info,  # Seller can see both products
            "initial_price": self.initial_seller_price,  # Initial total price
            "min_price": self.seller_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "num_buyers": 2,  # Inform seller there are 2 buyers
            "negotiation_mode": "sequential",  # Inform seller this is sequential negotiation
        }
        self.seller_agent.initialize(seller_context)
        
        # No initial seller offer - negotiation starts with buyers' first messages
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        selected_buyer: int,  # 1 or 2, which buyer seller chooses to negotiate with this round
        seller_action: Optional[str] = None,
        buyer_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, seller chooses one buyer to negotiate with, then buyer and seller exchange messages.
        Order: buyer -> seller (buyer responds first, then seller can see buyer's message)
        Prices represent total price for both products.
        
        Args:
            selected_buyer: Which buyer (1 or 2) seller chooses to negotiate with this round
            seller_action: Seller's response (optional)
            buyer_action: Selected buyer's response (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if selected_buyer not in [1, 2]:
            raise ValueError(f"selected_buyer must be 1 or 2, got {selected_buyer}")
        
        self.current_selected_buyer = selected_buyer
        
        # Add messages to memory in order: buyer -> seller
        # Process buyer action first
        if buyer_action is not None:
            if selected_buyer == 1:
                self.memory_buyer1.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer1.update(buyer_price=buyer_price)
            else:  # selected_buyer == 2
                self.memory_buyer2.add_message("buyer", buyer_action, self.current_round)
                buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer2.update(buyer_price=buyer_price)
        
        # Process seller action after buyer (seller can see buyer's message)
        if seller_action is not None:
            if selected_buyer == 1:
                self.memory_buyer1.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer1.update(seller_price=seller_price)
            else:  # selected_buyer == 2
                self.memory_buyer2.add_message("seller", seller_action, self.current_round)
                seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer2.update(seller_price=seller_price)
        
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
        else:  # selected_buyer == 2
            if (buyer_action is not None and 
                self._check_make_deal(buyer_action) and
                self.state_buyer2.buyer_price is not None and 
                self.state_buyer2.seller_price is not None):
                price_diff = abs(self.state_buyer2.buyer_price - self.state_buyer2.seller_price)
                # Only make deal if buyer wants to make deal AND prices are within tolerance
                if price_diff <= self.price_tolerance:
                    self.final_selected_buyer = 2
                    self.final_deal_price = (self.state_buyer2.buyer_price + self.state_buyer2.seller_price) / 2
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer1_reward = 0.0
        buyer2_reward = 0.0
        seller_reward = 0.0
        
        if self.final_selected_buyer is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller_reward = self._calculate_seller_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller_reward = self._calculate_seller_reward()
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        # Only calculate for the selected buyer in this round (sequential negotiation)
        step_buyer1_reward = self._calculate_step_buyer_reward(1) if selected_buyer == 1 else None
        step_buyer2_reward = self._calculate_step_buyer_reward(2) if selected_buyer == 2 else None
        step_seller_reward = self._calculate_step_seller_reward()
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        if step_buyer1_reward is not None:
            info["step_buyer1_reward"] = step_buyer1_reward
        if step_buyer2_reward is not None:
            info["step_buyer2_reward"] = step_buyer2_reward
        info["step_seller_reward"] = step_seller_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.final_selected_buyer
                info["final_deal_price"] = self.final_deal_price
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["seller_reward"] = seller_reward
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays seller and buyer outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        Prices shown are total prices for both products.
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
        # Get messages from the round that just completed
        # Note: In step(), messages are added to current_round
        # - If agreement reached: current_round stays the same, messages are in current_round
        # - If no agreement: current_round is incremented, messages are in current_round - 1
        history_buyer1 = self.memory_buyer1.get_history()
        history_buyer2 = self.memory_buyer2.get_history()
        
        # Determine which round's messages to display
        # If negotiation is agreed or timed out, messages are in current_round
        # Otherwise, messages are in current_round - 1 (because current_round was incremented)
        if self.negotiation_info.status in [NegotiationStatus.AGREED, NegotiationStatus.TIMEOUT]:
            round_to_display = self.current_round
            display_round = self.current_round
        else:
            round_to_display = self.current_round - 1 if self.current_round > 0 else 0
            display_round = self.current_round if self.current_round > 0 else 0
        
        # Display product info
        if self.product_info:
            products = self.product_info.get("products", [])
            if products:
                output_lines.append(f"\n{'='*60}")
                output_lines.append("Products:")
                for i, p in enumerate(products, 1):
                    name = p.get("name", f"Product {i}")
                    price = p.get("price", 0.0)
                    output_lines.append(f"  {i}. {name}: ${price:.2f}")
                total_price = sum(p.get("price", 0.0) for p in products)
                output_lines.append(f"  Total Product Price: ${total_price:.2f}")
                output_lines.append(f"{'='*60}")
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Round {display_round} - Sequential Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display which buyer was selected this round
        if self.current_selected_buyer is not None:
            output_lines.append(f"\n[Selected Buyer: Buyer {self.current_selected_buyer}]")
        
        # Display Buyer1 conversation (if this round negotiated with buyer1)
        if self.current_selected_buyer == 1 and history_buyer1:
            round_messages_b1 = [
                msg for msg in history_buyer1 if msg["round"] == round_to_display
            ]
            if round_messages_b1:
                output_lines.append(f"\n[BUYER 1 Conversation]:")
                # Display buyer message first (if exists)
                buyer_msg_b1 = next(
                    (msg for msg in round_messages_b1 if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg_b1:
                    output_lines.append(f"  [BUYER]: {buyer_msg_b1['content']}")
                
                # Display seller message (if exists)
                seller_msg_b1 = next(
                    (msg for msg in round_messages_b1 if msg["role"] == "seller"), 
                    None
                )
                if seller_msg_b1:
                    output_lines.append(f"  [SELLER]: {seller_msg_b1['content']}")
        
        # Display Buyer2 conversation (if this round negotiated with buyer2)
        if self.current_selected_buyer == 2 and history_buyer2:
            round_messages_b2 = [
                msg for msg in history_buyer2 if msg["round"] == round_to_display
            ]
            if round_messages_b2:
                output_lines.append(f"\n[BUYER 2 Conversation]:")
                # Display buyer message first (if exists)
                buyer_msg_b2 = next(
                    (msg for msg in round_messages_b2 if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg_b2:
                    output_lines.append(f"  [BUYER]: {buyer_msg_b2['content']}")
                
                # Display seller message (if exists)
                seller_msg_b2 = next(
                    (msg for msg in round_messages_b2 if msg["role"] == "seller"), 
                    None
                )
                if seller_msg_b2:
                    output_lines.append(f"  [SELLER]: {seller_msg_b2['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {display_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Buyer1 prices (total for both products)
        output_lines.append(f"\nBuyer 1:")
        if self.state_buyer1.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_buyer1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_buyer1.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_buyer1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display Buyer2 prices (total for both products)
        output_lines.append(f"\nBuyer 2:")
        if self.state_buyer2.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_buyer2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_buyer2.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_buyer2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display deal status
        if self.final_selected_buyer is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Buyer {self.final_selected_buyer}")
            if self.final_deal_price is not None:
                output_lines.append(f"  Final Deal Total Price: ${self.final_deal_price:.2f}")
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
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history_buyer1": self.memory_buyer1.get_history(),
            "conversation_history_buyer2": self.memory_buyer2.get_history(),
            "current_round": self.current_round,
            "current_selected_buyer": self.current_selected_buyer,
            "buyer1_price": self.state_buyer1.buyer_price,  # Total price for both products
            "seller_price_buyer1": self.state_buyer1.seller_price,  # Total price for both products
            "buyer2_price": self.state_buyer2.buyer_price,  # Total price for both products
            "seller_price_buyer2": self.state_buyer2.seller_price,  # Total price for both products
            "status": self.negotiation_info.status.value,
            "final_selected_buyer": self.final_selected_buyer,
            "final_deal_price": self.final_deal_price,
            "product_info": self.product_info,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "current_selected_buyer": self.current_selected_buyer,
            "buyer1_price": self.state_buyer1.buyer_price,  # Total price for both products
            "seller_price_buyer1": self.state_buyer1.seller_price,  # Total price for both products
            "buyer2_price": self.state_buyer2.buyer_price,  # Total price for both products
            "seller_price_buyer2": self.state_buyer2.seller_price,  # Total price for both products
            "final_selected_buyer": self.final_selected_buyer,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
            "product_info": self.product_info,
        }
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text
        
        Priority: 
        1. Extract from ### BUYER_PRICE($X) ### or ### SELLER_PRICE($X) ### format (preferred)
        2. Fall back to ### $X ### format
        3. Fall back to other price patterns
        
        Args:
            text: Text containing price
            
        Returns:
            Extracted price, returns None if not found
        """
        # Priority 1: Extract price from ### BUYER_PRICE($X) ### or ### SELLER_PRICE($X) ### format
        # Matches: ### BUYER_PRICE($100.50) ###, ### SELLER_PRICE($150) ###, etc.
        labeled_price_pattern = r'###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$(\d+\.?\d*)\)\s*###'
        matches = re.findall(labeled_price_pattern, text, re.IGNORECASE)
        if matches:
            try:
                price = float(matches[-1])  # Take the last match
                if price > 0:
                    return price
            except ValueError:
                pass
        
        # Priority 2: Extract price from ### $X ### format (backward compatibility)
        # Matches: ### $100.50 ###, ### $100 ###, ###$120###, etc.
        triple_hash_pattern = r'###\s*\$(\d+\.?\d*)\s*###'
        matches = re.findall(triple_hash_pattern, text, re.IGNORECASE)
        if matches:
            try:
                price = float(matches[-1])  # Take the last match
                if price > 0:
                    return price
            except ValueError:
                pass
        
        # Priority 3: Fall back to other price patterns
        fallback_patterns = [
            r'\$(\d+\.?\d*)',  # $100.50 or $100
            r'(\d+\.?\d*)\s*dollars?',  # 100.50 dollars
            r'(\d+\.?\d*)\s*USD',  # 100.50 USD
            r'price.*?(\d+\.?\d*)',  # price 100.50
            r'offer.*?(\d+\.?\d*)',  # offer 100.50
            r'total.*?(\d+\.?\d*)',  # total 100.50
        ]
        
        for pattern in fallback_patterns:
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
        If deal is reached with a buyer, use that buyer's max_price for calculation.
        Prices represent total price for both products.
        
        If deal is reached:
            reward = buyer savings + seller profit + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer for both products)
            - seller profit = deal_price - seller_min_price (extra profit for seller for both products)
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
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected buyer's max_price
            selected_buyer_max_price = None
            if self.final_selected_buyer == 1:
                selected_buyer_max_price = self.buyer1_max_price
            elif self.final_selected_buyer == 2:
                selected_buyer_max_price = self.buyer2_max_price
            
            # Calculate buyer savings: buyer_max_price - deal_price (for both products)
            if selected_buyer_max_price is not None:
                buyer_savings = selected_buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if selected_buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if self.seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer{self.final_selected_buyer}_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{self.final_selected_buyer}_max={selected_buyer_max_price}, deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self, buyer_id: int) -> float:
        """Calculate reward from buyer's perspective
        
        Calculate reward value based on negotiation result from buyer's perspective.
        This reward does not include seller profit.
        Prices represent total price for both products.
        
        If deal is reached with this buyer:
            reward = buyer savings + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer for both products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached or reached with another buyer:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Reward value from buyer's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        # Check if deal was reached with this buyer
        deal_reached_with_this_buyer = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.final_selected_buyer == buyer_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_buyer:
            # Deal reached with this buyer: buyer savings + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            # Get this buyer's max_price
            buyer_max_price = None
            if buyer_id == 1:
                buyer_max_price = self.buyer1_max_price
            elif buyer_id == 2:
                buyer_max_price = self.buyer2_max_price
            
            # Calculate buyer savings: buyer_max_price - deal_price (for both products)
            if buyer_max_price is not None:
                buyer_savings = buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if buyer_max_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{buyer_id}_max={buyer_max_price}, deal_price={deal_price:.2f}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached or reached with another buyer: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this buyer)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        Prices represent total price for both products.
        
        If deal is reached:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller for both products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from seller's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_deal_price is not None:
            # Deal reached: seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if self.seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_step_buyer_reward(self, buyer_id: int) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round.
        This is calculated every round, not just at the end.
        Prices represent total price for both products.
        
        reward = buyer savings (from current offer) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer for both products)
        - round cost = -current_round (penalty for number of rounds taken)
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        
        # Calculate buyer reward
        reward = 0.0
        buyer_savings = 0.0
        
        # Get buyer state
        buyer_state = None
        buyer_max_price = None
        if buyer_id == 1:
            buyer_state = self.state_buyer1
            buyer_max_price = self.buyer1_max_price
        elif buyer_id == 2:
            buyer_state = self.state_buyer2
            buyer_max_price = self.buyer2_max_price
        
        # Calculate buyer savings from current offer: buyer_max_price - buyer_price (for both products)
        if buyer_state is not None and buyer_state.buyer_price is not None and buyer_max_price is not None:
            buyer_savings = buyer_max_price - buyer_state.buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        Prices represent total price for both products.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer for both products)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If seller_price is not specified yet, only round cost is returned.
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        # Get seller price from the selected buyer
        seller_price = None
        if self.current_selected_buyer == 1:
            seller_price = self.state_buyer1.seller_price
        elif self.current_selected_buyer == 2:
            seller_price = self.state_buyer2.seller_price
        
        # Calculate seller profit from current offer: seller_price - seller_min_price (for both products)
        if seller_price is not None and self.seller_min_price is not None:
            seller_profit = seller_price - self.seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward

