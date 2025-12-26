"""Task3 Sequential Two-Buyer Two-Seller Two-Product Negotiation Environment Implementation

Supports sequential negotiation where two buyers each choose one seller per round to negotiate with
for two products. Each buyer can switch between two sellers and make a deal with either seller.
Prices represent total price for both products.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation(BaseEnv):
    """Task3 Sequential Two-Buyer Two-Seller Two-Product Negotiation Environment
    
    Manages sequential negotiation process where each buyer chooses one seller per round to negotiate with
    for two products. Each buyer can switch between two sellers and make a deal with either seller.
    Prices represent total price for both products.
    """
    
    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_price: float = 200.0,
        initial_seller2_price: float = 220.0,
        buyer1_max_price: Optional[float] = None,
        buyer2_max_price: Optional[float] = None,
        seller1_min_price: Optional[float] = None,
        seller2_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize sequential multi-buyer multi-seller multi-product negotiation environment
        
        Args:
            buyer1_agent: First Buyer Agent
            buyer2_agent: Second Buyer Agent
            seller1_agent: First Seller Agent
            seller2_agent: Second Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller1_price: Initial total price offered by seller1 for both products
            initial_seller2_price: Initial total price offered by seller2 for both products
            buyer1_max_price: Maximum acceptable total price for buyer1 (confidential, for both products)
            buyer2_max_price: Maximum acceptable total price for buyer2 (confidential, for both products)
            seller1_min_price: Minimum acceptable total price for seller1 (confidential, for both products)
            seller2_min_price: Minimum acceptable total price for seller2 (confidential, for both products)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
            reward_weights: Reward weights configuration dict with keys:
                - buyer_savings: weight for buyer savings (default: 1.0)
                - seller_profit: weight for seller profit (default: 1.0)
                - time_cost: weight for time cost (default: 0.1)
        """
        self.buyer1_agent = buyer1_agent
        self.buyer2_agent = buyer2_agent
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.max_rounds = max_rounds
        self.initial_seller1_price = initial_seller1_price
        self.initial_seller2_price = initial_seller2_price
        self.buyer1_max_price = buyer1_max_price
        self.buyer2_max_price = buyer2_max_price
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
        
        # State management - separate for each buyer-seller pair
        # buyer1-seller1
        self.memory_b1s1 = ConversationMemory()
        self.state_b1s1 = NegotiationState()
        # buyer1-seller2
        self.memory_b1s2 = ConversationMemory()
        self.state_b1s2 = NegotiationState()
        # buyer2-seller1
        self.memory_b2s1 = ConversationMemory()
        self.state_b2s1 = NegotiationState()
        # buyer2-seller2
        self.memory_b2s2 = ConversationMemory()
        self.state_b2s2 = NegotiationState()
        
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info: Optional[Dict[str, Any]] = None
        
        # Track which seller each buyer selected for current round and final deal
        self.buyer1_selected_seller: Optional[int] = None  # 1 or 2, selected for current round
        self.buyer2_selected_seller: Optional[int] = None  # 1 or 2, selected for current round
        self.final_selected_buyer: Optional[int] = None  # 1 or 2, chosen for final deal
        self.final_selected_seller: Optional[int] = None  # 1 or 2, chosen for final deal
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
        self.memory_b1s1.clear()
        self.memory_b1s2.clear()
        self.memory_b2s1.clear()
        self.memory_b2s2.clear()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.buyer1_selected_seller = None
        self.buyer2_selected_seller = None
        self.final_selected_buyer = None
        self.final_selected_seller = None
        self.final_deal_price = None
        self.product_info = product_info or {}
        
        # Extract product information
        products = self.product_info.get("products", [])
        if len(products) < 2:
            raise ValueError("product_info must contain at least 2 products in 'products' list")
        
        # Calculate total price of both products
        total_product_price = sum(p.get("price", 0.0) for p in products)
        
        # Initialize Buyer1 Agent (buyer1 knows about both sellers)
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 1,
            "num_sellers": 2,  # Inform buyer there are 2 sellers
            "negotiation_mode": "sequential",  # Inform buyer this is sequential negotiation
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent (buyer2 knows about both sellers)
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 2,
            "num_sellers": 2,  # Inform buyer there are 2 sellers
            "negotiation_mode": "sequential",  # Inform buyer this is sequential negotiation
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Seller1 Agent
        seller1_context = {
            "product_info": self.product_info,  # Seller can see both products
            "initial_price": self.initial_seller1_price,  # Initial total price for both products
            "min_price": self.seller1_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "seller_id": 1,  # Identify as seller 1
            "num_buyers": 2,  # Inform seller there are 2 buyers
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent
        seller2_context = {
            "product_info": self.product_info,  # Seller can see both products
            "initial_price": self.initial_seller2_price,  # Initial total price for both products
            "min_price": self.seller2_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "seller_id": 2,  # Identify as seller 2
            "num_buyers": 2,  # Inform seller there are 2 buyers
        }
        self.seller2_agent.initialize(seller2_context)
        
        # Sellers give initial offers to both buyers (total price for both products)
        product_names = [p.get("name", "Product") for p in products]
        initial_message_seller1 = f"I'm offering {product_names[0]} and {product_names[1]} for a total of ${self.initial_seller1_price:.2f}."
        initial_message_seller2 = f"I'm offering {product_names[0]} and {product_names[1]} for a total of ${self.initial_seller2_price:.2f}."
        
        # buyer1-seller1
        self.memory_b1s1.add_message("seller", initial_message_seller1, self.current_round)
        self.state_b1s1.update(seller_price=self.initial_seller1_price)
        
        # buyer1-seller2
        self.memory_b1s2.add_message("seller", initial_message_seller2, self.current_round)
        self.state_b1s2.update(seller_price=self.initial_seller2_price)
        
        # buyer2-seller1
        self.memory_b2s1.add_message("seller", initial_message_seller1, self.current_round)
        self.state_b2s1.update(seller_price=self.initial_seller1_price)
        
        # buyer2-seller2
        self.memory_b2s2.add_message("seller", initial_message_seller2, self.current_round)
        self.state_b2s2.update(seller_price=self.initial_seller2_price)
        
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        buyer1_selected_seller: int,  # 1 or 2, which seller buyer1 chooses to negotiate with this round
        buyer2_selected_seller: int,  # 1 or 2, which seller buyer2 chooses to negotiate with this round
        buyer1_action: Optional[str] = None,
        buyer2_action: Optional[str] = None,
        seller1_action_buyer1: Optional[str] = None,
        seller1_action_buyer2: Optional[str] = None,
        seller2_action_buyer1: Optional[str] = None,
        seller2_action_buyer2: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, each buyer chooses one seller to negotiate with, then buyers and sellers exchange messages.
        Order: buyer -> seller
        Prices represent total price for both products.
        
        Args:
            buyer1_selected_seller: Which seller (1 or 2) buyer1 chooses to negotiate with this round
            buyer2_selected_seller: Which seller (1 or 2) buyer2 chooses to negotiate with this round
            buyer1_action: Buyer1's response (optional)
            buyer2_action: Buyer2's response (optional)
            seller1_action_buyer1: Seller1's response to buyer1 (optional, only if buyer1 selected seller1)
            seller1_action_buyer2: Seller1's response to buyer2 (optional, only if buyer2 selected seller1)
            seller2_action_buyer1: Seller2's response to buyer1 (optional, only if buyer1 selected seller2)
            seller2_action_buyer2: Seller2's response to buyer2 (optional, only if buyer2 selected seller2)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if buyer1_selected_seller not in [1, 2]:
            raise ValueError(f"buyer1_selected_seller must be 1 or 2, got {buyer1_selected_seller}")
        if buyer2_selected_seller not in [1, 2]:
            raise ValueError(f"buyer2_selected_seller must be 1 or 2, got {buyer2_selected_seller}")
        
        self.buyer1_selected_seller = buyer1_selected_seller
        self.buyer2_selected_seller = buyer2_selected_seller
        
        # Process buyer actions first
        # buyer1 action
        if buyer1_action is not None:
            if buyer1_selected_seller == 1:
                self.memory_b1s1.add_message("buyer", buyer1_action, self.current_round)
                buyer_price = self._extract_price(buyer1_action)
                if buyer_price is not None:
                    self.state_b1s1.update(buyer_price=buyer_price)
            else:  # buyer1_selected_seller == 2
                self.memory_b1s2.add_message("buyer", buyer1_action, self.current_round)
                buyer_price = self._extract_price(buyer1_action)
                if buyer_price is not None:
                    self.state_b1s2.update(buyer_price=buyer_price)
        
        # buyer2 action
        if buyer2_action is not None:
            if buyer2_selected_seller == 1:
                self.memory_b2s1.add_message("buyer", buyer2_action, self.current_round)
                buyer_price = self._extract_price(buyer2_action)
                if buyer_price is not None:
                    self.state_b2s1.update(buyer_price=buyer_price)
            else:  # buyer2_selected_seller == 2
                self.memory_b2s2.add_message("buyer", buyer2_action, self.current_round)
                buyer_price = self._extract_price(buyer2_action)
                if buyer_price is not None:
                    self.state_b2s2.update(buyer_price=buyer_price)
        
        # Process seller actions after buyers
        # seller1-buyer1
        if buyer1_selected_seller == 1 and seller1_action_buyer1 is not None:
            self.memory_b1s1.add_message("seller", seller1_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer1)
            if seller_price is not None:
                self.state_b1s1.update(seller_price=seller_price)
        
        # seller1-buyer2
        if buyer2_selected_seller == 1 and seller1_action_buyer2 is not None:
            self.memory_b2s1.add_message("seller", seller1_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer2)
            if seller_price is not None:
                self.state_b2s1.update(seller_price=seller_price)
        
        # seller2-buyer1
        if buyer1_selected_seller == 2 and seller2_action_buyer1 is not None:
            self.memory_b1s2.add_message("seller", seller2_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer1)
            if seller_price is not None:
                self.state_b1s2.update(seller_price=seller_price)
        
        # seller2-buyer2
        if buyer2_selected_seller == 2 and seller2_action_buyer2 is not None:
            self.memory_b2s2.add_message("seller", seller2_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer2)
            if seller_price is not None:
                self.state_b2s2.update(seller_price=seller_price)
        
        # Check if deal can be made with the selected sellers
        # Buyer must explicitly express make deal intent AND price_tolerance condition must be satisfied
        deals = []  # List of (buyer_id, seller_id, price) tuples
        
        # Check buyer1-seller1
        if (buyer1_selected_seller == 1 and
            buyer1_action is not None and 
            self._check_make_deal(buyer1_action) and
            self.state_b1s1.buyer_price is not None and 
            self.state_b1s1.seller_price is not None):
            price_diff = abs(self.state_b1s1.buyer_price - self.state_b1s1.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (self.state_b1s1.buyer_price + self.state_b1s1.seller_price) / 2
                deals.append((1, 1, deal_price))
        
        # Check buyer1-seller2
        if (buyer1_selected_seller == 2 and
            buyer1_action is not None and 
            self._check_make_deal(buyer1_action) and
            self.state_b1s2.buyer_price is not None and 
            self.state_b1s2.seller_price is not None):
            price_diff = abs(self.state_b1s2.buyer_price - self.state_b1s2.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (self.state_b1s2.buyer_price + self.state_b1s2.seller_price) / 2
                deals.append((1, 2, deal_price))
        
        # Check buyer2-seller1
        if (buyer2_selected_seller == 1 and
            buyer2_action is not None and 
            self._check_make_deal(buyer2_action) and
            self.state_b2s1.buyer_price is not None and 
            self.state_b2s1.seller_price is not None):
            price_diff = abs(self.state_b2s1.buyer_price - self.state_b2s1.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (self.state_b2s1.buyer_price + self.state_b2s1.seller_price) / 2
                deals.append((2, 1, deal_price))
        
        # Check buyer2-seller2
        if (buyer2_selected_seller == 2 and
            buyer2_action is not None and 
            self._check_make_deal(buyer2_action) and
            self.state_b2s2.buyer_price is not None and 
            self.state_b2s2.seller_price is not None):
            price_diff = abs(self.state_b2s2.buyer_price - self.state_b2s2.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (self.state_b2s2.buyer_price + self.state_b2s2.seller_price) / 2
                deals.append((2, 2, deal_price))
        
        # Select the best deal: prioritize buyer's preference (lower price) and seller's preference (higher price)
        # If multiple deals exist, choose the one with the best price for both parties
        if deals:
            best_deal = None
            best_utility = float('-inf')
            
            for buyer_id, seller_id, price in deals:
                buyer_max = self.buyer1_max_price if buyer_id == 1 else self.buyer2_max_price
                seller_min = self.seller1_min_price if seller_id == 1 else self.seller2_min_price
                
                buyer_savings = (buyer_max - price) if buyer_max is not None else 0
                seller_profit = (price - seller_min) if seller_min is not None else 0
                utility = buyer_savings + seller_profit
                
                if utility > best_utility:
                    best_utility = utility
                    best_deal = (buyer_id, seller_id, price)
            
            if best_deal:
                self.final_selected_buyer, self.final_selected_seller, self.final_deal_price = best_deal
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer1_reward = 0.0
        buyer2_reward = 0.0
        seller1_reward = 0.0
        seller2_reward = 0.0
        
        if self.final_selected_buyer is not None and self.final_selected_seller is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        # Only calculate for the selected sellers in this round (sequential negotiation)
        step_buyer1_reward = self._calculate_step_buyer_reward(1)
        step_buyer2_reward = self._calculate_step_buyer_reward(2)
        step_seller1_reward = self._calculate_step_seller_reward(1) if (buyer1_selected_seller == 1 or buyer2_selected_seller == 1) else None
        step_seller2_reward = self._calculate_step_seller_reward(2) if (buyer1_selected_seller == 2 or buyer2_selected_seller == 2) else None
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_buyer1_reward"] = step_buyer1_reward
        info["step_buyer2_reward"] = step_buyer2_reward
        if step_seller1_reward is not None:
            info["step_seller1_reward"] = step_seller1_reward
        if step_seller2_reward is not None:
            info["step_seller2_reward"] = step_seller2_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.final_selected_buyer
                info["selected_seller"] = self.final_selected_seller
                info["final_deal_price"] = self.final_deal_price
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays buyer and seller outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        Prices shown are total prices for both products.
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
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
        output_lines.append(f"Round {self.current_round} - Sequential Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display which seller each buyer selected this round
        if self.buyer1_selected_seller is not None:
            output_lines.append(f"\n[Buyer 1 Selected Seller: Seller {self.buyer1_selected_seller}]")
        if self.buyer2_selected_seller is not None:
            output_lines.append(f"[Buyer 2 Selected Seller: Seller {self.buyer2_selected_seller}]")
        
        # Display Buyer1-Seller1 conversation (if this round buyer1 negotiated with seller1)
        if self.buyer1_selected_seller == 1:
            output_lines.append(f"\n[BUYER 1 - SELLER 1 Conversation]:")
            history_b1s1 = self.memory_b1s1.get_history()
            if history_b1s1:
                current_round_messages = [
                    msg for msg in history_b1s1 if msg["round"] == self.current_round
                ]
                for msg in current_round_messages:
                    role = msg["role"].upper()
                    output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer1-Seller2 conversation (if this round buyer1 negotiated with seller2)
        if self.buyer1_selected_seller == 2:
            output_lines.append(f"\n[BUYER 1 - SELLER 2 Conversation]:")
            history_b1s2 = self.memory_b1s2.get_history()
            if history_b1s2:
                current_round_messages = [
                    msg for msg in history_b1s2 if msg["round"] == self.current_round
                ]
                for msg in current_round_messages:
                    role = msg["role"].upper()
                    output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer2-Seller1 conversation (if this round buyer2 negotiated with seller1)
        if self.buyer2_selected_seller == 1:
            output_lines.append(f"\n[BUYER 2 - SELLER 1 Conversation]:")
            history_b2s1 = self.memory_b2s1.get_history()
            if history_b2s1:
                current_round_messages = [
                    msg for msg in history_b2s1 if msg["round"] == self.current_round
                ]
                for msg in current_round_messages:
                    role = msg["role"].upper()
                    output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer2-Seller2 conversation (if this round buyer2 negotiated with seller2)
        if self.buyer2_selected_seller == 2:
            output_lines.append(f"\n[BUYER 2 - SELLER 2 Conversation]:")
            history_b2s2 = self.memory_b2s2.get_history()
            if history_b2s2:
                current_round_messages = [
                    msg for msg in history_b2s2 if msg["round"] == self.current_round
                ]
                for msg in current_round_messages:
                    role = msg["role"].upper()
                    output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Buyer1-Seller1 prices (total for both products)
        output_lines.append(f"\nBuyer 1 - Seller 1:")
        if self.state_b1s1.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b1s1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b1s1.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b1s1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display Buyer1-Seller2 prices (total for both products)
        output_lines.append(f"\nBuyer 1 - Seller 2:")
        if self.state_b1s2.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b1s2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b1s2.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b1s2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display Buyer2-Seller1 prices (total for both products)
        output_lines.append(f"\nBuyer 2 - Seller 1:")
        if self.state_b2s1.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b2s1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b2s1.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b2s1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display Buyer2-Seller2 prices (total for both products)
        output_lines.append(f"\nBuyer 2 - Seller 2:")
        if self.state_b2s2.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b2s2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b2s2.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b2s2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display deal status
        if self.final_selected_buyer is not None and self.final_selected_seller is not None:
            output_lines.append(f"\n  ✓ DEAL MADE: Buyer {self.final_selected_buyer} with Seller {self.final_selected_seller}")
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
        self.memory_b1s1.clear()
        self.memory_b1s2.clear()
        self.memory_b2s1.clear()
        self.memory_b2s2.clear()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history_b1s1": self.memory_b1s1.get_history(),
            "conversation_history_b1s2": self.memory_b1s2.get_history(),
            "conversation_history_b2s1": self.memory_b2s1.get_history(),
            "conversation_history_b2s2": self.memory_b2s2.get_history(),
            "current_round": self.current_round,
            "buyer1_selected_seller": self.buyer1_selected_seller,
            "buyer2_selected_seller": self.buyer2_selected_seller,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,  # Total price for both products
            "b1s1_seller_price": self.state_b1s1.seller_price,  # Total price for both products
            "b1s2_buyer_price": self.state_b1s2.buyer_price,  # Total price for both products
            "b1s2_seller_price": self.state_b1s2.seller_price,  # Total price for both products
            "b2s1_buyer_price": self.state_b2s1.buyer_price,  # Total price for both products
            "b2s1_seller_price": self.state_b2s1.seller_price,  # Total price for both products
            "b2s2_buyer_price": self.state_b2s2.buyer_price,  # Total price for both products
            "b2s2_seller_price": self.state_b2s2.seller_price,  # Total price for both products
            "status": self.negotiation_info.status.value,
            "final_selected_buyer": self.final_selected_buyer,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "product_info": self.product_info,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "buyer1_selected_seller": self.buyer1_selected_seller,
            "buyer2_selected_seller": self.buyer2_selected_seller,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,  # Total price for both products
            "b1s1_seller_price": self.state_b1s1.seller_price,  # Total price for both products
            "b1s2_buyer_price": self.state_b1s2.buyer_price,  # Total price for both products
            "b1s2_seller_price": self.state_b1s2.seller_price,  # Total price for both products
            "b2s1_buyer_price": self.state_b2s1.buyer_price,  # Total price for both products
            "b2s1_seller_price": self.state_b2s1.seller_price,  # Total price for both products
            "b2s2_buyer_price": self.state_b2s2.buyer_price,  # Total price for both products
            "b2s2_seller_price": self.state_b2s2.seller_price,  # Total price for both products
            "final_selected_buyer": self.final_selected_buyer,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
            "product_info": self.product_info,
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
            r'total.*?(\d+\.?\d*)',  # total 100.50
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
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected buyer's max_price and seller's min_price
            selected_buyer_max_price = None
            if self.final_selected_buyer == 1:
                selected_buyer_max_price = self.buyer1_max_price
            elif self.final_selected_buyer == 2:
                selected_buyer_max_price = self.buyer2_max_price
            
            selected_seller_min_price = None
            if self.final_selected_seller == 1:
                selected_seller_min_price = self.seller1_min_price
            elif self.final_selected_seller == 2:
                selected_seller_min_price = self.seller2_min_price
            
            # Calculate buyer savings: buyer_max_price - deal_price (for both products)
            if selected_buyer_max_price is not None:
                buyer_savings = selected_buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
            if selected_seller_min_price is not None:
                seller_profit = deal_price - selected_seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if selected_buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if selected_seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer{self.final_selected_buyer}_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller{self.final_selected_seller}_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{self.final_selected_buyer}_max={selected_buyer_max_price}, deal_price={deal_price:.2f}, seller{self.final_selected_seller}_min={selected_seller_min_price}, round={self.current_round})")
            
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
    
    def _calculate_seller_reward(self, seller_id: int) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        
        If deal is reached with this seller:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller for both products)
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
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
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
    
    def _calculate_step_buyer_reward(self, buyer_id: int) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round with the selected seller.
        This is calculated every round, not just at the end.
        
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
        
        # Calculate buyer reward with the selected seller
        reward = 0.0
        buyer_savings = 0.0
        
        # Get buyer price from the selected seller
        buyer_price = None
        buyer_max_price = None
        if buyer_id == 1:
            buyer_max_price = self.buyer1_max_price
            if self.buyer1_selected_seller == 1:
                buyer_price = self.state_b1s1.buyer_price
            elif self.buyer1_selected_seller == 2:
                buyer_price = self.state_b1s2.buyer_price
        elif buyer_id == 2:
            buyer_max_price = self.buyer2_max_price
            if self.buyer2_selected_seller == 1:
                buyer_price = self.state_b2s1.buyer_price
            elif self.buyer2_selected_seller == 2:
                buyer_price = self.state_b2s2.buyer_price
        
        # Calculate buyer savings: buyer_max_price - buyer_price (for both products)
        if buyer_price is not None and buyer_max_price is not None:
            buyer_savings = buyer_max_price - buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self, seller_id: int) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer for both products)
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
        
        # Get seller state and min_price
        seller_state = None
        seller_min_price = None
        if seller_id == 1:
            seller_min_price = self.seller1_min_price
            # Get the most recent price from either buyer1 or buyer2
            if self.buyer1_selected_seller == 1 and self.state_b1s1.seller_price is not None:
                seller_state = self.state_b1s1
            elif self.buyer2_selected_seller == 1 and self.state_b2s1.seller_price is not None:
                seller_state = self.state_b2s1
            # If both buyers selected seller1, prefer the one with higher price
            if self.buyer1_selected_seller == 1 and self.buyer2_selected_seller == 1:
                if (self.state_b1s1.seller_price is not None and self.state_b2s1.seller_price is not None):
                    seller_state = self.state_b1s1 if self.state_b1s1.seller_price >= self.state_b2s1.seller_price else self.state_b2s1
                elif self.state_b1s1.seller_price is not None:
                    seller_state = self.state_b1s1
                elif self.state_b2s1.seller_price is not None:
                    seller_state = self.state_b2s1
        elif seller_id == 2:
            seller_min_price = self.seller2_min_price
            # Get the most recent price from either buyer1 or buyer2
            if self.buyer1_selected_seller == 2 and self.state_b1s2.seller_price is not None:
                seller_state = self.state_b1s2
            elif self.buyer2_selected_seller == 2 and self.state_b2s2.seller_price is not None:
                seller_state = self.state_b2s2
            # If both buyers selected seller2, prefer the one with higher price
            if self.buyer1_selected_seller == 2 and self.buyer2_selected_seller == 2:
                if (self.state_b1s2.seller_price is not None and self.state_b2s2.seller_price is not None):
                    seller_state = self.state_b1s2 if self.state_b1s2.seller_price >= self.state_b2s2.seller_price else self.state_b2s2
                elif self.state_b1s2.seller_price is not None:
                    seller_state = self.state_b1s2
                elif self.state_b2s2.seller_price is not None:
                    seller_state = self.state_b2s2
        
        # Calculate seller profit from current offer: seller_price - seller_min_price (for both products)
        if seller_state is not None and seller_state.seller_price is not None and seller_min_price is not None:
            seller_profit = seller_state.seller_price - seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
