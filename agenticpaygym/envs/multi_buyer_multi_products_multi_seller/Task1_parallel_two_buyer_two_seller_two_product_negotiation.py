"""Task1 Parallel Two-Buyer Two-Seller Two-Product Negotiation Environment Implementation

Supports parallel negotiation between two buyers and two sellers for two products.
Each buyer can negotiate with each seller, and deals are matched based on mutual agreement.
Prices represent total price for both products.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task1ParallelTwoBuyerTwoSellerTwoProductNegotiation(BaseEnv):
    """Task1 Parallel Two-Buyer Two-Seller Two-Product Negotiation Environment
    
    Manages parallel negotiation process between two buyers and two sellers for two products.
    Each buyer negotiates with both sellers simultaneously, and each seller negotiates with both buyers.
    Deals are matched when both buyer and seller agree on a price within tolerance.
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
        buyer_reward_aggregation: str = "average",
        seller_reward_aggregation: str = "average",
    ):
        """Initialize multi-buyer multi-seller multi-product negotiation environment
        
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
            buyer_reward_aggregation: How to aggregate buyer rewards across sellers.
                Options: "average", "max", "min" (default: "average")
            seller_reward_aggregation: How to aggregate seller rewards across buyers.
                Options: "average", "max", "min" (default: "average")
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
        
        # Set reward aggregation methods
        self.buyer_reward_aggregation = buyer_reward_aggregation
        self.seller_reward_aggregation = seller_reward_aggregation
        
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
        
        # Track which buyer-seller pair made the deal
        self.selected_buyer: Optional[int] = None  # 1 or 2
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
        self.selected_buyer = None
        self.selected_seller = None
        self.final_deal_price = None
        self.product_info = product_info or {}
        
        # Extract product information
        products = self.product_info.get("products", [])
        if len(products) < 2:
            raise ValueError("product_info must contain at least 2 products in 'products' list")
        
        # Calculate total price of both products
        total_product_price = sum(p.get("price", 0.0) for p in products)
        product_names = [p.get("name", "Product") for p in products]
        
        # Initialize Buyer1 Agent (buyer1 knows about both sellers and both products)
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 1,
            "num_sellers": 2,
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent (buyer2 knows about both sellers and both products)
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see both products
            "buyer_id": 2,
            "num_sellers": 2,
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Seller1 Agent (seller1 knows about both buyers and both products)
        seller1_context = {
            "product_info": self.product_info,  # Seller can see both products
            "initial_price": self.initial_seller1_price,  # Initial total price
            "min_price": self.seller1_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "seller_id": 1,
            "num_buyers": 2,
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent (seller2 knows about both buyers and both products)
        seller2_context = {
            "product_info": self.product_info,  # Seller can see both products
            "initial_price": self.initial_seller2_price,  # Initial total price
            "min_price": self.seller2_min_price,  # Total min price for both products
            "environment_info": self.environment_info,
            "seller_id": 2,
            "num_buyers": 2,
        }
        self.seller2_agent.initialize(seller2_context)
        
        # Sellers give initial offers to both buyers (total price for both products)
        initial_message_s1 = f"I'm offering {product_names[0]} and {product_names[1]} for a total of ${self.initial_seller1_price:.2f}."
        initial_message_s2 = f"I'm offering {product_names[0]} and {product_names[1]} for a total of ${self.initial_seller2_price:.2f}."
        
        # buyer1-seller1
        self.memory_b1s1.add_message("seller", initial_message_s1, self.current_round)
        self.state_b1s1.update(seller_price=self.initial_seller1_price)
        
        # buyer1-seller2
        self.memory_b1s2.add_message("seller", initial_message_s2, self.current_round)
        self.state_b1s2.update(seller_price=self.initial_seller2_price)
        
        # buyer2-seller1
        self.memory_b2s1.add_message("seller", initial_message_s1, self.current_round)
        self.state_b2s1.update(seller_price=self.initial_seller1_price)
        
        # buyer2-seller2
        self.memory_b2s2.add_message("seller", initial_message_s2, self.current_round)
        self.state_b2s2.update(seller_price=self.initial_seller2_price)
        
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        buyer1_action_seller1: Optional[str] = None,
        buyer1_action_seller2: Optional[str] = None,
        buyer2_action_seller1: Optional[str] = None,
        buyer2_action_seller2: Optional[str] = None,
        seller1_action_buyer1: Optional[str] = None,
        seller1_action_buyer2: Optional[str] = None,
        seller2_action_buyer1: Optional[str] = None,
        seller2_action_buyer2: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, all buyers respond to all sellers first, then all sellers respond to all buyers.
        Order: buyer -> seller
        Prices represent total price for both products.
        
        Args:
            buyer1_action_seller1: Buyer1's response to seller1 (optional)
            buyer1_action_seller2: Buyer1's response to seller2 (optional)
            buyer2_action_seller1: Buyer2's response to seller1 (optional)
            buyer2_action_seller2: Buyer2's response to seller2 (optional)
            seller1_action_buyer1: Seller1's response to buyer1 (optional)
            seller1_action_buyer2: Seller1's response to buyer2 (optional)
            seller2_action_buyer1: Seller2's response to buyer1 (optional)
            seller2_action_buyer2: Seller2's response to buyer2 (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Process buyer actions first
        # buyer1-seller1
        if buyer1_action_seller1 is not None:
            self.memory_b1s1.add_message("buyer", buyer1_action_seller1, self.current_round)
            buyer_price = self._extract_price(buyer1_action_seller1)
            if buyer_price is not None:
                self.state_b1s1.update(buyer_price=buyer_price)
        
        # buyer1-seller2
        if buyer1_action_seller2 is not None:
            self.memory_b1s2.add_message("buyer", buyer1_action_seller2, self.current_round)
            buyer_price = self._extract_price(buyer1_action_seller2)
            if buyer_price is not None:
                self.state_b1s2.update(buyer_price=buyer_price)
        
        # buyer2-seller1
        if buyer2_action_seller1 is not None:
            self.memory_b2s1.add_message("buyer", buyer2_action_seller1, self.current_round)
            buyer_price = self._extract_price(buyer2_action_seller1)
            if buyer_price is not None:
                self.state_b2s1.update(buyer_price=buyer_price)
        
        # buyer2-seller2
        if buyer2_action_seller2 is not None:
            self.memory_b2s2.add_message("buyer", buyer2_action_seller2, self.current_round)
            buyer_price = self._extract_price(buyer2_action_seller2)
            if buyer_price is not None:
                self.state_b2s2.update(buyer_price=buyer_price)
        
        # Process seller actions after buyers
        # seller1-buyer1
        if seller1_action_buyer1 is not None:
            self.memory_b1s1.add_message("seller", seller1_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer1)
            if seller_price is not None:
                self.state_b1s1.update(seller_price=seller_price)
        
        # seller1-buyer2
        if seller1_action_buyer2 is not None:
            self.memory_b2s1.add_message("seller", seller1_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer2)
            if seller_price is not None:
                self.state_b2s1.update(seller_price=seller_price)
        
        # seller2-buyer1
        if seller2_action_buyer1 is not None:
            self.memory_b1s2.add_message("seller", seller2_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer1)
            if seller_price is not None:
                self.state_b1s2.update(seller_price=seller_price)
        
        # seller2-buyer2
        if seller2_action_buyer2 is not None:
            self.memory_b2s2.add_message("seller", seller2_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer2)
            if seller_price is not None:
                self.state_b2s2.update(seller_price=seller_price)
        
        # Check for deals: each buyer-seller pair can make a deal
        # A deal is made when both buyer and seller want to make a deal and prices are within tolerance
        deals = []  # List of (buyer_id, seller_id, price) tuples
        
        # Check buyer1-seller1
        if (buyer1_action_seller1 is not None and seller1_action_buyer1 is not None and
            self._check_make_deal(buyer1_action_seller1) and
            self._check_make_deal(seller1_action_buyer1) and
            self.state_b1s1.buyer_price is not None and
            self.state_b1s1.seller_price is not None):
            price_diff = abs(self.state_b1s1.buyer_price - self.state_b1s1.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (self.state_b1s1.buyer_price + self.state_b1s1.seller_price) / 2
                deals.append((1, 1, deal_price))
        
        # Check buyer1-seller2
        if (buyer1_action_seller2 is not None and seller2_action_buyer1 is not None and
            self._check_make_deal(buyer1_action_seller2) and
            self._check_make_deal(seller2_action_buyer1) and
            self.state_b1s2.buyer_price is not None and
            self.state_b1s2.seller_price is not None):
            price_diff = abs(self.state_b1s2.buyer_price - self.state_b1s2.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (self.state_b1s2.buyer_price + self.state_b1s2.seller_price) / 2
                deals.append((1, 2, deal_price))
        
        # Check buyer2-seller1
        if (buyer2_action_seller1 is not None and seller1_action_buyer2 is not None and
            self._check_make_deal(buyer2_action_seller1) and
            self._check_make_deal(seller1_action_buyer2) and
            self.state_b2s1.buyer_price is not None and
            self.state_b2s1.seller_price is not None):
            price_diff = abs(self.state_b2s1.buyer_price - self.state_b2s1.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (self.state_b2s1.buyer_price + self.state_b2s1.seller_price) / 2
                deals.append((2, 1, deal_price))
        
        # Check buyer2-seller2
        if (buyer2_action_seller2 is not None and seller2_action_buyer2 is not None and
            self._check_make_deal(buyer2_action_seller2) and
            self._check_make_deal(seller2_action_buyer2) and
            self.state_b2s2.buyer_price is not None and
            self.state_b2s2.seller_price is not None):
            price_diff = abs(self.state_b2s2.buyer_price - self.state_b2s2.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (self.state_b2s2.buyer_price + self.state_b2s2.seller_price) / 2
                deals.append((2, 2, deal_price))
        
        # Select the best deal: prioritize buyer's preference (lower price) and seller's preference (higher price)
        # If multiple deals exist, choose the one with the best price for both parties
        if deals:
            # For buyers: prefer lower price, for sellers: prefer higher price
            # We'll choose the deal that maximizes overall utility (buyer savings + seller profit)
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
                self.selected_buyer, self.selected_seller, self.final_deal_price = best_deal
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer1_reward = 0.0
        buyer2_reward = 0.0
        seller1_reward = 0.0
        seller2_reward = 0.0
        
        if self.selected_buyer is not None and self.selected_seller is not None and self.final_deal_price is not None:
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
        step_buyer1_reward = self._calculate_step_buyer_reward(1)
        step_buyer2_reward = self._calculate_step_buyer_reward(2)
        step_seller1_reward = self._calculate_step_seller_reward(1)
        step_seller2_reward = self._calculate_step_seller_reward(2)
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_buyer1_reward"] = step_buyer1_reward
        info["step_buyer2_reward"] = step_buyer2_reward
        info["step_seller1_reward"] = step_seller1_reward
        info["step_seller2_reward"] = step_seller2_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.selected_buyer
                info["selected_seller"] = self.selected_seller
                info["final_deal_price"] = self.final_deal_price
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays all buyer-seller conversations for each round, followed by a round summary.
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
        output_lines.append(f"Round {self.current_round} - Parallel Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display Buyer1-Seller1 conversation
        output_lines.append(f"\n[BUYER 1 - SELLER 1 Conversation]:")
        history_b1s1 = self.memory_b1s1.get_history()
        if history_b1s1:
            current_round_messages = [
                msg for msg in history_b1s1 if msg["round"] == self.current_round
            ]
            for msg in current_round_messages:
                role = msg["role"].upper()
                output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer1-Seller2 conversation
        output_lines.append(f"\n[BUYER 1 - SELLER 2 Conversation]:")
        history_b1s2 = self.memory_b1s2.get_history()
        if history_b1s2:
            current_round_messages = [
                msg for msg in history_b1s2 if msg["round"] == self.current_round
            ]
            for msg in current_round_messages:
                role = msg["role"].upper()
                output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer2-Seller1 conversation
        output_lines.append(f"\n[BUYER 2 - SELLER 1 Conversation]:")
        history_b2s1 = self.memory_b2s1.get_history()
        if history_b2s1:
            current_round_messages = [
                msg for msg in history_b2s1 if msg["round"] == self.current_round
            ]
            for msg in current_round_messages:
                role = msg["role"].upper()
                output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Display Buyer2-Seller2 conversation
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
        
        # Display prices for each pair (total for both products)
        output_lines.append(f"\nBuyer1-Seller1:")
        if self.state_b1s1.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b1s1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b1s1.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b1s1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        output_lines.append(f"\nBuyer1-Seller2:")
        if self.state_b1s2.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b1s2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b1s2.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b1s2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        output_lines.append(f"\nBuyer2-Seller1:")
        if self.state_b2s1.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b2s1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b2s1.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b2s1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        output_lines.append(f"\nBuyer2-Seller2:")
        if self.state_b2s2.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b2s2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b2s2.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b2s2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display deal status
        if self.selected_buyer is not None and self.selected_seller is not None:
            output_lines.append(f"\n  ✓ DEAL MADE: Buyer {self.selected_buyer} - Seller {self.selected_seller}")
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
            "b1s1_buyer_price": self.state_b1s1.buyer_price,  # Total price for both products
            "b1s1_seller_price": self.state_b1s1.seller_price,  # Total price for both products
            "b1s2_buyer_price": self.state_b1s2.buyer_price,  # Total price for both products
            "b1s2_seller_price": self.state_b1s2.seller_price,  # Total price for both products
            "b2s1_buyer_price": self.state_b2s1.buyer_price,  # Total price for both products
            "b2s1_seller_price": self.state_b2s1.seller_price,  # Total price for both products
            "b2s2_buyer_price": self.state_b2s2.buyer_price,  # Total price for both products
            "b2s2_seller_price": self.state_b2s2.seller_price,  # Total price for both products
            "status": self.negotiation_info.status.value,
            "selected_buyer": self.selected_buyer,
            "selected_seller": self.selected_seller,
            "final_deal_price": self.final_deal_price,
            "product_info": self.product_info,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,  # Total price for both products
            "b1s1_seller_price": self.state_b1s1.seller_price,  # Total price for both products
            "b1s2_buyer_price": self.state_b1s2.buyer_price,  # Total price for both products
            "b1s2_seller_price": self.state_b1s2.seller_price,  # Total price for both products
            "b2s1_buyer_price": self.state_b2s1.buyer_price,  # Total price for both products
            "b2s1_seller_price": self.state_b2s1.seller_price,  # Total price for both products
            "b2s2_buyer_price": self.state_b2s2.buyer_price,  # Total price for both products
            "b2s2_seller_price": self.state_b2s2.seller_price,  # Total price for both products
            "selected_buyer": self.selected_buyer,
            "selected_seller": self.selected_seller,
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
        patterns = [
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*dollars?',
            r'(\d+\.?\d*)\s*USD',
            r'price.*?(\d+\.?\d*)',
            r'offer.*?(\d+\.?\d*)',
            r'total.*?(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    price = float(matches[-1])
                    if price > 0:
                        return price
                except ValueError:
                    continue
        
        return None
    
    def _check_make_deal(self, text: str) -> bool:
        """Check if agent wants to make a deal
        
        Args:
            text: Agent's response text
            
        Returns:
            Whether agent wants to make a deal
        """
        if 'MAKE_DEAL' in text.upper():
            return True
        
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
    
    def _calculate_reward(self) -> float:
        """Calculate global reward
        
        Returns:
            Reward value
        """
        time_cost = -self.current_round
        
        if (self.negotiation_info.status == NegotiationStatus.AGREED and
            self.selected_buyer is not None and
            self.selected_seller is not None and
            self.final_deal_price is not None):
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            buyer_max_price = self.buyer1_max_price if self.selected_buyer == 1 else self.buyer2_max_price
            seller_min_price = self.seller1_min_price if self.selected_seller == 1 else self.seller2_min_price
            
            if buyer_max_price is not None:
                buyer_savings = buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            if seller_min_price is not None:
                seller_profit = deal_price - seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            reward += time_cost * self.reward_weights["time_cost"]
            
            print(f"Global Reward = buyer{self.selected_buyer}_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller{self.selected_seller}_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{self.selected_buyer}_max={buyer_max_price}, deal_price={deal_price:.2f}, seller{self.selected_seller}_min={seller_min_price}, round={self.current_round})")
            
            return reward
        else:
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self, buyer_id: int) -> float:
        """Calculate reward from buyer's perspective
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Reward value from buyer's perspective
        """
        time_cost = -self.current_round
        
        deal_reached_with_this_buyer = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.selected_buyer == buyer_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_buyer:
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            buyer_max_price = self.buyer1_max_price if buyer_id == 1 else self.buyer2_max_price
            
            if buyer_max_price is not None:
                buyer_savings = buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            reward += time_cost * self.reward_weights["time_cost"]
            
            print(f"Buyer{buyer_id} Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{buyer_id}_max={buyer_max_price}, deal_price={deal_price:.2f}, round={self.current_round})")
            
            return reward
        else:
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this buyer)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self, seller_id: int) -> float:
        """Calculate reward from seller's perspective
        
        Args:
            seller_id: Seller ID (1 or 2)
        
        Returns:
            Reward value from seller's perspective
        """
        time_cost = -self.current_round
        
        deal_reached_with_this_seller = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.selected_seller == seller_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_seller:
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            seller_min_price = self.seller1_min_price if seller_id == 1 else self.seller2_min_price
            
            if seller_min_price is not None:
                seller_profit = deal_price - seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            reward += time_cost * self.reward_weights["time_cost"]
            
            print(f"Seller{seller_id} Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller{seller_id}_min={seller_min_price}, round={self.current_round})")
            
            return reward
        else:
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller{seller_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this seller)")
            return weighted_time_cost
    
    def _calculate_step_buyer_reward(self, buyer_id: int) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        round_cost = -self.current_round
        
        buyer_rewards = []
        buyer_max_price = self.buyer1_max_price if buyer_id == 1 else self.buyer2_max_price
        
        # Buyer reward with seller1
        if buyer_id == 1:
            state = self.state_b1s1
        else:
            state = self.state_b2s1
        
        if state.buyer_price is not None and buyer_max_price is not None:
            buyer_savings = buyer_max_price - state.buyer_price
            reward = buyer_savings * self.reward_weights["buyer_savings"]
            buyer_rewards.append(reward)
        
        # Buyer reward with seller2
        if buyer_id == 1:
            state = self.state_b1s2
        else:
            state = self.state_b2s2
        
        if state.buyer_price is not None and buyer_max_price is not None:
            buyer_savings = buyer_max_price - state.buyer_price
            reward = buyer_savings * self.reward_weights["buyer_savings"]
            buyer_rewards.append(reward)
        
        if buyer_rewards:
            aggregated_reward = self._aggregate_rewards(buyer_rewards, self.buyer_reward_aggregation)
        else:
            aggregated_reward = 0.0
        
        reward = aggregated_reward + round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self, seller_id: int) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Args:
            seller_id: Seller ID (1 or 2)
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        round_cost = -self.current_round
        
        seller_rewards = []
        seller_min_price = self.seller1_min_price if seller_id == 1 else self.seller2_min_price
        
        # Seller reward with buyer1
        if seller_id == 1:
            state = self.state_b1s1
        else:
            state = self.state_b1s2
        
        if state.seller_price is not None and seller_min_price is not None:
            seller_profit = state.seller_price - seller_min_price
            reward = seller_profit * self.reward_weights["seller_profit"]
            seller_rewards.append(reward)
        
        # Seller reward with buyer2
        if seller_id == 1:
            state = self.state_b2s1
        else:
            state = self.state_b2s2
        
        if state.seller_price is not None and seller_min_price is not None:
            seller_profit = state.seller_price - seller_min_price
            reward = seller_profit * self.reward_weights["seller_profit"]
            seller_rewards.append(reward)
        
        if seller_rewards:
            aggregated_reward = self._aggregate_rewards(seller_rewards, self.seller_reward_aggregation)
        else:
            aggregated_reward = 0.0
        
        reward = aggregated_reward + round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _aggregate_rewards(self, rewards: list, method: str) -> float:
        """Aggregate multiple rewards using specified method
        
        Args:
            rewards: List of reward values to aggregate
            method: Aggregation method - "average", "max", or "min"
        
        Returns:
            Aggregated reward value
        """
        if not rewards:
            return 0.0
        
        if method == "average":
            return sum(rewards) / len(rewards)
        elif method == "max":
            return max(rewards)
        elif method == "min":
            return min(rewards)
        else:
            return sum(rewards) / len(rewards)

