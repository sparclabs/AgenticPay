"""Task4 Sequential Three-Buyer Three-Seller Three-Product Negotiation Environment Implementation

Supports sequential negotiation where three buyers each choose one seller per round to negotiate with
for three products. Each buyer can switch between three sellers and make a deal with any seller.
Prices represent total price for all three products.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task4SequentialThreeBuyerThreeSellerThreeProductNegotiation(BaseEnv):
    """Task4 Sequential Three-Buyer Three-Seller Three-Product Negotiation Environment
    
    Manages sequential negotiation process where each buyer chooses one seller per round to negotiate with
    for three products. Each buyer can switch between three sellers and make a deal with any seller.
    Prices represent total price for all three products.
    """
    
    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        buyer3_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        seller3_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_price: float = 300.0,
        initial_seller2_price: float = 320.0,
        initial_seller3_price: float = 340.0,
        buyer1_max_price: Optional[float] = None,
        buyer2_max_price: Optional[float] = None,
        buyer3_max_price: Optional[float] = None,
        seller1_min_price: Optional[float] = None,
        seller2_min_price: Optional[float] = None,
        seller3_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize sequential multi-buyer multi-seller multi-product negotiation environment
        
        Args:
            buyer1_agent: First Buyer Agent
            buyer2_agent: Second Buyer Agent
            buyer3_agent: Third Buyer Agent
            seller1_agent: First Seller Agent
            seller2_agent: Second Seller Agent
            seller3_agent: Third Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller1_price: Initial total price offered by seller1 for all three products
            initial_seller2_price: Initial total price offered by seller2 for all three products
            initial_seller3_price: Initial total price offered by seller3 for all three products
            buyer1_max_price: Maximum acceptable total price for buyer1 (confidential, for all three products)
            buyer2_max_price: Maximum acceptable total price for buyer2 (confidential, for all three products)
            buyer3_max_price: Maximum acceptable total price for buyer3 (confidential, for all three products)
            seller1_min_price: Minimum acceptable total price for seller1 (confidential, for all three products)
            seller2_min_price: Minimum acceptable total price for seller2 (confidential, for all three products)
            seller3_min_price: Minimum acceptable total price for seller3 (confidential, for all three products)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
            reward_weights: Reward weights configuration dict with keys:
                - buyer_savings: weight for buyer savings (default: 1.0)
                - seller_profit: weight for seller profit (default: 1.0)
                - time_cost: weight for time cost (default: 0.1)
        """
        self.buyer1_agent = buyer1_agent
        self.buyer2_agent = buyer2_agent
        self.buyer3_agent = buyer3_agent
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.seller3_agent = seller3_agent
        self.max_rounds = max_rounds
        self.initial_seller1_price = initial_seller1_price
        self.initial_seller2_price = initial_seller2_price
        self.initial_seller3_price = initial_seller3_price
        self.buyer1_max_price = buyer1_max_price
        self.buyer2_max_price = buyer2_max_price
        self.buyer3_max_price = buyer3_max_price
        self.seller1_min_price = seller1_min_price
        self.seller2_min_price = seller2_min_price
        self.seller3_min_price = seller3_min_price
        self.environment_info = environment_info or {}
        self.price_tolerance = price_tolerance
        
        # Set default reward weights
        default_weights = {
            "buyer_savings": 1.0,      # 买方节省权重
            "seller_profit": 1.0,      # 卖方利润权重
            "time_cost": 0.1,          # 时间成本权重（降低影响）
        }
        if reward_weights is not None:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights
        
        # Call parent class initialization
        super().__init__()
        
        # State management - separate for each buyer-seller pair
        # buyer1-seller pairs
        self.memory_b1s1 = ConversationMemory()
        self.state_b1s1 = NegotiationState()
        self.memory_b1s2 = ConversationMemory()
        self.state_b1s2 = NegotiationState()
        self.memory_b1s3 = ConversationMemory()
        self.state_b1s3 = NegotiationState()
        # buyer2-seller pairs
        self.memory_b2s1 = ConversationMemory()
        self.state_b2s1 = NegotiationState()
        self.memory_b2s2 = ConversationMemory()
        self.state_b2s2 = NegotiationState()
        self.memory_b2s3 = ConversationMemory()
        self.state_b2s3 = NegotiationState()
        # buyer3-seller pairs
        self.memory_b3s1 = ConversationMemory()
        self.state_b3s1 = NegotiationState()
        self.memory_b3s2 = ConversationMemory()
        self.state_b3s2 = NegotiationState()
        self.memory_b3s3 = ConversationMemory()
        self.state_b3s3 = NegotiationState()
        
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info: Optional[Dict[str, Any]] = None
        
        # Track which seller each buyer selected for current round and final deal
        self.buyer1_selected_seller: Optional[int] = None  # 1, 2, or 3
        self.buyer2_selected_seller: Optional[int] = None  # 1, 2, or 3
        self.buyer3_selected_seller: Optional[int] = None  # 1, 2, or 3
        self.final_selected_buyer: Optional[int] = None  # 1, 2, or 3
        self.final_selected_seller: Optional[int] = None  # 1, 2, or 3
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
            user_requirement: User requirement description (should describe purchasing three products)
            product_info: Product information containing three products and their prices
                Expected format: {
                    "products": [
                        {"name": "Product1", "price": 100.0, ...},
                        {"name": "Product2", "price": 80.0, ...},
                        {"name": "Product3", "price": 70.0, ...}
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
        self.memory_b1s3.clear()
        self.memory_b2s1.clear()
        self.memory_b2s2.clear()
        self.memory_b2s3.clear()
        self.memory_b3s1.clear()
        self.memory_b3s2.clear()
        self.memory_b3s3.clear()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b1s3 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
        self.state_b2s3 = NegotiationState()
        self.state_b3s1 = NegotiationState()
        self.state_b3s2 = NegotiationState()
        self.state_b3s3 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.buyer1_selected_seller = None
        self.buyer2_selected_seller = None
        self.buyer3_selected_seller = None
        self.final_selected_buyer = None
        self.final_selected_seller = None
        self.final_deal_price = None
        self.product_info = product_info or {}
        
        # Extract product information
        products = self.product_info.get("products", [])
        if len(products) < 3:
            raise ValueError("product_info must contain at least 3 products in 'products' list")
        
        # Calculate total price of all three products
        total_product_price = sum(p.get("price", 0.0) for p in products)
        
        # Initialize Buyer1 Agent
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,  # Total max price for all three products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,
            "buyer_id": 1,
            "num_sellers": 3,
            "negotiation_mode": "sequential",
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,
            "buyer_id": 2,
            "num_sellers": 3,
            "negotiation_mode": "sequential",
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Buyer3 Agent
        buyer3_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer3_max_price,
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,
            "buyer_id": 3,
            "num_sellers": 3,
            "negotiation_mode": "sequential",
        }
        self.buyer3_agent.initialize(buyer3_context)
        
        # Initialize Seller1 Agent
        seller1_context = {
            "product_info": self.product_info,
            "initial_price": self.initial_seller1_price,
            "min_price": self.seller1_min_price,
            "environment_info": self.environment_info,
            "seller_id": 1,
            "num_buyers": 3,
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent
        seller2_context = {
            "product_info": self.product_info,
            "initial_price": self.initial_seller2_price,
            "min_price": self.seller2_min_price,
            "environment_info": self.environment_info,
            "seller_id": 2,
            "num_buyers": 3,
        }
        self.seller2_agent.initialize(seller2_context)
        
        # Initialize Seller3 Agent
        seller3_context = {
            "product_info": self.product_info,
            "initial_price": self.initial_seller3_price,
            "min_price": self.seller3_min_price,
            "environment_info": self.environment_info,
            "seller_id": 3,
            "num_buyers": 3,
        }
        self.seller3_agent.initialize(seller3_context)
        
        # Sellers give initial offers to all buyers (total price for all three products)
        product_names = [p.get("name", "Product") for p in products]
        initial_message_seller1 = f"I'm offering {product_names[0]}, {product_names[1]}, and {product_names[2]} for a total of ${self.initial_seller1_price:.2f}."
        initial_message_seller2 = f"I'm offering {product_names[0]}, {product_names[1]}, and {product_names[2]} for a total of ${self.initial_seller2_price:.2f}."
        initial_message_seller3 = f"I'm offering {product_names[0]}, {product_names[1]}, and {product_names[2]} for a total of ${self.initial_seller3_price:.2f}."
        
        # buyer1-seller pairs
        self.memory_b1s1.add_message("seller", initial_message_seller1, self.current_round)
        self.state_b1s1.update(seller_price=self.initial_seller1_price)
        self.memory_b1s2.add_message("seller", initial_message_seller2, self.current_round)
        self.state_b1s2.update(seller_price=self.initial_seller2_price)
        self.memory_b1s3.add_message("seller", initial_message_seller3, self.current_round)
        self.state_b1s3.update(seller_price=self.initial_seller3_price)
        
        # buyer2-seller pairs
        self.memory_b2s1.add_message("seller", initial_message_seller1, self.current_round)
        self.state_b2s1.update(seller_price=self.initial_seller1_price)
        self.memory_b2s2.add_message("seller", initial_message_seller2, self.current_round)
        self.state_b2s2.update(seller_price=self.initial_seller2_price)
        self.memory_b2s3.add_message("seller", initial_message_seller3, self.current_round)
        self.state_b2s3.update(seller_price=self.initial_seller3_price)
        
        # buyer3-seller pairs
        self.memory_b3s1.add_message("seller", initial_message_seller1, self.current_round)
        self.state_b3s1.update(seller_price=self.initial_seller1_price)
        self.memory_b3s2.add_message("seller", initial_message_seller2, self.current_round)
        self.state_b3s2.update(seller_price=self.initial_seller2_price)
        self.memory_b3s3.add_message("seller", initial_message_seller3, self.current_round)
        self.state_b3s3.update(seller_price=self.initial_seller3_price)
        
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        buyer1_selected_seller: int,  # 1, 2, or 3
        buyer2_selected_seller: int,  # 1, 2, or 3
        buyer3_selected_seller: int,  # 1, 2, or 3
        buyer1_action: Optional[str] = None,
        buyer2_action: Optional[str] = None,
        buyer3_action: Optional[str] = None,
        seller1_action_buyer1: Optional[str] = None,
        seller1_action_buyer2: Optional[str] = None,
        seller1_action_buyer3: Optional[str] = None,
        seller2_action_buyer1: Optional[str] = None,
        seller2_action_buyer2: Optional[str] = None,
        seller2_action_buyer3: Optional[str] = None,
        seller3_action_buyer1: Optional[str] = None,
        seller3_action_buyer2: Optional[str] = None,
        seller3_action_buyer3: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, each buyer chooses one seller to negotiate with, then buyers and sellers exchange messages.
        Order: buyer -> seller
        Prices represent total price for all three products.
        
        Args:
            buyer1_selected_seller: Which seller (1, 2, or 3) buyer1 chooses to negotiate with this round
            buyer2_selected_seller: Which seller (1, 2, or 3) buyer2 chooses to negotiate with this round
            buyer3_selected_seller: Which seller (1, 2, or 3) buyer3 chooses to negotiate with this round
            buyer1_action: Buyer1's response (optional)
            buyer2_action: Buyer2's response (optional)
            buyer3_action: Buyer3's response (optional)
            seller1_action_buyer1: Seller1's response to buyer1 (optional, only if buyer1 selected seller1)
            seller1_action_buyer2: Seller1's response to buyer2 (optional, only if buyer2 selected seller1)
            seller1_action_buyer3: Seller1's response to buyer3 (optional, only if buyer3 selected seller1)
            seller2_action_buyer1: Seller2's response to buyer1 (optional, only if buyer1 selected seller2)
            seller2_action_buyer2: Seller2's response to buyer2 (optional, only if buyer2 selected seller2)
            seller2_action_buyer3: Seller2's response to buyer3 (optional, only if buyer3 selected seller2)
            seller3_action_buyer1: Seller3's response to buyer1 (optional, only if buyer1 selected seller3)
            seller3_action_buyer2: Seller3's response to buyer2 (optional, only if buyer2 selected seller3)
            seller3_action_buyer3: Seller3's response to buyer3 (optional, only if buyer3 selected seller3)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if buyer1_selected_seller not in [1, 2, 3]:
            raise ValueError(f"buyer1_selected_seller must be 1, 2, or 3, got {buyer1_selected_seller}")
        if buyer2_selected_seller not in [1, 2, 3]:
            raise ValueError(f"buyer2_selected_seller must be 1, 2, or 3, got {buyer2_selected_seller}")
        if buyer3_selected_seller not in [1, 2, 3]:
            raise ValueError(f"buyer3_selected_seller must be 1, 2, or 3, got {buyer3_selected_seller}")
        
        self.buyer1_selected_seller = buyer1_selected_seller
        self.buyer2_selected_seller = buyer2_selected_seller
        self.buyer3_selected_seller = buyer3_selected_seller
        
        # Process buyer actions first
        # buyer1 action
        if buyer1_action is not None:
            if buyer1_selected_seller == 1:
                self.memory_b1s1.add_message("buyer", buyer1_action, self.current_round)
                buyer_price = self._extract_price(buyer1_action)
                if buyer_price is not None:
                    self.state_b1s1.update(buyer_price=buyer_price)
            elif buyer1_selected_seller == 2:
                self.memory_b1s2.add_message("buyer", buyer1_action, self.current_round)
                buyer_price = self._extract_price(buyer1_action)
                if buyer_price is not None:
                    self.state_b1s2.update(buyer_price=buyer_price)
            else:  # buyer1_selected_seller == 3
                self.memory_b1s3.add_message("buyer", buyer1_action, self.current_round)
                buyer_price = self._extract_price(buyer1_action)
                if buyer_price is not None:
                    self.state_b1s3.update(buyer_price=buyer_price)
        
        # buyer2 action
        if buyer2_action is not None:
            if buyer2_selected_seller == 1:
                self.memory_b2s1.add_message("buyer", buyer2_action, self.current_round)
                buyer_price = self._extract_price(buyer2_action)
                if buyer_price is not None:
                    self.state_b2s1.update(buyer_price=buyer_price)
            elif buyer2_selected_seller == 2:
                self.memory_b2s2.add_message("buyer", buyer2_action, self.current_round)
                buyer_price = self._extract_price(buyer2_action)
                if buyer_price is not None:
                    self.state_b2s2.update(buyer_price=buyer_price)
            else:  # buyer2_selected_seller == 3
                self.memory_b2s3.add_message("buyer", buyer2_action, self.current_round)
                buyer_price = self._extract_price(buyer2_action)
                if buyer_price is not None:
                    self.state_b2s3.update(buyer_price=buyer_price)
        
        # buyer3 action
        if buyer3_action is not None:
            if buyer3_selected_seller == 1:
                self.memory_b3s1.add_message("buyer", buyer3_action, self.current_round)
                buyer_price = self._extract_price(buyer3_action)
                if buyer_price is not None:
                    self.state_b3s1.update(buyer_price=buyer_price)
            elif buyer3_selected_seller == 2:
                self.memory_b3s2.add_message("buyer", buyer3_action, self.current_round)
                buyer_price = self._extract_price(buyer3_action)
                if buyer_price is not None:
                    self.state_b3s2.update(buyer_price=buyer_price)
            else:  # buyer3_selected_seller == 3
                self.memory_b3s3.add_message("buyer", buyer3_action, self.current_round)
                buyer_price = self._extract_price(buyer3_action)
                if buyer_price is not None:
                    self.state_b3s3.update(buyer_price=buyer_price)
        
        # Process seller actions after buyers
        # seller1 actions
        if buyer1_selected_seller == 1 and seller1_action_buyer1 is not None:
            self.memory_b1s1.add_message("seller", seller1_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer1)
            if seller_price is not None:
                self.state_b1s1.update(seller_price=seller_price)
        if buyer2_selected_seller == 1 and seller1_action_buyer2 is not None:
            self.memory_b2s1.add_message("seller", seller1_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer2)
            if seller_price is not None:
                self.state_b2s1.update(seller_price=seller_price)
        if buyer3_selected_seller == 1 and seller1_action_buyer3 is not None:
            self.memory_b3s1.add_message("seller", seller1_action_buyer3, self.current_round)
            seller_price = self._extract_price(seller1_action_buyer3)
            if seller_price is not None:
                self.state_b3s1.update(seller_price=seller_price)
        
        # seller2 actions
        if buyer1_selected_seller == 2 and seller2_action_buyer1 is not None:
            self.memory_b1s2.add_message("seller", seller2_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer1)
            if seller_price is not None:
                self.state_b1s2.update(seller_price=seller_price)
        if buyer2_selected_seller == 2 and seller2_action_buyer2 is not None:
            self.memory_b2s2.add_message("seller", seller2_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer2)
            if seller_price is not None:
                self.state_b2s2.update(seller_price=seller_price)
        if buyer3_selected_seller == 2 and seller2_action_buyer3 is not None:
            self.memory_b3s2.add_message("seller", seller2_action_buyer3, self.current_round)
            seller_price = self._extract_price(seller2_action_buyer3)
            if seller_price is not None:
                self.state_b3s2.update(seller_price=seller_price)
        
        # seller3 actions
        if buyer1_selected_seller == 3 and seller3_action_buyer1 is not None:
            self.memory_b1s3.add_message("seller", seller3_action_buyer1, self.current_round)
            seller_price = self._extract_price(seller3_action_buyer1)
            if seller_price is not None:
                self.state_b1s3.update(seller_price=seller_price)
        if buyer2_selected_seller == 3 and seller3_action_buyer2 is not None:
            self.memory_b2s3.add_message("seller", seller3_action_buyer2, self.current_round)
            seller_price = self._extract_price(seller3_action_buyer2)
            if seller_price is not None:
                self.state_b2s3.update(seller_price=seller_price)
        if buyer3_selected_seller == 3 and seller3_action_buyer3 is not None:
            self.memory_b3s3.add_message("seller", seller3_action_buyer3, self.current_round)
            seller_price = self._extract_price(seller3_action_buyer3)
            if seller_price is not None:
                self.state_b3s3.update(seller_price=seller_price)
        
        # Check if deal can be made with the selected sellers
        deals = []  # List of (buyer_id, seller_id, price) tuples
        
        # Check all buyer-seller pairs
        for buyer_id in [1, 2, 3]:
            for seller_id in [1, 2, 3]:
                selected_seller = None
                buyer_action = None
                state = None
                
                if buyer_id == 1:
                    selected_seller = self.buyer1_selected_seller
                    buyer_action = buyer1_action
                    if seller_id == 1:
                        state = self.state_b1s1
                    elif seller_id == 2:
                        state = self.state_b1s2
                    else:
                        state = self.state_b1s3
                elif buyer_id == 2:
                    selected_seller = self.buyer2_selected_seller
                    buyer_action = buyer2_action
                    if seller_id == 1:
                        state = self.state_b2s1
                    elif seller_id == 2:
                        state = self.state_b2s2
                    else:
                        state = self.state_b2s3
                else:  # buyer_id == 3
                    selected_seller = self.buyer3_selected_seller
                    buyer_action = buyer3_action
                    if seller_id == 1:
                        state = self.state_b3s1
                    elif seller_id == 2:
                        state = self.state_b3s2
                    else:
                        state = self.state_b3s3
                
                if (selected_seller == seller_id and
                    buyer_action is not None and 
                    self._check_make_deal(buyer_action) and
                    state.buyer_price is not None and 
                    state.seller_price is not None):
                    price_diff = abs(state.buyer_price - state.seller_price)
                    if price_diff <= self.price_tolerance:
                        deal_price = (state.buyer_price + state.seller_price) / 2
                        deals.append((buyer_id, seller_id, deal_price))
        
        # Select the best deal
        if deals:
            best_deal = None
            best_utility = float('-inf')
            
            for buyer_id, seller_id, price in deals:
                if buyer_id == 1:
                    buyer_max = self.buyer1_max_price
                elif buyer_id == 2:
                    buyer_max = self.buyer2_max_price
                else:
                    buyer_max = self.buyer3_max_price
                
                if seller_id == 1:
                    seller_min = self.seller1_min_price
                elif seller_id == 2:
                    seller_min = self.seller2_min_price
                else:
                    seller_min = self.seller3_min_price
                
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
        buyer3_reward = 0.0
        seller1_reward = 0.0
        seller2_reward = 0.0
        seller3_reward = 0.0
        
        if self.final_selected_buyer is not None and self.final_selected_seller is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            buyer3_reward = self._calculate_buyer_reward(3)
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
            seller3_reward = self._calculate_seller_reward(3)
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            buyer3_reward = self._calculate_buyer_reward(3)
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
            seller3_reward = self._calculate_seller_reward(3)
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        step_buyer1_reward = self._calculate_step_buyer_reward(1)
        step_buyer2_reward = self._calculate_step_buyer_reward(2)
        step_buyer3_reward = self._calculate_step_buyer_reward(3)
        step_seller1_reward = self._calculate_step_seller_reward(1) if (buyer1_selected_seller == 1 or buyer2_selected_seller == 1 or buyer3_selected_seller == 1) else None
        step_seller2_reward = self._calculate_step_seller_reward(2) if (buyer1_selected_seller == 2 or buyer2_selected_seller == 2 or buyer3_selected_seller == 2) else None
        step_seller3_reward = self._calculate_step_seller_reward(3) if (buyer1_selected_seller == 3 or buyer2_selected_seller == 3 or buyer3_selected_seller == 3) else None
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_buyer1_reward"] = step_buyer1_reward
        info["step_buyer2_reward"] = step_buyer2_reward
        info["step_buyer3_reward"] = step_buyer3_reward
        if step_seller1_reward is not None:
            info["step_seller1_reward"] = step_seller1_reward
        if step_seller2_reward is not None:
            info["step_seller2_reward"] = step_seller2_reward
        if step_seller3_reward is not None:
            info["step_seller3_reward"] = step_seller3_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.final_selected_buyer
                info["selected_seller"] = self.final_selected_seller
                info["final_deal_price"] = self.final_deal_price
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["buyer3_reward"] = buyer3_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
            info["seller3_reward"] = seller3_reward
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays buyer and seller outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        Prices shown are total prices for all three products.
        
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
        if self.buyer3_selected_seller is not None:
            output_lines.append(f"[Buyer 3 Selected Seller: Seller {self.buyer3_selected_seller}]")
        
        # Display conversations for all buyer-seller pairs (only for selected sellers this round)
        buyer_seller_pairs = [
            (1, 1, self.buyer1_selected_seller, self.memory_b1s1),
            (1, 2, self.buyer1_selected_seller, self.memory_b1s2),
            (1, 3, self.buyer1_selected_seller, self.memory_b1s3),
            (2, 1, self.buyer2_selected_seller, self.memory_b2s1),
            (2, 2, self.buyer2_selected_seller, self.memory_b2s2),
            (2, 3, self.buyer2_selected_seller, self.memory_b2s3),
            (3, 1, self.buyer3_selected_seller, self.memory_b3s1),
            (3, 2, self.buyer3_selected_seller, self.memory_b3s2),
            (3, 3, self.buyer3_selected_seller, self.memory_b3s3),
        ]
        
        for buyer_id, seller_id, selected_seller, memory in buyer_seller_pairs:
            if selected_seller == seller_id:
                output_lines.append(f"\n[BUYER {buyer_id} - SELLER {seller_id} Conversation]:")
                history = memory.get_history()
                if history:
                    current_round_messages = [
                        msg for msg in history if msg["round"] == self.current_round
                    ]
                    for msg in current_round_messages:
                        role = msg["role"].upper()
                        output_lines.append(f"  [{role}]: {msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display prices for all buyer-seller pairs (total for all three products)
        for buyer_id in [1, 2, 3]:
            for seller_id in [1, 2, 3]:
                if buyer_id == 1:
                    if seller_id == 1:
                        state = self.state_b1s1
                    elif seller_id == 2:
                        state = self.state_b1s2
                    else:
                        state = self.state_b1s3
                elif buyer_id == 2:
                    if seller_id == 1:
                        state = self.state_b2s1
                    elif seller_id == 2:
                        state = self.state_b2s2
                    else:
                        state = self.state_b2s3
                else:
                    if seller_id == 1:
                        state = self.state_b3s1
                    elif seller_id == 2:
                        state = self.state_b3s2
                    else:
                        state = self.state_b3s3
                
                output_lines.append(f"\nBuyer {buyer_id} - Seller {seller_id}:")
                if state.buyer_price is not None:
                    output_lines.append(f"  Buyer Total Price: ${state.buyer_price:.2f}")
                else:
                    output_lines.append(f"  Buyer Total Price: Not specified")
                if state.seller_price is not None:
                    output_lines.append(f"  Seller Total Price: ${state.seller_price:.2f}")
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
        self.memory_b1s3.clear()
        self.memory_b2s1.clear()
        self.memory_b2s2.clear()
        self.memory_b2s3.clear()
        self.memory_b3s1.clear()
        self.memory_b3s2.clear()
        self.memory_b3s3.clear()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b1s3 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
        self.state_b2s3 = NegotiationState()
        self.state_b3s1 = NegotiationState()
        self.state_b3s2 = NegotiationState()
        self.state_b3s3 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history_b1s1": self.memory_b1s1.get_history(),
            "conversation_history_b1s2": self.memory_b1s2.get_history(),
            "conversation_history_b1s3": self.memory_b1s3.get_history(),
            "conversation_history_b2s1": self.memory_b2s1.get_history(),
            "conversation_history_b2s2": self.memory_b2s2.get_history(),
            "conversation_history_b2s3": self.memory_b2s3.get_history(),
            "conversation_history_b3s1": self.memory_b3s1.get_history(),
            "conversation_history_b3s2": self.memory_b3s2.get_history(),
            "conversation_history_b3s3": self.memory_b3s3.get_history(),
            "current_round": self.current_round,
            "buyer1_selected_seller": self.buyer1_selected_seller,
            "buyer2_selected_seller": self.buyer2_selected_seller,
            "buyer3_selected_seller": self.buyer3_selected_seller,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,
            "b1s1_seller_price": self.state_b1s1.seller_price,
            "b1s2_buyer_price": self.state_b1s2.buyer_price,
            "b1s2_seller_price": self.state_b1s2.seller_price,
            "b1s3_buyer_price": self.state_b1s3.buyer_price,
            "b1s3_seller_price": self.state_b1s3.seller_price,
            "b2s1_buyer_price": self.state_b2s1.buyer_price,
            "b2s1_seller_price": self.state_b2s1.seller_price,
            "b2s2_buyer_price": self.state_b2s2.buyer_price,
            "b2s2_seller_price": self.state_b2s2.seller_price,
            "b2s3_buyer_price": self.state_b2s3.buyer_price,
            "b2s3_seller_price": self.state_b2s3.seller_price,
            "b3s1_buyer_price": self.state_b3s1.buyer_price,
            "b3s1_seller_price": self.state_b3s1.seller_price,
            "b3s2_buyer_price": self.state_b3s2.buyer_price,
            "b3s2_seller_price": self.state_b3s2.seller_price,
            "b3s3_buyer_price": self.state_b3s3.buyer_price,
            "b3s3_seller_price": self.state_b3s3.seller_price,
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
            "buyer3_selected_seller": self.buyer3_selected_seller,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,
            "b1s1_seller_price": self.state_b1s1.seller_price,
            "b1s2_buyer_price": self.state_b1s2.buyer_price,
            "b1s2_seller_price": self.state_b1s2.seller_price,
            "b1s3_buyer_price": self.state_b1s3.buyer_price,
            "b1s3_seller_price": self.state_b1s3.seller_price,
            "b2s1_buyer_price": self.state_b2s1.buyer_price,
            "b2s1_seller_price": self.state_b2s1.seller_price,
            "b2s2_buyer_price": self.state_b2s2.buyer_price,
            "b2s2_seller_price": self.state_b2s2.seller_price,
            "b2s3_buyer_price": self.state_b2s3.buyer_price,
            "b2s3_seller_price": self.state_b2s3.seller_price,
            "b3s1_buyer_price": self.state_b3s1.buyer_price,
            "b3s1_seller_price": self.state_b3s1.seller_price,
            "b3s2_buyer_price": self.state_b3s2.buyer_price,
            "b3s2_seller_price": self.state_b3s2.seller_price,
            "b3s3_buyer_price": self.state_b3s3.buyer_price,
            "b3s3_seller_price": self.state_b3s3.seller_price,
            "final_selected_buyer": self.final_selected_buyer,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
            "product_info": self.product_info,
        }
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text"""
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
        """Check if buyer wants to make a deal"""
        if 'MAKE_DEAL' in text.upper():
            return True
        
        make_deal_patterns = [
            r'\bi\s+accept\b',
            r'\bi\s+agree\b',
            r'\baccept\s+your\s+offer\b',
            r'\bagree\s+to\s+the\s+deal\b',
            r'\bmake\s+a\s+deal\b',
            r'\bwe\s+have\s+a\s+deal\b',
            r'\blet\'?s\s+do\s+it\b',
            r'\bi\'?ll\s+take\s+it\b',
            r'\bi\'?m\s+accepting\b',
            r'\bi\'?m\s+agreeing\b',
            r'\bdeal\s*!',
            r'\bi\s+accept\s+the\s+price\b',
        ]
        
        text_lower = text.lower()
        
        false_positive_patterns = [
            r'hope\s+.*\s+agreement',
            r'hoping\s+.*\s+agreement',
            r'would\s+like\s+.*\s+agreement',
            r'looking\s+forward\s+.*\s+agreement',
        ]
        
        for pattern in false_positive_patterns:
            if re.search(pattern, text_lower):
                return False
        
        for pattern in make_deal_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _calculate_reward(self) -> float:
        """Calculate global reward"""
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_selected_seller is not None and self.final_deal_price is not None:
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            if self.final_selected_buyer == 1:
                selected_buyer_max_price = self.buyer1_max_price
            elif self.final_selected_buyer == 2:
                selected_buyer_max_price = self.buyer2_max_price
            else:
                selected_buyer_max_price = self.buyer3_max_price
            
            if self.final_selected_seller == 1:
                selected_seller_min_price = self.seller1_min_price
            elif self.final_selected_seller == 2:
                selected_seller_min_price = self.seller2_min_price
            else:
                selected_seller_min_price = self.seller3_min_price
            
            if selected_buyer_max_price is not None:
                buyer_savings = selected_buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            if selected_seller_min_price is not None:
                seller_profit = deal_price - selected_seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            reward += time_cost * self.reward_weights["time_cost"]
            
            print(f"Global Reward = buyer{self.final_selected_buyer}_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller{self.final_selected_seller}_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{self.final_selected_buyer}_max={selected_buyer_max_price}, deal_price={deal_price:.2f}, seller{self.final_selected_seller}_min={selected_seller_min_price}, round={self.current_round})")
            
            return reward
        else:
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self, buyer_id: int) -> float:
        """Calculate reward from buyer's perspective"""
        time_cost = -self.current_round
        
        deal_reached_with_this_buyer = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.final_selected_buyer == buyer_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_buyer:
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            if buyer_id == 1:
                buyer_max_price = self.buyer1_max_price
            elif buyer_id == 2:
                buyer_max_price = self.buyer2_max_price
            else:
                buyer_max_price = self.buyer3_max_price
            
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
        """Calculate reward from seller's perspective"""
        time_cost = -self.current_round
        
        deal_reached_with_this_seller = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.final_selected_seller == seller_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_seller:
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            if seller_id == 1:
                seller_min_price = self.seller1_min_price
            elif seller_id == 2:
                seller_min_price = self.seller2_min_price
            else:
                seller_min_price = self.seller3_min_price
            
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
        """Calculate step reward from buyer's perspective for current round"""
        round_cost = -self.current_round
        reward = 0.0
        buyer_savings = 0.0
        
        buyer_price = None
        buyer_max_price = None
        if buyer_id == 1:
            buyer_max_price = self.buyer1_max_price
            if self.buyer1_selected_seller == 1:
                buyer_price = self.state_b1s1.buyer_price
            elif self.buyer1_selected_seller == 2:
                buyer_price = self.state_b1s2.buyer_price
            else:
                buyer_price = self.state_b1s3.buyer_price
        elif buyer_id == 2:
            buyer_max_price = self.buyer2_max_price
            if self.buyer2_selected_seller == 1:
                buyer_price = self.state_b2s1.buyer_price
            elif self.buyer2_selected_seller == 2:
                buyer_price = self.state_b2s2.buyer_price
            else:
                buyer_price = self.state_b2s3.buyer_price
        else:
            buyer_max_price = self.buyer3_max_price
            if self.buyer3_selected_seller == 1:
                buyer_price = self.state_b3s1.buyer_price
            elif self.buyer3_selected_seller == 2:
                buyer_price = self.state_b3s2.buyer_price
            else:
                buyer_price = self.state_b3s3.buyer_price
        
        if buyer_price is not None and buyer_max_price is not None:
            buyer_savings = buyer_max_price - buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self, seller_id: int) -> float:
        """Calculate step reward from seller's perspective for current round"""
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        seller_state = None
        seller_min_price = None
        if seller_id == 1:
            seller_min_price = self.seller1_min_price
            states = []
            if self.buyer1_selected_seller == 1 and self.state_b1s1.seller_price is not None:
                states.append((self.state_b1s1.seller_price, self.state_b1s1))
            if self.buyer2_selected_seller == 1 and self.state_b2s1.seller_price is not None:
                states.append((self.state_b2s1.seller_price, self.state_b2s1))
            if self.buyer3_selected_seller == 1 and self.state_b3s1.seller_price is not None:
                states.append((self.state_b3s1.seller_price, self.state_b3s1))
            if states:
                seller_state = max(states, key=lambda x: x[0])[1]
        elif seller_id == 2:
            seller_min_price = self.seller2_min_price
            states = []
            if self.buyer1_selected_seller == 2 and self.state_b1s2.seller_price is not None:
                states.append((self.state_b1s2.seller_price, self.state_b1s2))
            if self.buyer2_selected_seller == 2 and self.state_b2s2.seller_price is not None:
                states.append((self.state_b2s2.seller_price, self.state_b2s2))
            if self.buyer3_selected_seller == 2 and self.state_b3s2.seller_price is not None:
                states.append((self.state_b3s2.seller_price, self.state_b3s2))
            if states:
                seller_state = max(states, key=lambda x: x[0])[1]
        else:
            seller_min_price = self.seller3_min_price
            states = []
            if self.buyer1_selected_seller == 3 and self.state_b1s3.seller_price is not None:
                states.append((self.state_b1s3.seller_price, self.state_b1s3))
            if self.buyer2_selected_seller == 3 and self.state_b2s3.seller_price is not None:
                states.append((self.state_b2s3.seller_price, self.state_b2s3))
            if self.buyer3_selected_seller == 3 and self.state_b3s3.seller_price is not None:
                states.append((self.state_b3s3.seller_price, self.state_b3s3))
            if states:
                seller_state = max(states, key=lambda x: x[0])[1]
        
        if seller_state is not None and seller_state.seller_price is not None and seller_min_price is not None:
            seller_profit = seller_state.seller_price - seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
