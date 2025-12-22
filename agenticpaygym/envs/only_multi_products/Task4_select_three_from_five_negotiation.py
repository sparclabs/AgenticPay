"""Task4 Select Three from Five Products Negotiation Environment Implementation

Supports negotiation where user needs 3 products, and buyer selects 3 from 5 available products
for total price negotiation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, List

from agenticpaygym.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.utils.negotiation_state import NegotiationState


class Task4SelectThreeFromFiveNegotiation(BaseEnv):
    """Task4 Select Three from Five Products Negotiation Environment
    
    Manages negotiation process where user needs 3 products, and buyer selects 3 from 5 available products.
    buyer_max_price and seller_min_price represent the total expected cost for the selected 3 products.
    Buyer agent automatically selects 3 products based on user_requirement and product_info.
    """
    
    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_price: float = 100.0,
        buyer_max_price: Optional[float] = None,
        seller_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize select-three-from-five negotiation environment
        
        Args:
            buyer_agent: Buyer Agent (will automatically select 3 products from 5)
            seller_agent: Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller_price: Initial total price offered by seller for selected 3 products
            buyer_max_price: Maximum acceptable total price for buyer (confidential, for selected 3 products)
            seller_min_price: Minimum acceptable total price for seller (confidential, for selected 3 products)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
            reward_weights: Reward weights configuration dict with keys:
                - buyer_savings: weight for buyer savings (default: 1.0)
                - seller_profit: weight for seller profit (default: 1.0)
                - time_cost: weight for time cost (default: 0.1)
        """
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_price = initial_seller_price
        self.buyer_max_price = buyer_max_price
        self.seller_min_price = seller_min_price
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
        
        # State management
        self.memory = ConversationMemory()
        self.state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info: Optional[Dict[str, Any]] = None
        self.selected_products: List[Dict[str, Any]] = []  # Track selected 3 products
    
    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new negotiation
        
        Args:
            user_requirement: User requirement description (should describe needing 3 products)
            product_info: Product information containing five products and their prices
                Expected format: {
                    "products": [
                        {"name": "Product1", "price": 100.0, ...},
                        {"name": "Product2", "price": 80.0, ...},
                        {"name": "Product3", "price": 60.0, ...},
                        {"name": "Product4", "price": 50.0, ...},
                        {"name": "Product5", "price": 40.0, ...}
                    ]
                }
                All 5 products should meet the user requirement for 3 products.
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory.clear()
        self.state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info = product_info or {}
        self.selected_products = []
        
        # Extract product information
        products = self.product_info.get("products", [])
        if len(products) < 5:
            raise ValueError("product_info must contain at least 5 products in 'products' list")
        
        # Initialize Agents
        # Buyer agent will automatically select 3 products based on user_requirement and product_info
        buyer_context = {
            "user_requirement": user_requirement,  # User needs 3 products
            "max_price": self.buyer_max_price,  # Total max price for selected 3 products
            "user_profile": user_profile,
            "environment_info": self.environment_info,
            "product_info": self.product_info,  # Buyer can see all 5 products and will select 3
            "required_count": 3,  # User needs 3 products
        }
        self.buyer_agent.initialize(buyer_context)
        
        seller_context = {
            "product_info": self.product_info,  # Seller can see all 5 products
            "initial_price": self.initial_seller_price,  # Initial total price for selected 3 products
            "min_price": self.seller_min_price,  # Total min price for selected 3 products
            "environment_info": self.environment_info,
            "user_requirement": user_requirement,  # Seller knows user needs 3 products
            "required_count": 3,  # User needs 3 products
        }
        self.seller_agent.initialize(seller_context)
        
        # Seller gives initial offer (total price for selected 3 products)
        # Seller knows user needs 3 products, but doesn't know which 3 buyer will select
        product_names = [p.get("name", "Product") for p in products]
        product_list = ", ".join(product_names)
        initial_message = f"I have 5 products available: {product_list}. Based on your requirement for 3 products, I'm offering a total of ${self.initial_seller_price:.2f} for the 3 products you select."
        self.memory.add_message("seller", initial_message, self.current_round)
        self.state.update(seller_price=self.initial_seller_price)
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
        Prices represent total price for the selected 3 products.
        Buyer agent automatically selects 3 products based on user_requirement and product_info.
        
        Args:
            buyer_action: Buyer's response text (optional, can be None if buyer doesn't respond)
            seller_action: Seller's response text (optional, can be None if seller doesn't respond)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Add messages to memory (both agents respond in the same round)
        # Order: buyer -> seller
        if buyer_action is not None:
            self.memory.add_message("buyer", buyer_action, self.current_round)
            # Extract buyer price (total price for selected 3 products)
            buyer_price = self._extract_price(buyer_action)
            if buyer_price is not None:
                self.state.update(buyer_price=buyer_price)
                self.negotiation_info.buyer_price = buyer_price
                self.negotiation_info.current_price = buyer_price
        
        if seller_action is not None:
            self.memory.add_message("seller", seller_action, self.current_round)
            # Extract seller price (total price for selected 3 products)
            seller_price = self._extract_price(seller_action)
            if seller_price is not None:
                self.state.update(seller_price=seller_price)
                self.negotiation_info.seller_price = seller_price
                self.negotiation_info.current_price = seller_price
            
            # Try to extract selected products from buyer's message (optional, buyer agent handles selection)
            # This is just for tracking, buyer agent makes the selection decision
            selected = self._extract_selected_products(buyer_action)
            if selected:
                self.selected_products = selected
        
        # Check if agreement is reached (after both agents have responded)
        terminated = False
        truncated = False
        reward = 0.0
        seller_reward = 0.0
        buyer_reward = 0.0
        
        if self._check_agreement():
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            agreed_price = (self.state.buyer_price + self.state.seller_price) / 2
            self.state.update(agreed_price=agreed_price)
            self.negotiation_info.current_price = agreed_price
            reward = self._calculate_reward()
            seller_reward = self._calculate_seller_reward()
            buyer_reward = self._calculate_buyer_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            reward = self._calculate_reward()
            seller_reward = self._calculate_seller_reward()
            buyer_reward = self._calculate_buyer_reward()
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        step_seller_reward = self._calculate_step_seller_reward()
        step_buyer_reward = self._calculate_step_buyer_reward()
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_seller_reward"] = step_seller_reward
        info["step_buyer_reward"] = step_buyer_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            info["seller_reward"] = seller_reward
            info["buyer_reward"] = buyer_reward
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays buyer and seller outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        Prices shown are total prices for the selected 3 products.
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
        # Display all available products
        if self.product_info:
            products = self.product_info.get("products", [])
            if products:
                output_lines.append(f"\n{'='*60}")
                output_lines.append("Available Products (5 total, user needs 3):")
                for i, p in enumerate(products, 1):
                    name = p.get("name", f"Product {i}")
                    price = p.get("price", 0.0)
                    output_lines.append(f"  {i}. {name}: ${price:.2f}")
                total_price = sum(p.get("price", 0.0) for p in products)
                output_lines.append(f"  Total Price (all 5): ${total_price:.2f}")
                
                # Display selected products if known
                if self.selected_products:
                    output_lines.append(f"\n  Selected Products (3):")
                    selected_total = sum(p.get("price", 0.0) for p in self.selected_products)
                    for i, p in enumerate(self.selected_products, 1):
                        name = p.get("name", f"Product {i}")
                        price = p.get("price", 0.0)
                        output_lines.append(f"    {i}. {name}: ${price:.2f}")
                    output_lines.append(f"    Total Price (selected 3): ${selected_total:.2f}")
                else:
                    output_lines.append(f"\n  Selected Products: Buyer will select 3 products during negotiation")
                
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
                
                # Display buyer message first (if exists)
                buyer_msg = next(
                    (msg for msg in current_round_messages if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg:
                    output_lines.append(f"\n[BUYER Output]:")
                    output_lines.append(f"  {buyer_msg['content']}")
                
                # Display seller message (if exists)
                seller_msg = next(
                    (msg for msg in current_round_messages if msg["role"] == "seller"), 
                    None
                )
                if seller_msg:
                    output_lines.append(f"\n[SELLER Output]:")
                    output_lines.append(f"  {seller_msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display buyer price (total for selected 3 products)
        if self.state.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price (for 3 products): ${self.state.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price (for 3 products): Not specified")
        
        # Display seller price (total for selected 3 products)
        if self.state.seller_price is not None:
            output_lines.append(f"  Seller Total Price (for 3 products): ${self.state.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price (for 3 products): Not specified")
        
        # Check agreement status and provide reason
        is_agreed = self._check_agreement()
        agreement_reason = self._get_agreement_reason()
        
        output_lines.append(f"  Agreement Status: {'✓ AGREED' if is_agreed else '✗ NOT AGREED'}")
        output_lines.append(f"  Reason: {agreement_reason}")
        
        # Display agreed price if agreement is reached
        if self.state.agreed_price is not None:
            output_lines.append(f"  Agreed Total Price (for 3 products): ${self.state.agreed_price:.2f}")
        
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
    
    def _extract_selected_products(self, text: str) -> List[Dict[str, Any]]:
        """Extract selected products from buyer's message (optional helper)
        
        This is a helper method to track which products buyer mentions.
        The actual selection is done by the buyer agent based on user_requirement and product_info.
        
        Args:
            text: Buyer's message text
            
        Returns:
            List of selected product dictionaries, empty if not found
        """
        # This is optional - buyer agent handles selection internally
        # We can try to extract mentioned product names, but it's not required
        products = self.product_info.get("products", [])
        selected = []
        
        for product in products:
            product_name = product.get("name", "").lower()
            if product_name and product_name in text.lower():
                selected.append(product)
        
        # Return up to 3 products if found
        return selected[:3] if len(selected) >= 3 else []
    
    def _get_agreement_reason(self) -> str:
        """Get reason for agreement or non-agreement
        
        Returns:
            String describing why agreement was or was not reached
        """
        if self.state.buyer_price is None or self.state.seller_price is None:
            if self.state.buyer_price is None and self.state.seller_price is None:
                return "Both buyer and seller prices are not specified yet"
            elif self.state.buyer_price is None:
                return "Buyer price is not specified yet"
            else:
                return "Seller price is not specified yet"
        
        price_diff = abs(self.state.buyer_price - self.state.seller_price)
        
        if price_diff <= self.price_tolerance:
            return f"Price difference (${price_diff:.2f}) is within tolerance (${self.price_tolerance:.2f})"
        else:
            return f"Price difference (${price_diff:.2f}) exceeds tolerance (${self.price_tolerance:.2f})"
    
    def close(self):
        """Close environment, cleanup resources"""
        self.memory.clear()
        self.state = NegotiationState()
        self.selected_products = []
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        return {
            "conversation_history": self.memory.get_history(),
            "current_round": self.current_round,
            "seller_price": self.state.seller_price,  # Total price for selected 3 products
            "buyer_price": self.state.buyer_price,  # Total price for selected 3 products
            "status": self.negotiation_info.status.value,
            "product_info": self.product_info,  # All 5 products
            "selected_products": self.selected_products,  # Selected 3 products (if known)
            "required_count": 3,  # User needs 3 products
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "seller_price": self.state.seller_price,  # Total price for selected 3 products
            "buyer_price": self.state.buyer_price,  # Total price for selected 3 products
            "agreed_price": self.state.agreed_price,  # Total agreed price for selected 3 products
            "negotiation_info": self.negotiation_info,
            "product_info": self.product_info,  # All 5 products
            "selected_products": self.selected_products,  # Selected 3 products (if known)
            "required_count": 3,  # User needs 3 products
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
    
    def _check_agreement(self) -> bool:
        """Check if agreement is reached
        
        When the total prices of both buyer and seller are within the tolerance range, an agreement is considered reached.
        
        Returns:
            Whether agreement is reached
        """
        if self.state.buyer_price is None or self.state.seller_price is None:
            return False
        
        price_diff = abs(self.state.buyer_price - self.state.seller_price)
        return price_diff <= self.price_tolerance
    
    def _calculate_reward(self) -> float:
        """Calculate reward
        
        Calculate reward value based on negotiation result.
        
        If deal is reached:
            reward = buyer savings + seller profit + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer for selected 3 products)
            - seller profit = deal_price - seller_min_price (extra profit for seller for selected 3 products)
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
            if self.state.agreed_price is None:
                weighted_time_cost = time_cost * self.reward_weights["time_cost"]
                print(f"Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round})")
                return weighted_time_cost
            
            deal_price = self.state.agreed_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Calculate buyer savings: buyer_max_price - deal_price (for selected 3 products)
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price (for selected 3 products)
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if self.buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if self.seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        
        If deal is reached:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller for selected 3 products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from seller's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            # Deal reached: seller profit + time cost
            if self.state.agreed_price is None:
                weighted_time_cost = time_cost * self.reward_weights["time_cost"]
                print(f"Seller Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round})")
                return weighted_time_cost
            
            deal_price = self.state.agreed_price
            reward = 0.0
            seller_profit = 0.0
            
            # Calculate seller profit: deal_price - seller_min_price (for selected 3 products)
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
    
    def _calculate_buyer_reward(self) -> float:
        """Calculate reward from buyer's perspective
        
        Calculate reward value based on negotiation result from buyer's perspective.
        This reward does not include seller profit.
        
        If deal is reached:
            reward = buyer savings + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer for selected 3 products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from buyer's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            # Deal reached: buyer savings + time cost
            if self.state.agreed_price is None:
                weighted_time_cost = time_cost * self.reward_weights["time_cost"]
                print(f"Buyer Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round})")
                return weighted_time_cost
            
            deal_price = self.state.agreed_price
            reward = 0.0
            buyer_savings = 0.0
            
            # Calculate buyer savings: buyer_max_price - deal_price (for selected 3 products)
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
    
    def _calculate_step_seller_reward(self) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer for selected 3 products)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If seller_price is not specified yet, only round cost is returned.
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        # Calculate seller profit from current offer: seller_price - seller_min_price
        if self.state.seller_price is not None and self.seller_min_price is not None:
            seller_profit = self.state.seller_price - self.seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_buyer_reward(self) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = buyer savings (from current offer) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer for selected 3 products)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If buyer_price is not specified yet, only round cost is returned.
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        buyer_savings = 0.0
        
        # Calculate buyer savings from current offer: buyer_max_price - buyer_price
        if self.state.buyer_price is not None and self.buyer_max_price is not None:
            buyer_savings = self.buyer_max_price - self.state.buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward

