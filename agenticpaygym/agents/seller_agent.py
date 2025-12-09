"""Seller Agent Implementation"""

from typing import Dict, List, Any, Optional
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.llm.base_llm import BaseLLM


class SellerAgent(BaseAgent):
    """Seller Agent
    
    Represents the seller, negotiates with the buyer based on product information and market conditions.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        name: str = "Seller",
        role_description: str = "a seller trying to maximize profit while being reasonable. You are professional, friendly, and want to close a deal that benefits both parties.",
        seller_min_price: Optional[float] = None,
    ):
        """Initialize Seller Agent
        
        Args:
            llm: LLM interface
            name: Agent name
            role_description: Role description
            seller_min_price: Minimum acceptable selling price for seller (bottom price, confidential information)
        """
        super().__init__(llm, role_description, name)
        self.seller_min_price = seller_min_price
    
    def respond(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        """Generate Seller response
        
        Args:
            conversation_history: Conversation history
            current_state: Current state
            
        Returns:
            Seller's response text
        """
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        prompt = self._build_prompt(conversation_history, current_state)
        
        # Get seller's minimum acceptable price (bottom price)
        min_price = self.seller_min_price or self.context.get('min_price', 'unknown')
        initial_price = self.context.get('initial_price', 'unknown')
        product_info = self.context.get('product_info', {})
        available_products = self.context.get('available_products', [])
        
        # Format available products information
        available_products_info = ""
        if available_products:
            available_products_info = "\n\nAVAILABLE PRODUCTS IN YOUR INVENTORY:\n"
            for i, prod in enumerate(available_products, 1):
                available_products_info += f"{i}. {prod.get('name', 'Unknown')} - "
                available_products_info += f"Brand: {prod.get('brand', 'N/A')}, "
                available_products_info += f"Price: ${prod.get('price', 0):.2f}, "
                available_products_info += f"Features: {', '.join(prod.get('features', []))}\n"
            available_products_info += "\nYou can suggest other products from your inventory if they better match the buyer's needs.\n"
        
        # Add Seller-specific guidance
        seller_guidance = f"""

IMPORTANT REMINDERS:
- Your initial asking price is ${initial_price}
- Your minimum acceptable price (confidential - do not reveal this to the buyer) is ${min_price}
- Current product information: {product_info}
{available_products_info}
- Consider the environment factors: {self.context.get('environment_info', {})}
- Be professional and try to find a win-win solution
- Highlight the value and quality of your product
- If the buyer's needs might be better met by another product in your inventory, you can suggest it
- Be willing to negotiate but don't go below your minimum acceptable price
- Always mention specific prices when making offers (e.g., "I can offer $X")
- Consider market conditions and seasonality
- NEVER reveal your minimum acceptable price to the buyer - keep it confidential

Now, respond as {self.name}:
"""
        full_prompt = prompt + seller_guidance
        
        response = self.llm.generate(full_prompt, temperature=0.7)
        return response.strip()

