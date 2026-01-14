"""Seller Agent Implementation"""

import re
from typing import Dict, List, Any, Optional, Union
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.models.base_llm import BaseLLM
from agenticpaygym.models.base_vlm import BaseVLM


class SellerAgent(BaseAgent):
    """Seller Agent
    
    Represents the seller, negotiates with the buyer based on product information and market conditions.
    """
    
    def __init__(
        self,
        model: Union[BaseLLM, BaseVLM],
        name: str = "Seller",
        role_description: str = "You are a seller trying to maximize profit while being reasonable. You are professional, friendly, and want to close a deal that benefits both parties.",
        seller_min_price: Optional[float] = None,
    ):
        """Initialize Seller Agent
        
        Args:
            model: LLM or VLM interface (supports both BaseLLM and BaseVLM)
            name: Agent name
            role_description: Role description
            seller_min_price: Minimum acceptable selling price for seller (bottom price, confidential information)
        """
        super().__init__(model, role_description, name)
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
#         seller_guidance = f"""

# IMPORTANT REMINDERS:
# - Your initial asking price is ${initial_price}
# - Your minimum acceptable price (confidential - do not reveal this to the buyer) is ${min_price}
# - Current product information: {product_info}
# {available_products_info}
# - Consider the environment factors: {self.context.get('environment_info', {})}
# - Be professional and try to find a win-win solution
# - Highlight the value and quality of your product
# - If the buyer's needs might be better met by another product in your inventory, you can suggest it
# - Be willing to negotiate but don't go below your minimum acceptable price
# - Try to negotiate the price as high as possible, but ensure the deal is successful in the end
# - **CRITICAL: Each conversation you MUST make one price offer, you MUST use the format: ### SELLER_PRICE($X) ###**
# - Example: "I can offer ### SELLER_PRICE($150) ### for this product"
# - Example: "How about ### SELLER_PRICE($130.00) ###?"
# - This specific format is required for the system to correctly extract your offer price
# - Consider market conditions and seasonality
# - NEVER reveal your minimum acceptable price to the buyer - keep it confidential
# - Keep communication short and concise.

# Now, respond as {self.name}:
# """
        seller_guidance = f"""
IMPORTANT REMINDERS:
- Your initial asking price is ${initial_price}.
- Your minimum acceptable price (confidential) is ${min_price}. Never reveal it.
- Current product information: {product_info}
{available_products_info}
- Consider the environment factors: {self.context.get('environment_info', {})}.


- **CRITICAL: In each turn, you MUST make exactly one price offer using the format:**
  ### SELLER_PRICE($X) ###
- Example: "I can offer ### SELLER_PRICE($15) ### for this product."
- Example: "How about ### SELLER_PRICE($13.00) ###?"
- This specific format is required for the system to correctly extract your offer price.
- NEVER reveal your minimum acceptable price to the buyer.
- Keep communication short, professional, and negotiation-focused.

Now, respond as {self.name}:
"""

        full_prompt = prompt + seller_guidance
        
        # Extract images from current_state if VLM is used
        images = None
        if self.is_vlm:
            # Check for images in current_state (e.g., product images)
            images = current_state.get('images') or current_state.get('product_images')
            # Also check in context
            if images is None:
                images = self.context.get('images') or self.context.get('product_images')
        
        # Generate response: VLM supports images, LLM doesn't
        if self.is_vlm and images is not None:
            response = self.model.generate(
                full_prompt, 
                images=images,
                temperature=0.7,
                max_tokens=1024  # Ensure complete response generation
            )
        else:
            response = self.model.generate(
                full_prompt, 
                temperature=0.7,
                max_tokens=1024  # Ensure complete response generation
            )
        
        # Remove <think>...</think> tags and their content using regex
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        return cleaned_response.strip()

