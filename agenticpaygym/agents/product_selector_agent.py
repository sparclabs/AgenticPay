"""Product Selector Agent Implementation"""

import re
from typing import Dict, List, Any, Optional
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.models.base_llm import BaseLLM


class ProductSelectorAgent(BaseAgent):
    """Product Selector Agent
    
    Selects the most appropriate product from available products based on user requirements.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        name: str = "ProductSelector",
        role_description: str = "a product selector that helps match buyer requirements with available products. You analyze buyer needs and select the most appropriate product from the inventory.",
    ):
        """Initialize Product Selector Agent
        
        Args:
            llm: LLM interface
            name: Agent name
            role_description: Role description
        """
        super().__init__(llm, role_description, name)
    
    def select_product(
        self,
        user_requirement: str,
        products: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Select the most appropriate product based on user requirement
        
        Args:
            user_requirement: User's product requirement
            products: List of available products
            
        Returns:
            Selected product dictionary
        """
        if not products:
            raise ValueError("Products list cannot be empty")
        
        # Build prompt for product selection
        products_info = "\n".join([
            f"{i+1}. {p['name']} - Brand: {p['brand']}, Price: ${p['price']:.2f}, "
            f"Features: {', '.join(p['features'])}, Material: {p['material']}"
            for i, p in enumerate(products)
        ])
        
        prompt = f"""You are a product selector helping to match buyer requirements with available products.

Available Products:
{products_info}

Buyer's Requirement: "{user_requirement}"

Based on the buyer's requirement, select the most appropriate product by responding with ONLY the product number (1, 2, etc.).

Your response should be just the number:"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.3).strip()
            # Extract number from response
            match = re.search(r'\d+', response)
            if match:
                product_index = int(match.group()) - 1
                if 0 <= product_index < len(products):
                    return products[product_index]
        except Exception as e:
            print(f"Error selecting product with LLM: {e}")
        
        # Fallback: simple keyword matching
        user_lower = user_requirement.lower()
        for product in products:
            product_name_lower = product['name'].lower()
            # Check if any keyword from product name appears in user requirement
            product_keywords = product_name_lower.split()
            if any(keyword in user_lower for keyword in product_keywords if len(keyword) > 3):
                return product
        
        # Additional fallback: check features
        for product in products:
            features = [f.lower() for f in product.get('features', [])]
            if any(feature in user_lower for feature in features):
                return product
        
        # Default: return first product if no match
        return products[0]
    
    def respond(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        """Generate response (not used for product selection, but required by BaseAgent)
        
        This method is required by BaseAgent but ProductSelectorAgent primarily uses
        select_product() method instead.
        
        Args:
            conversation_history: Conversation history
            current_state: Current state
            
        Returns:
            Agent's response text
        """
        # This agent doesn't participate in conversations, so return empty string
        return ""

