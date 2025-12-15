"""Task4 Sequential Three-Seller Per One Product Negotiation Example

Demonstrates how to use the Task4SequentialThreeSellerPerOneProductNegotiation to negotiate sequentially with three sellers,
where each seller has their own unique product. Buyer chooses one seller per round to negotiate with.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.multi_products_multi_seller.Task4_sequential_three_seller_per_one_product_negotiation import Task4SequentialThreeSellerPerOneProductNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM
import re


def extract_seller_choice(buyer_response: str, observation: dict) -> int:
    """Extract seller choice from buyer's response
    
    Buyer should indicate which seller they want to negotiate with.
    Look for patterns like "seller 1", "seller1", "first seller", etc.
    
    Args:
        buyer_response: Buyer's response text
        observation: Current observation from environment
        
    Returns:
        1, 2, or 3, indicating which seller buyer wants to negotiate with
    """
    response_lower = buyer_response.lower()
    
    # Look for explicit seller mentions
    if re.search(r'seller\s*[123]|first\s+seller|second\s+seller|third\s+seller|seller\s*one|seller\s*two|seller\s*three', response_lower):
        if re.search(r'seller\s*3|third\s+seller|seller\s*three', response_lower):
            return 3
        elif re.search(r'seller\s*2|second\s+seller|seller\s*two', response_lower):
            return 2
        elif re.search(r'seller\s*1|first\s+seller|seller\s*one', response_lower):
            return 1
    
    # If no explicit mention, try to infer from context
    # Check if buyer mentions prices or other indicators
    seller1_price = observation.get("seller1_price")
    seller2_price = observation.get("seller2_price")
    seller3_price = observation.get("seller3_price")
    
    # If buyer mentions a specific price, try to match it
    price_match = re.search(r'\$?(\d+\.?\d*)', buyer_response)
    if price_match:
        mentioned_price = float(price_match.group(1))
        if seller1_price is not None and abs(mentioned_price - seller1_price) < 5:
            return 1
        elif seller2_price is not None and abs(mentioned_price - seller2_price) < 5:
            return 2
        elif seller3_price is not None and abs(mentioned_price - seller3_price) < 5:
            return 3
    
    # Default: if no clear indication, check which seller has a better price
    prices = []
    if seller1_price is not None:
        prices.append((1, seller1_price))
    if seller2_price is not None:
        prices.append((2, seller2_price))
    if seller3_price is not None:
        prices.append((3, seller3_price))
    
    if prices:
        # Choose the one with lowest price if available
        prices.sort(key=lambda x: x[1])
        return prices[0][0]
    
    # Final default: seller1
    return 1


def main():
    """Main function: Demonstrates sequential multi-seller negotiation flow with different products"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize LLM
    print("Initializing LLM...")
    llm = OpenAILLM(api_key=api_key, model="gpt-4o-mini-2024-07-18")  # gpt-4o-mini-2024-07-18, gpt-3.5-turbo
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    print("Creating agents...")
    buyer_max_price = 120.0  # Maximum acceptable purchase price for buyer (confidential)
    seller1_min_price = 80.0  # Minimum acceptable selling price for seller1 (confidential)
    seller2_min_price = 85.0  # Minimum acceptable selling price for seller2 (confidential, different from seller1)
    seller3_min_price = 90.0  # Minimum acceptable selling price for seller3 (confidential, different from seller1 and seller2)
    
    buyer = BuyerAgent(llm=llm, buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(llm=llm, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(llm=llm, seller_min_price=seller2_min_price)
    seller3 = SellerAgent(llm=llm, seller_min_price=seller3_min_price)
    
    # Create environment
    print("Creating sequential multi-seller negotiation environment...")
    env = Task4SequentialThreeSellerPerOneProductNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        seller3_agent=seller3,
        max_rounds=20,
        initial_seller1_price=150.0,  # Initial price offered by seller1
        initial_seller2_price=160.0,  # Initial price offered by seller2 (higher)
        initial_seller3_price=170.0,  # Initial price offered by seller3 (highest)
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 bottom price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 bottom price (confidential)
        seller3_min_price=seller3_min_price,  # Seller3 bottom price (confidential)
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=5.0,
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
    print(f"User Profile: {user_profile}")
    
    # Get user requirement
    print("\n" + "="*60)
    print("Please enter the product requirement you want to purchase:")
    user_requirement = input("> ").strip()
    if not user_requirement:
        print("No requirement entered, using default requirement...")
        user_requirement = "I need a high-quality winter jacket for cold weather"
        print(f"Using default requirement: {user_requirement}")
    
    # Reset environment with different products for each seller
    print("\n" + "="*60)
    print("Starting new sequential negotiation with three sellers (each with different jacket models)...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        seller1_product_info={
            "name": "Premium Winter Jacket",
            "brand": "Mountain Gear",
            "price": 180.0,  # The product's own price
            "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
            "condition": "New",
            "material": "Gore-Tex",
            "color": "Black",
            "model": "MG-Pro-2024",  # Different model from seller2 and seller3
        },
        seller2_product_info={
            "name": "Deluxe Winter Jacket",
            "brand": "Alpine Outfitters",
            "price": 190.0,  # The product's own price (different from seller1)
            "features": ["Waterproof", "Insulated", "Windproof", "Fleece Lined"],
            "condition": "New",
            "material": "Polyester Blend",
            "color": "Navy Blue",
            "model": "AW-2024",  # Different model from seller1 and seller3
        },
        seller3_product_info={
            "name": "Ultra Winter Jacket",
            "brand": "Summit Wear",
            "price": 200.0,  # The product's own price (different from seller1 and seller2)
            "features": ["Waterproof", "Insulated", "Windproof", "Down Filled"],
            "condition": "New",
            "material": "Nylon with Down",
            "color": "Dark Gray",
            "model": "SW-Ultra-2024",  # Different model from seller1 and seller2
        },
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Each round, buyer chooses one seller to negotiate with
        # Buyer can see all three sellers' information in the observation
        # Let buyer decide which seller to negotiate with and provide negotiation message
        # We'll use a combined conversation history that includes all sellers' conversations
        combined_history = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_seller1", []):
            combined_history.append({
                **msg,
                "content": f"[Seller 1] {msg['content']}"
            })
        # Add seller2 messages with prefix
        for msg in observation.get("conversation_history_seller2", []):
            combined_history.append({
                **msg,
                "content": f"[Seller 2] {msg['content']}"
            })
        # Add seller3 messages with prefix
        for msg in observation.get("conversation_history_seller3", []):
            combined_history.append({
                **msg,
                "content": f"[Seller 3] {msg['content']}"
            })
        
        # Get buyer's response - buyer should indicate which seller they want to negotiate with
        buyer_response = buyer.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": "You are negotiating with three sellers, each offering a different jacket model. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1, 2, or 3) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2' or 'I'll negotiate with seller 3'."
            }
        )
        
        # Extract seller choice from buyer's response
        selected_seller = extract_seller_choice(buyer_response, observation)
        print(f"\n[Buyer chooses to negotiate with Seller {selected_seller} this round]")
        
        # Use buyer's full response as the negotiation message
        # The response may include the choice statement, which is fine as it's buyer's natural expression
        buyer_action = buyer_response
        
        # Get the conversation history for the selected seller
        if selected_seller == 1:
            conversation_history = observation["conversation_history_seller1"]
        elif selected_seller == 2:
            conversation_history = observation["conversation_history_seller2"]
        else:  # selected_seller == 3
            conversation_history = observation["conversation_history_seller3"]
        
        # Get the selected seller's response
        if selected_seller == 1:
            seller_action = seller1.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        elif selected_seller == 2:
            seller_action = seller2.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        else:  # selected_seller == 3
            seller_action = seller3.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        
        # Print conversation content for this round
        current_round = observation.get('current_round', 0)
        print(f"\n{'='*60}")
        print(f"Round {current_round} Conversation:")
        print(f"{'='*60}")
        print(f"[BUYER to Seller {selected_seller}]: {buyer_action}")
        print(f"[SELLER {selected_seller}]: {seller_action}")
        print(f"{'='*60}")
        
        # Execute step with selected seller and actions
        observation, reward, terminated, truncated, info = env.step(
            selected_seller=selected_seller,
            buyer_action=buyer_action,
            seller_action=seller_action
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_seller'):
                print(f"Final Selected Seller: Seller {info['selected_seller']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
                # Display product info for selected seller
                if info['selected_seller'] == 1:
                    product_info = info.get('seller1_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
                elif info['selected_seller'] == 2:
                    product_info = info.get('seller2_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
                elif info['selected_seller'] == 3:
                    product_info = info.get('seller3_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
            seller1_price = info.get('seller1_price', 0) or 0
            buyer_price_seller1 = info.get('buyer_price_seller1', 0) or 0
            seller2_price = info.get('seller2_price', 0) or 0
            buyer_price_seller2 = info.get('buyer_price_seller2', 0) or 0
            seller3_price = info.get('seller3_price', 0) or 0
            buyer_price_seller3 = info.get('buyer_price_seller3', 0) or 0
            print(f"Seller1 Prices: Seller=${seller1_price:.2f} | Buyer=${buyer_price_seller1:.2f}")
            print(f"Seller2 Prices: Seller=${seller2_price:.2f} | Buyer=${buyer_price_seller2:.2f}")
            print(f"Seller3 Prices: Seller=${seller3_price:.2f} | Buyer=${buyer_price_seller3:.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Reward: {reward:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Close environment
    env.close()
    print("\nNegotiation completed!")


if __name__ == "__main__":
    main()

