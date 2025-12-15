"""Task1 Parallel Two-Seller Per One Product Negotiation Example

Demonstrates how to use the Task1ParallelTwoSellerPerOneProductNegotiation to negotiate with two sellers
in parallel, where each seller has their own unique product. Buyer chooses the seller with the lower price.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.multi_products_multi_seller.Task1_parallel_two_seller_per_one_product_negotiation import Task1ParallelTwoSellerPerOneProductNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM


def main():
    """Main function: Demonstrates multi-seller negotiation flow with different products"""
    
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
    
    buyer = BuyerAgent(llm=llm, buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(llm=llm, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(llm=llm, seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating multi-seller negotiation environment...")
    env = Task1ParallelTwoSellerPerOneProductNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=20,
        initial_seller1_price=150.0,  # Initial price offered by seller1
        initial_seller2_price=160.0,  # Initial price offered by seller2 (higher)
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 bottom price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 bottom price (confidential)
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
    print("Starting new negotiation with two sellers (each with different jacket models)...")
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
            "model": "MG-Pro-2024",  # Different model from seller2
        },
        seller2_product_info={
            "name": "Deluxe Winter Jacket",
            "brand": "Alpine Outfitters",
            "price": 190.0,  # The product's own price (different from seller1)
            "features": ["Waterproof", "Insulated", "Windproof", "Fleece Lined"],
            "condition": "New",
            "material": "Polyester Blend",
            "color": "Navy Blue",
            "model": "AW-2024",  # Different model from seller1
        },
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Each round, order is: buyer -> seller
        # Get buyer's response to seller1 first
        buyer_action_seller1 = buyer.respond(
            conversation_history=observation["conversation_history_seller1"],
            current_state=observation
        )
        
        # Get buyer's response to seller2
        buyer_action_seller2 = buyer.respond(
            conversation_history=observation["conversation_history_seller2"],
            current_state=observation
        )
        
        # Then get seller1's response
        seller1_action = seller1.respond(
            conversation_history=observation["conversation_history_seller1"],
            current_state=observation
        )
        
        # Get seller2's response
        seller2_action = seller2.respond(
            conversation_history=observation["conversation_history_seller2"],
            current_state=observation
        )
        
        # Execute step with all actions
        observation, reward, terminated, truncated, info = env.step(
            buyer_action_seller1=buyer_action_seller1,
            buyer_action_seller2=buyer_action_seller2,
            seller1_action=seller1_action,
            seller2_action=seller2_action
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
                print(f"Selected Seller: Seller {info['selected_seller']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
                # Display product info for selected seller
                if info['selected_seller'] == 1:
                    product_info = info.get('seller1_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
                elif info['selected_seller'] == 2:
                    product_info = info.get('seller2_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
            print(f"Seller1 Prices: Seller=${info.get('seller1_price', 0):.2f} | Buyer=${info.get('buyer_price_seller1', 0):.2f}")
            print(f"Seller2 Prices: Seller=${info.get('seller2_price', 0):.2f} | Buyer=${info.get('buyer_price_seller2', 0):.2f}")
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

