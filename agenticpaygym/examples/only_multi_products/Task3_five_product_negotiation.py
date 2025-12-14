"""Task3 Five-Product Negotiation Example

Demonstrates how to use the Task3FiveProductNegotiation to negotiate five products
with total price negotiation.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym import make  # Use registration system
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM


def main():
    """Main function: Demonstrates five-product negotiation flow"""
    
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
    # buyer_max_price and seller_min_price represent total expected cost for all five products
    print("Creating agents...")
    buyer_max_price = 450.0  # Maximum acceptable total purchase price for buyer (confidential, for all five products)
    seller_min_price = 350.0  # Minimum acceptable total selling price for seller (confidential, for all five products)
    
    buyer = BuyerAgent(llm=llm, buyer_max_price=buyer_max_price)
    seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)
    
    # Create environment using registration system
    print("Creating five-product negotiation environment...")
    env = make(
        "Task3_five_product_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=20,
        initial_seller_price=550.0,  # Initial total price offered by seller for all five products
        buyer_max_price=buyer_max_price,  # Buyer total max price (confidential, for all five products)
        seller_min_price=seller_min_price,  # Seller total min price (confidential, for all five products)
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
    
    # Define five products with their individual prices (each product has different price)
    # The product_info should contain a list of five products
    product_info = {
        "products": [
            {
                "name": "Premium Winter Jacket",
                "brand": "Mountain Gear",
                "price": 150.0,  # Individual price of first product
                "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                "condition": "New",
                "material": "Gore-Tex",
            },
            {
                "name": "Running Shoes",
                "brand": "SportMax",
                "price": 120.0,  # Individual price of second product
                "features": ["Lightweight", "Cushioned", "Breathable", "Durable"],
                "condition": "New",
                "material": "Mesh and Synthetic",
            },
            {
                "name": "Backpack",
                "brand": "Adventure Pro",
                "price": 80.0,  # Individual price of third product
                "features": ["Waterproof", "Multiple Compartments", "Ergonomic", "Lightweight"],
                "condition": "New",
                "material": "Nylon",
            },
            {
                "name": "Water Bottle",
                "brand": "HydroFlow",
                "price": 25.0,  # Individual price of fourth product
                "features": ["Insulated", "BPA-Free", "Leak-Proof", "Easy to Clean"],
                "condition": "New",
                "material": "Stainless Steel",
            },
            {
                "name": "Fitness Tracker",
                "brand": "FitTech",
                "price": 75.0,  # Individual price of fifth product
                "features": ["Heart Rate Monitor", "GPS", "Water Resistant", "Long Battery Life"],
                "condition": "New",
                "material": "Silicone and Plastic",
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts:")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Product Price: ${total_product_price:.2f}")
    
    # Get user requirement (should describe purchasing five products)
    print("\n" + "="*60)
    print("Please enter the product requirement (should describe purchasing five products):")
    user_requirement = input("> ").strip()
    if not user_requirement:
        print("No requirement entered, using default requirement...")
        user_requirement = "I need a complete outdoor gear set including a winter jacket, running shoes, a backpack, a water bottle, and a fitness tracker for my outdoor activities"
        print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting five-product negotiation...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Each round, both buyer and seller respond
        # Order: buyer -> seller
        # Get buyer's response first
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Get seller's response
        seller_action = seller.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Execute step with both actions
        observation, reward, terminated, truncated, info = env.step(
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
            print(f"Final Total Prices: Seller=${info.get('seller_price', 0):.2f} | Buyer=${info.get('buyer_price', 0):.2f}")
            if info.get('agreed_price'):
                print(f"Agreed Total Price: ${info.get('agreed_price', 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Reward: {reward:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Close environment
    env.close()
    print("\nFive-product negotiation completed!")


if __name__ == "__main__":
    main()

