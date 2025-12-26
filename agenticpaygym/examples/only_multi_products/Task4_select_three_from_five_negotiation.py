"""Task4 Select Three from Five Products Negotiation Example

Demonstrates how to use the Task4SelectThreeFromFiveNegotiation where user needs 3 products,
and buyer automatically selects 3 from 5 available products for total price negotiation.
"""

import os
import sys

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from agenticpaygym import make  # Use registration system
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.models.custom_llm import CustomLLM

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance


def main():
    """Main function: Demonstrates select-three-from-five negotiation flow"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize LLM
    print("Initializing LLM...")
    llm = CustomLLM(api_key=api_key, model="gpt-4o-mini-2024-07-18")  # gpt-4o-mini-2024-07-18, gpt-3.5-turbo
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price represent total expected cost for the selected 3 products
    print("Creating agents...")
    buyer_max_price = 300.0  # Maximum acceptable total purchase price for buyer (confidential, for selected 3 products)
    seller_min_price = 250.0  # Minimum acceptable total selling price for seller (confidential, for selected 3 products)
    
    buyer = BuyerAgent(llm=llm, buyer_max_price=buyer_max_price)
    seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)
    
    # Create environment using registration system
    print("Creating select-three-from-five negotiation environment...")
    env = make(
        "Task4_select_three_from_five_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=max_rounds,
        initial_seller_price=350.0,  # Initial total price offered by seller for selected 3 products
        buyer_max_price=buyer_max_price,  # Buyer total max price (confidential, for selected 3 products)
        seller_min_price=seller_min_price,  # Seller total min price (confidential, for selected 3 products)
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
    print(f"User Profile: {user_profile}")
    
    # Define five products with their individual prices (all 5 products meet the requirement for 3 products)
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
    print(f"\nAvailable Products (5 total, user needs 3):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Price (all 5): ${total_product_price:.2f}")
    print(f"\nNote: Buyer will automatically select 3 products based on user requirement.")
    
    # Get user requirement (should describe needing 3 products)
    print("\n" + "="*60)
    print("Please enter the product requirement (should describe needing 3 products):")
    user_requirement = input("> ").strip()
    if not user_requirement:
        print("No requirement entered, using default requirement...")
        user_requirement = "I need 3 items for my outdoor activities: a jacket, shoes, and a backpack"
        print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting select-three-from-five negotiation...")
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
        # Get buyer's response first (buyer will automatically select 3 products based on user_requirement and product_info)
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
        
        # Display step rewards for each round with detailed calculation
        if 'step_seller_reward' in info or 'step_buyer_reward' in info:
            print(f"\n[Step Rewards] ", end="")
            if 'step_seller_reward' in info:
                print(f"Seller: {info['step_seller_reward']:.3f}", end="")
            if 'step_buyer_reward' in info:
                if 'step_seller_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer: {info['step_buyer_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            if 'step_seller_reward' in info and info.get('seller_price') is not None:
                seller_price = info.get('seller_price', 0)
                seller_min = env.seller_min_price
                if seller_min is not None:
                    seller_profit = seller_price - seller_min
                    weighted_seller_profit = seller_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = seller_profit({seller_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller_reward']:.2f} (seller_price={seller_price:.2f}, seller_min={seller_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price={seller_price:.2f}, seller_min not specified, round={info['round']})")
            elif 'step_seller_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
            
            if 'step_buyer_reward' in info and info.get('buyer_price') is not None:
                buyer_price = info.get('buyer_price', 0)
                buyer_max = env.buyer_max_price
                if buyer_max is not None:
                    buyer_savings = buyer_max - buyer_price
                    weighted_buyer_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer_reward']:.2f} (buyer_max={buyer_max}, buyer_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price={buyer_price:.2f}, buyer_max not specified, round={info['round']})")
            elif 'step_buyer_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Buyer Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            print(f"Final Total Prices (for selected 3 products): Seller=${info.get('seller_price', 0):.2f} | Buyer=${info.get('buyer_price', 0):.2f}")
            if info.get('agreed_price'):
                print(f"Agreed Total Price (for selected 3 products): ${info.get('agreed_price', 0):.2f}")
            if info.get('selected_products'):
                print(f"\nSelected Products:")
                for i, p in enumerate(info['selected_products'], 1):
                    print(f"  {i}. {p.get('name', 'Unknown')}: ${p.get('price', 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Total Reward: {reward:.3f}")
            if 'seller_reward' in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
            if 'buyer_reward' in info:
                print(f"Buyer Reward: {info['buyer_reward']:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Close environment
    env.close()
    print("\nSelect-three-from-five negotiation completed!")


if __name__ == "__main__":
    main()

