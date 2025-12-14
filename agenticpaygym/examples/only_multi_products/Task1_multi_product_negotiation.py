"""Task1 Multi-Product Negotiation Example

Demonstrates how to use the Task1MultiProductNegotiation to negotiate multiple products
while preserving conversation context across different products.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym import make  # Use registration system
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.agents.product_selector_agent import ProductSelectorAgent
from agenticpaygym.llm.openai_llm import OpenAILLM


def main():
    """Main function: Demonstrates multi-product negotiation flow"""
    
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
    seller_min_price = 80.0  # Minimum acceptable selling price for seller (confidential)
    
    buyer = BuyerAgent(llm=llm, buyer_max_price=buyer_max_price)
    seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)
    product_selector = ProductSelectorAgent(llm=llm)
    
    # Create environment using registration system
    print("Creating multi-product negotiation environment...")
    env = make(
        "Task1_multi_product_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds_per_product=20,
        initial_seller_price=150.0,  # Initial price offered by seller
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller_min_price=seller_min_price,  # Seller bottom price (confidential)
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
    
    # Define available products (similar to simple_negotiation.py structure)
    products = [
        {
            "name": "Premium Winter Jacket",
            "brand": "Mountain Gear",
            "price": 180.0,  # The product's own price
            "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
            "condition": "New",
            "material": "Gore-Tex",
        },
        {
            "name": "Running Shoes",
            "brand": "SportMax",
            "price": 120.0,  # The product's own price
            "features": ["Lightweight", "Cushioned", "Breathable", "Durable"],
            "condition": "New",
            "material": "Mesh and Synthetic",
        },
    ]
    
    # First product negotiation (Winter Jacket)
    print("\n" + "="*60)
    print("Starting first product negotiation...")
    print("="*60)
    
    # Get user requirement
    print("\nPlease enter the product requirement you want to purchase:")
    user_requirement = input("> ").strip()
    if not user_requirement:
        print("No requirement entered, using default requirement...")
        user_requirement = "I need a high-quality winter jacket for cold weather"
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new negotiation...")
    print("="*60)
    
    # For first product, we can use a simple selection or default
    # Let's use the first product for the first negotiation
    selected_product = products[0]  # First product: Winter Jacket
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=selected_product,
        user_profile=user_profile,  # Pass user profile
        available_products=products,  # Pass all products to seller
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Each round, both buyer and seller respond
        # Get seller's response
        seller_action = seller.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Get buyer's response
        buyer_action = buyer.respond(
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
            print(f"Final Prices: Seller=${info.get('seller_price', 0):.2f} | Buyer=${info.get('buyer_price', 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Reward: {reward:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Ask if user wants to continue with another product
    print("\n" + "="*60)
    print("Please enter the product requirement you want to purchase:")
    user_requirement = input("> ").strip()
    
    # If user enters "no", exit
    if user_requirement.lower() == "no":
        # Display final summary and exit
        print("\n" + "="*60)
        print("All Products Negotiation Summary")
        print("="*60)
        
        final_info = env._get_info()
        if final_info.get("product_results"):
            print(f"\nTotal products negotiated: {len(final_info['product_results'])}")
            for i, result in enumerate(final_info["product_results"], 1):
                status_str = "✓ AGREED" if result["status"] == "agreed" else "✗ TIMEOUT"
                price_str = f"${result['agreed_price']:.2f}" if result.get('agreed_price') else "N/A"
                print(f"  {i}. {result['product_name']}: {status_str} @ {price_str} (Rounds: {result['rounds']})")
        else:
            print("No products were negotiated.")
        
        print("="*60)
        env.close()
        print("\nMulti-product negotiation completed!")
        return
    
    # If user entered a requirement, use it; otherwise use default
    if not user_requirement:
        print("No requirement entered, using default requirement...")
        user_requirement = "I need a high-quality pair of running shoes for jogging"
    
    # Select the most appropriate product based on user requirement
    print("\nSelecting the most appropriate product based on your requirement...")
    selected_product = product_selector.select_product(user_requirement, products)
    print(f"Selected product: {selected_product['name']} (${selected_product['price']:.2f})")
    
    # Second product negotiation
    print("\n" + "="*60)
    print("Starting second product negotiation...")
    print("="*60)
    
    # Reset environment for new product (preserves context)
    print("\n" + "="*60)
    print("Starting new negotiation...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=selected_product,  # Selected product based on requirement
        user_profile=user_profile,  # Pass user profile
        clear_history=False,  # Preserve previous context
        available_products=products,  # Pass all products to seller
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Each round, both buyer and seller respond
        # Get seller's response
        seller_action = seller.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Get buyer's response
        buyer_action = buyer.respond(
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
            print(f"Final Prices: Seller=${info.get('seller_price', 0):.2f} | Buyer=${info.get('buyer_price', 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Reward: {reward:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Display final summary
    print("\n" + "="*60)
    print("All Products Negotiation Summary")
    print("="*60)
    
    final_info = env._get_info()
    if final_info.get("product_results"):
        print(f"\nTotal products negotiated: {len(final_info['product_results'])}")
        for i, result in enumerate(final_info["product_results"], 1):
            status_str = "✓ AGREED" if result["status"] == "agreed" else "✗ TIMEOUT"
            price_str = f"${result['agreed_price']:.2f}" if result.get('agreed_price') else "N/A"
            print(f"  {i}. {result['product_name']}: {status_str} @ {price_str} (Rounds: {result['rounds']})")
    else:
        print("No products were negotiated.")
    
    print("="*60)
    
    # Close environment
    env.close()
    print("\nMulti-product negotiation completed!")


if __name__ == "__main__":
    main()

