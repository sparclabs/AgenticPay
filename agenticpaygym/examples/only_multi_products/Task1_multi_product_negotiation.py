"""Task1 Multi-Product Negotiation Example

Demonstrates how to use the Task1MultiProductNegotiation to negotiate multiple products
while preserving conversation context across different products.
"""

import os
import sys

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from agenticpaygym import make  # Use registration system
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
# from agenticpaygym.agents.product_selector_agent import ProductSelectorAgent  # Removed
from agenticpaygym.models.custom_llm import CustomLLM
from agenticpaygym.models.qwen3_vl import Qwen3VL
from agenticpaygym.models.vllm_vlm import VLLMVLM

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance, OPENAI_API_KEY


def main():
    """Main function: Demonstrates multi-product negotiation flow"""
    
    print("Initializing model...")
    
    # Check API key
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
    #     print("You can set it with: export OPENAI_API_KEY='your-key-here'")
    #     return
    
    model = CustomLLM(api_key=OPENAI_API_KEY, model="gpt-5.2") # gpt-4o-mini-2024-07-18, gpt-3.5-turbo

    # model_path = "/root/autodl-tmp/AgenticPayGym/agenticpaygym/models/download_models/Qwen3-VL-2B-Instruct"

    # model = VLLMVLM(
    #     model_path=model_path,
    #     trust_remote_code=True,
    #     gpu_memory_utilization=0.9,
    #     tensor_parallel_size=2,
    # )

    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    print("Creating agents...")
    buyer_max_price = 120.0  # Maximum acceptable purchase price for buyer (confidential)
    seller_min_price = 80.0  # Minimum acceptable selling price for seller (confidential)
    
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_price)
    # product_selector = ProductSelectorAgent(model=llm)  # Removed
    
    # Create environment using registration system
    print("Creating multi-product negotiation environment...")
    env = make(
        "Task1_multi_product_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds_per_product=max_rounds,
        initial_seller_price=150.0,  # Initial price offered by seller
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller_min_price=seller_min_price,  # Seller bottom price (confidential)
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
        print(f"Using default requirement: {user_requirement}")
    
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
        # Each round: buyer responds first, then seller responds (seeing buyer's message)
        # Get buyer's response
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Create updated conversation history that includes buyer's response
        # So seller can see buyer's message before responding
        updated_conversation_history = observation["conversation_history"].copy()
        if buyer_action:
            current_round = observation.get("current_round", 0)
            updated_conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })
        
        # Get seller's response (seller can now see buyer's message)
        seller_action = seller.respond(
            conversation_history=updated_conversation_history,
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
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
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
            seller_price = info.get('seller_price')
            buyer_price = info.get('buyer_price')
            seller_price_str = f"${seller_price:.2f}" if seller_price is not None else "Not specified"
            buyer_price_str = f"${buyer_price:.2f}" if buyer_price is not None else "Not specified"
            print(f"Final Prices: Seller={seller_price_str} | Buyer={buyer_price_str}")
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
    # Simple product selection: use the second product (Running Shoes) as default
    # ProductSelectorAgent has been removed, so we use a simple selection logic
    if "shoe" in user_requirement.lower() or "running" in user_requirement.lower() or "jogging" in user_requirement.lower():
        selected_product = products[1]  # Running Shoes
    else:
        selected_product = products[0]  # Default to first product
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
        # Each round: buyer responds first, then seller responds (seeing buyer's message)
        # Get buyer's response
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Create updated conversation history that includes buyer's response
        # So seller can see buyer's message before responding
        updated_conversation_history = observation["conversation_history"].copy()
        if buyer_action:
            current_round = observation.get("current_round", 0)
            updated_conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })
        
        # Get seller's response (seller can now see buyer's message)
        seller_action = seller.respond(
            conversation_history=updated_conversation_history,
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
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
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
            seller_price = info.get('seller_price')
            buyer_price = info.get('buyer_price')
            seller_price_str = f"${seller_price:.2f}" if seller_price is not None else "Not specified"
            buyer_price_str = f"${buyer_price:.2f}" if buyer_price is not None else "Not specified"
            print(f"Final Prices: Seller={seller_price_str} | Buyer={buyer_price_str}")
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

