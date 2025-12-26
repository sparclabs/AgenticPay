"""Task2 Parallel Three-Buyer Three-Seller Two-Product Negotiation Example

Demonstrates how to use the Task2ParallelThreeBuyerThreeSellerTwoProductNegotiation to negotiate with
three buyers and three sellers in parallel for two products.
Prices represent total price for both products.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.multi_buyer_multi_products_multi_seller.Task2_parallel_three_buyer_three_seller_two_product_negotiation import Task2ParallelThreeBuyerThreeSellerTwoProductNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance


def main():
    """Main function: Demonstrates multi-buyer multi-seller multi-product negotiation flow"""
    
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
    # buyer_max_price and seller_min_price represent total expected cost for both products
    print("Creating agents...")
    buyer1_max_price = 200.0  # Maximum acceptable total purchase price for buyer1 (confidential, for both products)
    buyer2_max_price = 220.0  # Maximum acceptable total purchase price for buyer2 (confidential, for both products)
    buyer3_max_price = 230.0  # Maximum acceptable total purchase price for buyer3 (confidential, for both products)
    seller1_min_price = 150.0  # Minimum acceptable total selling price for seller1 (confidential, for both products)
    seller2_min_price = 160.0  # Minimum acceptable total selling price for seller2 (confidential, for both products)
    seller3_min_price = 170.0  # Minimum acceptable total selling price for seller3 (confidential, for both products)
    
    buyer1 = BuyerAgent(llm=llm, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(llm=llm, buyer_max_price=buyer2_max_price)
    buyer3 = BuyerAgent(llm=llm, buyer_max_price=buyer3_max_price)
    seller1 = SellerAgent(llm=llm, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(llm=llm, seller_min_price=seller2_min_price)
    seller3 = SellerAgent(llm=llm, seller_min_price=seller3_min_price)
    
    # Create environment
    print("Creating multi-buyer multi-seller multi-product negotiation environment...")
    env = Task2ParallelThreeBuyerThreeSellerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        buyer3_agent=buyer3,
        seller1_agent=seller1,
        seller2_agent=seller2,
        seller3_agent=seller3,
        max_rounds=max_rounds,
        initial_seller1_price=250.0,  # Initial total price offered by seller1 for both products
        initial_seller2_price=260.0,  # Initial total price offered by seller2 for both products
        initial_seller3_price=270.0,  # Initial total price offered by seller3 for both products
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential, for both products)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential, for both products)
        buyer3_max_price=buyer3_max_price,  # Buyer3 total max price (confidential, for both products)
        seller1_min_price=seller1_min_price,  # Seller1 total min price (confidential, for both products)
        seller2_min_price=seller2_min_price,  # Seller2 total min price (confidential, for both products)
        seller3_min_price=seller3_min_price,  # Seller3 total min price (confidential, for both products)
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
        buyer_reward_aggregation=buyer_reward_aggregation,  # Buyer reward aggregation method
        seller_reward_aggregation=seller_reward_aggregation,  # Seller reward aggregation method
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
    print(f"User Profile: {user_profile}")
    
    # Define two products with their individual prices
    # The product_info should contain a list of two products
    product_info = {
        "products": [
            {
                "name": "Premium Winter Jacket",
                "brand": "Mountain Gear",
                "price": 120.0,  # Individual price of first product
                "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                "condition": "New",
                "material": "Gore-Tex",
            },
            {
                "name": "Running Shoes",
                "brand": "SportMax",
                "price": 80.0,  # Individual price of second product
                "features": ["Lightweight", "Cushioned", "Breathable", "Durable"],
                "condition": "New",
                "material": "Mesh and Synthetic",
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts:")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Product Price: ${total_product_price:.2f}")
    
    # Get user requirement (should describe purchasing two products)
    print("\n" + "="*60)
    print("Please enter the product requirement (should describe purchasing two products):")
    user_requirement = input("> ").strip()
    if not user_requirement:
        print("No requirement entered, using default requirement...")
        user_requirement = "I need a high-quality winter jacket and a pair of running shoes for my outdoor activities"
        print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new negotiation with three buyers and three sellers for two products...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Each round, order is: buyer -> seller
        # Get buyer1's responses
        buyer1_action_seller1 = buyer1.respond(
            conversation_history=observation["conversation_history_b1s1"],
            current_state=observation
        )
        
        buyer1_action_seller2 = buyer1.respond(
            conversation_history=observation["conversation_history_b1s2"],
            current_state=observation
        )
        
        buyer1_action_seller3 = buyer1.respond(
            conversation_history=observation["conversation_history_b1s3"],
            current_state=observation
        )
        
        # Get buyer2's responses
        buyer2_action_seller1 = buyer2.respond(
            conversation_history=observation["conversation_history_b2s1"],
            current_state=observation
        )
        
        buyer2_action_seller2 = buyer2.respond(
            conversation_history=observation["conversation_history_b2s2"],
            current_state=observation
        )
        
        buyer2_action_seller3 = buyer2.respond(
            conversation_history=observation["conversation_history_b2s3"],
            current_state=observation
        )
        
        # Get buyer3's responses
        buyer3_action_seller1 = buyer3.respond(
            conversation_history=observation["conversation_history_b3s1"],
            current_state=observation
        )
        
        buyer3_action_seller2 = buyer3.respond(
            conversation_history=observation["conversation_history_b3s2"],
            current_state=observation
        )
        
        buyer3_action_seller3 = buyer3.respond(
            conversation_history=observation["conversation_history_b3s3"],
            current_state=observation
        )
        
        # Get seller1's responses
        seller1_action_buyer1 = seller1.respond(
            conversation_history=observation["conversation_history_b1s1"],
            current_state=observation
        )
        
        seller1_action_buyer2 = seller1.respond(
            conversation_history=observation["conversation_history_b2s1"],
            current_state=observation
        )
        
        seller1_action_buyer3 = seller1.respond(
            conversation_history=observation["conversation_history_b3s1"],
            current_state=observation
        )
        
        # Get seller2's responses
        seller2_action_buyer1 = seller2.respond(
            conversation_history=observation["conversation_history_b1s2"],
            current_state=observation
        )
        
        seller2_action_buyer2 = seller2.respond(
            conversation_history=observation["conversation_history_b2s2"],
            current_state=observation
        )
        
        seller2_action_buyer3 = seller2.respond(
            conversation_history=observation["conversation_history_b3s2"],
            current_state=observation
        )
        
        # Get seller3's responses
        seller3_action_buyer1 = seller3.respond(
            conversation_history=observation["conversation_history_b1s3"],
            current_state=observation
        )
        
        seller3_action_buyer2 = seller3.respond(
            conversation_history=observation["conversation_history_b2s3"],
            current_state=observation
        )
        
        seller3_action_buyer3 = seller3.respond(
            conversation_history=observation["conversation_history_b3s3"],
            current_state=observation
        )
        
        # Execute step with all actions
        observation, reward, terminated, truncated, info = env.step(
            buyer1_action_seller1=buyer1_action_seller1,
            buyer1_action_seller2=buyer1_action_seller2,
            buyer1_action_seller3=buyer1_action_seller3,
            buyer2_action_seller1=buyer2_action_seller1,
            buyer2_action_seller2=buyer2_action_seller2,
            buyer2_action_seller3=buyer2_action_seller3,
            buyer3_action_seller1=buyer3_action_seller1,
            buyer3_action_seller2=buyer3_action_seller2,
            buyer3_action_seller3=buyer3_action_seller3,
            seller1_action_buyer1=seller1_action_buyer1,
            seller1_action_buyer2=seller1_action_buyer2,
            seller1_action_buyer3=seller1_action_buyer3,
            seller2_action_buyer1=seller2_action_buyer1,
            seller2_action_buyer2=seller2_action_buyer2,
            seller2_action_buyer3=seller2_action_buyer3,
            seller3_action_buyer1=seller3_action_buyer1,
            seller3_action_buyer2=seller3_action_buyer2,
            seller3_action_buyer3=seller3_action_buyer3
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Display step rewards for each round with detailed calculation
        if ('step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or
            'step_seller1_reward' in info or 'step_seller2_reward' in info or 'step_seller3_reward' in info):
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer1_reward' in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if 'step_buyer2_reward' in info:
                if 'step_buyer1_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if 'step_buyer3_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer3: {info['step_buyer3_reward']:.3f}", end="")
            if 'step_seller1_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info:
                    print(f" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if 'step_seller2_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or 'step_seller1_reward' in info:
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            if 'step_seller3_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or 'step_seller1_reward' in info or 'step_seller2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller3: {info['step_seller3_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer1 step reward details
            if 'step_buyer1_reward' in info:
                buyer_rewards_detail = []
                for seller_id in [1, 2, 3]:
                    price_key = f'b1s{seller_id}_buyer_price'
                    if info.get(price_key) is not None and env.buyer1_max_price is not None:
                        buyer_price = info.get(price_key, 0)
                        buyer_savings = env.buyer1_max_price - buyer_price
                        weighted_savings = buyer_savings * weights["buyer_savings"]
                        buyer_rewards_detail.append(f"buyer_savings_s{seller_id}({buyer_savings:.2f} * {weights['buyer_savings']:.2f})={weighted_savings:.2f}")
                
                if buyer_rewards_detail:
                    aggregated_detail = f"aggregated({env.buyer_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = {aggregated_detail}[{', '.join(buyer_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info:
                buyer_rewards_detail = []
                for seller_id in [1, 2, 3]:
                    price_key = f'b2s{seller_id}_buyer_price'
                    if info.get(price_key) is not None and env.buyer2_max_price is not None:
                        buyer_price = info.get(price_key, 0)
                        buyer_savings = env.buyer2_max_price - buyer_price
                        weighted_savings = buyer_savings * weights["buyer_savings"]
                        buyer_rewards_detail.append(f"buyer_savings_s{seller_id}({buyer_savings:.2f} * {weights['buyer_savings']:.2f})={weighted_savings:.2f}")
                
                if buyer_rewards_detail:
                    aggregated_detail = f"aggregated({env.buyer_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = {aggregated_detail}[{', '.join(buyer_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Buyer3 step reward details
            if 'step_buyer3_reward' in info:
                buyer_rewards_detail = []
                for seller_id in [1, 2, 3]:
                    price_key = f'b3s{seller_id}_buyer_price'
                    if info.get(price_key) is not None and env.buyer3_max_price is not None:
                        buyer_price = info.get(price_key, 0)
                        buyer_savings = env.buyer3_max_price - buyer_price
                        weighted_savings = buyer_savings * weights["buyer_savings"]
                        buyer_rewards_detail.append(f"buyer_savings_s{seller_id}({buyer_savings:.2f} * {weights['buyer_savings']:.2f})={weighted_savings:.2f}")
                
                if buyer_rewards_detail:
                    aggregated_detail = f"aggregated({env.buyer_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer3 Step Reward = {aggregated_detail}[{', '.join(buyer_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer3_reward']:.2f} (buyer3_max={env.buyer3_max_price}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer3 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info:
                seller_rewards_detail = []
                for buyer_id in [1, 2, 3]:
                    price_key = f'b{buyer_id}s1_seller_price'
                    if info.get(price_key) is not None and env.seller1_min_price is not None:
                        seller_price = info.get(price_key, 0)
                        seller_profit = seller_price - env.seller1_min_price
                        weighted_profit = seller_profit * weights["seller_profit"]
                        seller_rewards_detail.append(f"seller_profit_b{buyer_id}({seller_profit:.2f} * {weights['seller_profit']:.2f})={weighted_profit:.2f}")
                
                if seller_rewards_detail:
                    aggregated_detail = f"aggregated({env.seller_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = {aggregated_detail}[{', '.join(seller_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_min={env.seller1_min_price}, round={info['round']}, aggregation={env.seller_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
            
            # Seller2 step reward details
            if 'step_seller2_reward' in info:
                seller_rewards_detail = []
                for buyer_id in [1, 2, 3]:
                    price_key = f'b{buyer_id}s2_seller_price'
                    if info.get(price_key) is not None and env.seller2_min_price is not None:
                        seller_price = info.get(price_key, 0)
                        seller_profit = seller_price - env.seller2_min_price
                        weighted_profit = seller_profit * weights["seller_profit"]
                        seller_rewards_detail.append(f"seller_profit_b{buyer_id}({seller_profit:.2f} * {weights['seller_profit']:.2f})={weighted_profit:.2f}")
                
                if seller_rewards_detail:
                    aggregated_detail = f"aggregated({env.seller_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = {aggregated_detail}[{', '.join(seller_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_min={env.seller2_min_price}, round={info['round']}, aggregation={env.seller_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
            
            # Seller3 step reward details
            if 'step_seller3_reward' in info:
                seller_rewards_detail = []
                for buyer_id in [1, 2, 3]:
                    price_key = f'b{buyer_id}s3_seller_price'
                    if info.get(price_key) is not None and env.seller3_min_price is not None:
                        seller_price = info.get(price_key, 0)
                        seller_profit = seller_price - env.seller3_min_price
                        weighted_profit = seller_profit * weights["seller_profit"]
                        seller_rewards_detail.append(f"seller_profit_b{buyer_id}({seller_profit:.2f} * {weights['seller_profit']:.2f})={weighted_profit:.2f}")
                
                if seller_rewards_detail:
                    aggregated_detail = f"aggregated({env.seller_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller3 Step Reward = {aggregated_detail}[{', '.join(seller_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller3_reward']:.2f} (seller3_min={env.seller3_min_price}, round={info['round']}, aggregation={env.seller_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller3 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer') and info.get('selected_seller'):
                print(f"Selected Deal: Buyer {info['selected_buyer']} - Seller {info['selected_seller']}")
                print(f"Final Deal Total Price: ${info.get('final_deal_price', 0):.2f}")
            # Display all 9 buyer-seller pairs
            for buyer_id in [1, 2, 3]:
                for seller_id in [1, 2, 3]:
                    buyer_price_key = f'b{buyer_id}s{seller_id}_buyer_price'
                    seller_price_key = f'b{buyer_id}s{seller_id}_seller_price'
                    print(f"Buyer{buyer_id}-Seller{seller_id} Total Prices: Buyer=${info.get(buyer_price_key, 0):.2f} | Seller=${info.get(seller_price_key, 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer1_reward' in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if 'buyer2_reward' in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if 'buyer3_reward' in info:
                print(f"Buyer3 Reward: {info['buyer3_reward']:.3f}")
            if 'seller1_reward' in info:
                print(f"Seller1 Reward: {info['seller1_reward']:.3f}")
            if 'seller2_reward' in info:
                print(f"Seller2 Reward: {info['seller2_reward']:.3f}")
            if 'seller3_reward' in info:
                print(f"Seller3 Reward: {info['seller3_reward']:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Close environment
    env.close()
    print("\nMulti-buyer multi-seller multi-product negotiation completed!")


if __name__ == "__main__":
    main()

