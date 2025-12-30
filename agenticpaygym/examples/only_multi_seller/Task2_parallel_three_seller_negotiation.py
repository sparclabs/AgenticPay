"""Task2 Parallel Three-Seller Negotiation Example

Demonstrates how to use the Task2ParallelThreeSellerNegotiation to negotiate with three sellers
in parallel for the same product, and choose the seller with the lowest price.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.only_multi_seller.Task2_parallel_three_seller_negotiation import Task2ParallelThreeSellerNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.models.custom_llm import CustomLLM
from agenticpaygym.examples.config import reward_weights, max_rounds, price_tolerance, buyer_reward_aggregation, seller_reward_aggregation, OPENAI_API_KEY


def main():
    """Main function: Demonstrates multi-seller negotiation flow with three sellers"""
    
    print("Initializing model...")
    
    # Check API key
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
    #     print("You can set it with: export OPENAI_API_KEY='your-key-here'")
    #     return
    
    model = CustomLLM(api_key=OPENAI_API_KEY, model="gpt-5.2")  # gpt-4o-mini-2024-07-18, gpt-3.5-turbo
    
    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    print("Creating agents...")
    buyer_max_price = 120.0  # Maximum acceptable purchase price for buyer (confidential)
    seller1_min_price = 80.0  # Minimum acceptable selling price for seller1 (confidential)
    seller2_min_price = 85.0  # Minimum acceptable selling price for seller2 (confidential, different from seller1)
    seller3_min_price = 90.0  # Minimum acceptable selling price for seller3 (confidential, different from seller1 and seller2)
    
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(model=model, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, seller_min_price=seller2_min_price)
    seller3 = SellerAgent(model=model, seller_min_price=seller3_min_price)
    
    # Create environment
    print("Creating multi-seller negotiation environment with three sellers...")
    env = Task2ParallelThreeSellerNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        seller3_agent=seller3,
        max_rounds=max_rounds,
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
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
        buyer_reward_aggregation=buyer_reward_aggregation,  # Buyer reward aggregation method
        seller_reward_aggregation=seller_reward_aggregation,  # Seller reward aggregation method
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
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new negotiation with three sellers...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info={
            "name": "Premium Winter Jacket",
            "brand": "Mountain Gear",
            "price": 180.0,  # The product's own price
            "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
            "condition": "New",
            "material": "Gore-Tex",
        },
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Each round: buyer responds first, then sellers respond (seeing buyer's messages)
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
        
        # Get buyer's response to seller3
        buyer_action_seller3 = buyer.respond(
            conversation_history=observation["conversation_history_seller3"],
            current_state=observation
        )
        
        # Create updated conversation histories that include buyer's responses
        # So sellers can see buyer's messages before responding
        updated_conversation_history_seller1 = observation["conversation_history_seller1"].copy()
        updated_conversation_history_seller2 = observation["conversation_history_seller2"].copy()
        updated_conversation_history_seller3 = observation["conversation_history_seller3"].copy()
        
        if buyer_action_seller1:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_seller1.append({
                "role": "buyer",
                "content": buyer_action_seller1,
                "round": current_round
            })
        
        if buyer_action_seller2:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_seller2.append({
                "role": "buyer",
                "content": buyer_action_seller2,
                "round": current_round
            })
        
        if buyer_action_seller3:
            current_round = observation.get("current_round", 0)
            updated_conversation_history_seller3.append({
                "role": "buyer",
                "content": buyer_action_seller3,
                "round": current_round
            })
        
        # Then get seller1's response (seller1 can now see buyer's message)
        seller1_action = seller1.respond(
            conversation_history=updated_conversation_history_seller1,
            current_state=observation
        )
        
        # Get seller2's response (seller2 can now see buyer's message)
        seller2_action = seller2.respond(
            conversation_history=updated_conversation_history_seller2,
            current_state=observation
        )
        
        # Get seller3's response (seller3 can now see buyer's message)
        seller3_action = seller3.respond(
            conversation_history=updated_conversation_history_seller3,
            current_state=observation
        )
        
        # Execute step with all actions
        observation, reward, terminated, truncated, info = env.step(
            buyer_action_seller1=buyer_action_seller1,
            buyer_action_seller2=buyer_action_seller2,
            buyer_action_seller3=buyer_action_seller3,
            seller1_action=seller1_action,
            seller2_action=seller2_action,
            seller3_action=seller3_action
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # Display step rewards for each round with detailed calculation
        if 'step_buyer_reward' in info or 'step_seller1_reward' in info or 'step_seller2_reward' in info or 'step_seller3_reward' in info:
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer_reward' in info:
                print(f"Buyer: {info['step_buyer_reward']:.3f}", end="")
            if 'step_seller1_reward' in info:
                if 'step_buyer_reward' in info:
                    print(f" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if 'step_seller2_reward' in info:
                if 'step_buyer_reward' in info or 'step_seller1_reward' in info:
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            if 'step_seller3_reward' in info:
                if 'step_buyer_reward' in info or 'step_seller1_reward' in info or 'step_seller2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller3: {info['step_seller3_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer step reward details
            if 'step_buyer_reward' in info:
                buyer_rewards_detail = []
                if info.get('buyer_price_seller1') is not None and env.buyer_max_price is not None:
                    buyer_price_s1 = info.get('buyer_price_seller1', 0)
                    buyer_savings_s1 = env.buyer_max_price - buyer_price_s1
                    weighted_savings_s1 = buyer_savings_s1 * weights["buyer_savings"]
                    buyer_rewards_detail.append(f"buyer_savings_s1({buyer_savings_s1:.2f} * {weights['buyer_savings']:.2f})={weighted_savings_s1:.2f}")
                
                if info.get('buyer_price_seller2') is not None and env.buyer_max_price is not None:
                    buyer_price_s2 = info.get('buyer_price_seller2', 0)
                    buyer_savings_s2 = env.buyer_max_price - buyer_price_s2
                    weighted_savings_s2 = buyer_savings_s2 * weights["buyer_savings"]
                    buyer_rewards_detail.append(f"buyer_savings_s2({buyer_savings_s2:.2f} * {weights['buyer_savings']:.2f})={weighted_savings_s2:.2f}")
                
                if info.get('buyer_price_seller3') is not None and env.buyer_max_price is not None:
                    buyer_price_s3 = info.get('buyer_price_seller3', 0)
                    buyer_savings_s3 = env.buyer_max_price - buyer_price_s3
                    weighted_savings_s3 = buyer_savings_s3 * weights["buyer_savings"]
                    buyer_rewards_detail.append(f"buyer_savings_s3({buyer_savings_s3:.2f} * {weights['buyer_savings']:.2f})={weighted_savings_s3:.2f}")
                
                if buyer_rewards_detail:
                    aggregated_detail = f"aggregated({env.buyer_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = {aggregated_detail}[{', '.join(buyer_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer_reward']:.2f} (buyer_max={env.buyer_max_price}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info and info.get('seller1_price') is not None:
                seller1_price = info.get('seller1_price', 0)
                seller1_min = env.seller1_min_price
                if seller1_min is not None:
                    seller1_profit = seller1_price - seller1_min
                    weighted_seller1_profit = seller1_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_price={seller1_price:.2f}, seller1_min={seller1_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price={seller1_price:.2f}, seller1_min not specified, round={info['round']})")
            elif 'step_seller1_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price not specified, round={info['round']})")
            
            # Seller2 step reward details
            if 'step_seller2_reward' in info and info.get('seller2_price') is not None:
                seller2_price = info.get('seller2_price', 0)
                seller2_min = env.seller2_min_price
                if seller2_min is not None:
                    seller2_profit = seller2_price - seller2_min
                    weighted_seller2_profit = seller2_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_price={seller2_price:.2f}, seller2_min={seller2_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price={seller2_price:.2f}, seller2_min not specified, round={info['round']})")
            elif 'step_seller2_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")
            
            # Seller3 step reward details
            if 'step_seller3_reward' in info and info.get('seller3_price') is not None:
                seller3_price = info.get('seller3_price', 0)
                seller3_min = env.seller3_min_price
                if seller3_min is not None:
                    seller3_profit = seller3_price - seller3_min
                    weighted_seller3_profit = seller3_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller3 Step Reward = seller_profit({seller3_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller3_reward']:.2f} (seller3_price={seller3_price:.2f}, seller3_min={seller3_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller3 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller3_price={seller3_price:.2f}, seller3_min not specified, round={info['round']})")
            elif 'step_seller3_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller3 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller3_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_seller'):
                print(f"Selected Seller: Seller {info['selected_seller']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
            print(f"Seller1 Prices: Seller=${info.get('seller1_price', 0):.2f} | Buyer=${info.get('buyer_price_seller1', 0):.2f}")
            print(f"Seller2 Prices: Seller=${info.get('seller2_price', 0):.2f} | Buyer=${info.get('buyer_price_seller2', 0):.2f}")
            print(f"Seller3 Prices: Seller=${info.get('seller3_price', 0):.2f} | Buyer=${info.get('buyer_price_seller3', 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer_reward' in info:
                print(f"Buyer Reward: {info['buyer_reward']:.3f}")
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
    print("\nNegotiation completed!")


if __name__ == "__main__":
    main()
