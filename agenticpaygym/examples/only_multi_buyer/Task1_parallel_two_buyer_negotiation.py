"""Task1 Parallel Two-Buyer Negotiation Example

Demonstrates how to use the Task1ParallelTwoBuyerNegotiation to negotiate with two buyers
in parallel for the same product, and automatically choose the buyer with the higher price.
"""

import os
import sys

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from agenticpaygym.envs.only_multi_buyer.Task1_parallel_two_buyer_negotiation import Task1ParallelTwoBuyerNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance


def main():
    """Main function: Demonstrates multi-buyer negotiation flow"""
    
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
    buyer1_max_price = 120.0  # Maximum acceptable purchase price for buyer1 (confidential)
    buyer2_max_price = 130.0  # Maximum acceptable purchase price for buyer2 (confidential, different from buyer1)
    seller_min_price = 80.0  # Minimum acceptable selling price for seller (confidential)
    
    buyer1 = BuyerAgent(llm=llm, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(llm=llm, buyer_max_price=buyer2_max_price)
    seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)
    
    # Create environment
    print("Creating multi-buyer negotiation environment...")
    env = Task1ParallelTwoBuyerNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller_agent=seller,
        max_rounds=max_rounds,
        initial_seller_price=150.0,  # Initial price offered by seller
        buyer1_max_price=buyer1_max_price,  # Buyer1 bottom price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 bottom price (confidential)
        seller_min_price=seller_min_price,  # Seller bottom price (confidential)
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
    print("Starting new negotiation with two buyers...")
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
        # Each round, order is: buyer -> seller
        # Get buyer1's response first
        buyer1_action = buyer1.respond(
            conversation_history=observation["conversation_history_buyer1"],
            current_state=observation
        )
        
        # Get buyer2's response
        buyer2_action = buyer2.respond(
            conversation_history=observation["conversation_history_buyer2"],
            current_state=observation
        )
        
        # Then get seller's response to buyer1
        seller_action_buyer1 = seller.respond(
            conversation_history=observation["conversation_history_buyer1"],
            current_state=observation
        )
        
        # Get seller's response to buyer2
        seller_action_buyer2 = seller.respond(
            conversation_history=observation["conversation_history_buyer2"],
            current_state=observation
        )
        
        # Execute step with all actions
        observation, reward, terminated, truncated, info = env.step(
            buyer1_action=buyer1_action,
            buyer2_action=buyer2_action,
            seller_action_buyer1=seller_action_buyer1,
            seller_action_buyer2=seller_action_buyer2
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Display step rewards for each round with detailed calculation
        if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_seller_reward' in info:
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer1_reward' in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if 'step_buyer2_reward' in info:
                if 'step_buyer1_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if 'step_seller_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller: {info['step_seller_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer1 step reward details
            if 'step_buyer1_reward' in info and info.get('buyer1_price') is not None:
                buyer1_price = info.get('buyer1_price', 0)
                buyer1_max = env.buyer1_max_price
                if buyer1_max is not None:
                    buyer1_savings = buyer1_max - buyer1_price
                    weighted_savings = buyer1_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer1_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_price={buyer1_price:.2f}, buyer1_max={buyer1_max}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer1_price={buyer1_price:.2f}, buyer1_max not specified, round={info['round']})")
            elif 'step_buyer1_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer1_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info and info.get('buyer2_price') is not None:
                buyer2_price = info.get('buyer2_price', 0)
                buyer2_max = env.buyer2_max_price
                if buyer2_max is not None:
                    buyer2_savings = buyer2_max - buyer2_price
                    weighted_savings = buyer2_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer2_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_price={buyer2_price:.2f}, buyer2_max={buyer2_max}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer2_price={buyer2_price:.2f}, buyer2_max not specified, round={info['round']})")
            elif 'step_buyer2_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer2_price not specified, round={info['round']})")
            
            # Seller step reward details
            if 'step_seller_reward' in info:
                seller_rewards_detail = []
                if info.get('seller_price_buyer1') is not None and env.seller_min_price is not None:
                    seller_price_b1 = info.get('seller_price_buyer1', 0)
                    seller_profit_b1 = seller_price_b1 - env.seller_min_price
                    weighted_profit_b1 = seller_profit_b1 * weights["seller_profit"]
                    seller_rewards_detail.append(f"seller_profit_b1({seller_profit_b1:.2f} * {weights['seller_profit']:.2f})={weighted_profit_b1:.2f}")
                
                if info.get('seller_price_buyer2') is not None and env.seller_min_price is not None:
                    seller_price_b2 = info.get('seller_price_buyer2', 0)
                    seller_profit_b2 = seller_price_b2 - env.seller_min_price
                    weighted_profit_b2 = seller_profit_b2 * weights["seller_profit"]
                    seller_rewards_detail.append(f"seller_profit_b2({seller_profit_b2:.2f} * {weights['seller_profit']:.2f})={weighted_profit_b2:.2f}")
                
                if seller_rewards_detail:
                    aggregated_detail = f"aggregated({env.seller_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = {aggregated_detail}[{', '.join(seller_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller_reward']:.2f} (seller_min={env.seller_min_price}, round={info['round']}, aggregation={env.seller_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer'):
                print(f"Selected Buyer: Buyer {info['selected_buyer']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
            print(f"Buyer1 Prices: Buyer=${info.get('buyer1_price', 0):.2f} | Seller=${info.get('seller_price_buyer1', 0):.2f}")
            print(f"Buyer2 Prices: Buyer=${info.get('buyer2_price', 0):.2f} | Seller=${info.get('seller_price_buyer2', 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer1_reward' in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if 'buyer2_reward' in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if 'seller_reward' in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Close environment
    env.close()
    print("\nNegotiation completed!")


if __name__ == "__main__":
    main()

