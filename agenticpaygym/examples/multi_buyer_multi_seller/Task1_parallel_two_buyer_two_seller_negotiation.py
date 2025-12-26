"""Task1 Parallel Two-Buyer Two-Seller Negotiation Example

Demonstrates how to use the Task1ParallelTwoBuyerTwoSellerNegotiation to negotiate with
two buyers and two sellers in parallel for the same product.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.multi_buyer_multi_seller.Task1_parallel_two_buyer_two_seller_negotiation import Task1ParallelTwoBuyerTwoSellerNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.models.custom_llm import CustomLLM

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance


def main():
    """Main function: Demonstrates multi-buyer multi-seller negotiation flow"""
    
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
    print("Creating agents...")
    buyer1_max_price = 120.0  # Maximum acceptable purchase price for buyer1 (confidential)
    buyer2_max_price = 130.0  # Maximum acceptable purchase price for buyer2 (confidential)
    seller1_min_price = 80.0  # Minimum acceptable selling price for seller1 (confidential)
    seller2_min_price = 85.0  # Minimum acceptable selling price for seller2 (confidential)
    
    buyer1 = BuyerAgent(llm=llm, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(llm=llm, buyer_max_price=buyer2_max_price)
    seller1 = SellerAgent(llm=llm, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(llm=llm, seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating multi-buyer multi-seller negotiation environment...")
    env = Task1ParallelTwoBuyerTwoSellerNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=150.0,  # Initial price offered by seller1
        initial_seller2_price=160.0,  # Initial price offered by seller2
        buyer1_max_price=buyer1_max_price,  # Buyer1 bottom price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 bottom price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 bottom price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 bottom price (confidential)
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
    print("Starting new negotiation with two buyers and two sellers...")
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
        # Get buyer1's responses
        buyer1_action_seller1 = buyer1.respond(
            conversation_history=observation["conversation_history_b1s1"],
            current_state=observation
        )
        
        buyer1_action_seller2 = buyer1.respond(
            conversation_history=observation["conversation_history_b1s2"],
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
        
        # Get seller1's responses
        seller1_action_buyer1 = seller1.respond(
            conversation_history=observation["conversation_history_b1s1"],
            current_state=observation
        )
        
        seller1_action_buyer2 = seller1.respond(
            conversation_history=observation["conversation_history_b2s1"],
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
        
        # Execute step with all actions
        observation, reward, terminated, truncated, info = env.step(
            buyer1_action_seller1=buyer1_action_seller1,
            buyer1_action_seller2=buyer1_action_seller2,
            buyer2_action_seller1=buyer2_action_seller1,
            buyer2_action_seller2=buyer2_action_seller2,
            seller1_action_buyer1=seller1_action_buyer1,
            seller1_action_buyer2=seller1_action_buyer2,
            seller2_action_buyer1=seller2_action_buyer1,
            seller2_action_buyer2=seller2_action_buyer2
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Display step rewards for each round with detailed calculation
        if ('step_buyer1_reward' in info or 'step_buyer2_reward' in info or
            'step_seller1_reward' in info or 'step_seller2_reward' in info):
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer1_reward' in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if 'step_buyer2_reward' in info:
                if 'step_buyer1_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if 'step_seller1_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if 'step_seller2_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_seller1_reward' in info:
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer1 step reward details
            if 'step_buyer1_reward' in info:
                buyer_rewards_detail = []
                if info.get('b1s1_buyer_price') is not None and env.buyer1_max_price is not None:
                    buyer_price_s1 = info.get('b1s1_buyer_price', 0)
                    buyer_savings_s1 = env.buyer1_max_price - buyer_price_s1
                    weighted_savings_s1 = buyer_savings_s1 * weights["buyer_savings"]
                    buyer_rewards_detail.append(f"buyer_savings_s1({buyer_savings_s1:.2f} * {weights['buyer_savings']:.2f})={weighted_savings_s1:.2f}")
                
                if info.get('b1s2_buyer_price') is not None and env.buyer1_max_price is not None:
                    buyer_price_s2 = info.get('b1s2_buyer_price', 0)
                    buyer_savings_s2 = env.buyer1_max_price - buyer_price_s2
                    weighted_savings_s2 = buyer_savings_s2 * weights["buyer_savings"]
                    buyer_rewards_detail.append(f"buyer_savings_s2({buyer_savings_s2:.2f} * {weights['buyer_savings']:.2f})={weighted_savings_s2:.2f}")
                
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
                if info.get('b2s1_buyer_price') is not None and env.buyer2_max_price is not None:
                    buyer_price_s1 = info.get('b2s1_buyer_price', 0)
                    buyer_savings_s1 = env.buyer2_max_price - buyer_price_s1
                    weighted_savings_s1 = buyer_savings_s1 * weights["buyer_savings"]
                    buyer_rewards_detail.append(f"buyer_savings_s1({buyer_savings_s1:.2f} * {weights['buyer_savings']:.2f})={weighted_savings_s1:.2f}")
                
                if info.get('b2s2_buyer_price') is not None and env.buyer2_max_price is not None:
                    buyer_price_s2 = info.get('b2s2_buyer_price', 0)
                    buyer_savings_s2 = env.buyer2_max_price - buyer_price_s2
                    weighted_savings_s2 = buyer_savings_s2 * weights["buyer_savings"]
                    buyer_rewards_detail.append(f"buyer_savings_s2({buyer_savings_s2:.2f} * {weights['buyer_savings']:.2f})={weighted_savings_s2:.2f}")
                
                if buyer_rewards_detail:
                    aggregated_detail = f"aggregated({env.buyer_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = {aggregated_detail}[{', '.join(buyer_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, round={info['round']}, aggregation={env.buyer_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info:
                seller_rewards_detail = []
                if info.get('b1s1_seller_price') is not None and env.seller1_min_price is not None:
                    seller_price_b1 = info.get('b1s1_seller_price', 0)
                    seller_profit_b1 = seller_price_b1 - env.seller1_min_price
                    weighted_profit_b1 = seller_profit_b1 * weights["seller_profit"]
                    seller_rewards_detail.append(f"seller_profit_b1({seller_profit_b1:.2f} * {weights['seller_profit']:.2f})={weighted_profit_b1:.2f}")
                
                if info.get('b2s1_seller_price') is not None and env.seller1_min_price is not None:
                    seller_price_b2 = info.get('b2s1_seller_price', 0)
                    seller_profit_b2 = seller_price_b2 - env.seller1_min_price
                    weighted_profit_b2 = seller_profit_b2 * weights["seller_profit"]
                    seller_rewards_detail.append(f"seller_profit_b2({seller_profit_b2:.2f} * {weights['seller_profit']:.2f})={weighted_profit_b2:.2f}")
                
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
                if info.get('b1s2_seller_price') is not None and env.seller2_min_price is not None:
                    seller_price_b1 = info.get('b1s2_seller_price', 0)
                    seller_profit_b1 = seller_price_b1 - env.seller2_min_price
                    weighted_profit_b1 = seller_profit_b1 * weights["seller_profit"]
                    seller_rewards_detail.append(f"seller_profit_b1({seller_profit_b1:.2f} * {weights['seller_profit']:.2f})={weighted_profit_b1:.2f}")
                
                if info.get('b2s2_seller_price') is not None and env.seller2_min_price is not None:
                    seller_price_b2 = info.get('b2s2_seller_price', 0)
                    seller_profit_b2 = seller_price_b2 - env.seller2_min_price
                    weighted_profit_b2 = seller_profit_b2 * weights["seller_profit"]
                    seller_rewards_detail.append(f"seller_profit_b2({seller_profit_b2:.2f} * {weights['seller_profit']:.2f})={weighted_profit_b2:.2f}")
                
                if seller_rewards_detail:
                    aggregated_detail = f"aggregated({env.seller_reward_aggregation})"
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = {aggregated_detail}[{', '.join(seller_rewards_detail)}] + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_min={env.seller2_min_price}, round={info['round']}, aggregation={env.seller_reward_aggregation})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer') and info.get('selected_seller'):
                print(f"Selected Deal: Buyer {info['selected_buyer']} - Seller {info['selected_seller']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
            print(f"Buyer1-Seller1 Prices: Buyer=${info.get('b1s1_buyer_price', 0):.2f} | Seller=${info.get('b1s1_seller_price', 0):.2f}")
            print(f"Buyer1-Seller2 Prices: Buyer=${info.get('b1s2_buyer_price', 0):.2f} | Seller=${info.get('b1s2_seller_price', 0):.2f}")
            print(f"Buyer2-Seller1 Prices: Buyer=${info.get('b2s1_buyer_price', 0):.2f} | Seller=${info.get('b2s1_seller_price', 0):.2f}")
            print(f"Buyer2-Seller2 Prices: Buyer=${info.get('b2s2_buyer_price', 0):.2f} | Seller=${info.get('b2s2_seller_price', 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer1_reward' in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if 'buyer2_reward' in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if 'seller1_reward' in info:
                print(f"Seller1 Reward: {info['seller1_reward']:.3f}")
            if 'seller2_reward' in info:
                print(f"Seller2 Reward: {info['seller2_reward']:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Close environment
    env.close()
    print("\nMulti-buyer multi-seller negotiation completed!")


if __name__ == "__main__":
    main()
