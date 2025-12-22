"""Task2 Close Price Negotiation Example

Tests negotiation scenarios where buyer_max_price is close to seller_min_price
to see if a deal can be reached. This is useful for testing edge cases in negotiation.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym import make, Task2ClosePriceNegotiation  # Use registration system
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM


def main():
    """Main function: Tests negotiation with buyer_max_price close to seller_min_price"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize LLM
    print("Initializing LLM...")
    llm = OpenAILLM(api_key=api_key, model="gpt-3.5-turbo") # gpt-4o-mini-2024-07-18, gpt-3.5-turbo
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # Task2: buyer_max_price is close to seller_min_price to test if deal can be reached
    print("Creating agents...")
    seller_min_price = 80.0  # Minimum acceptable selling price for seller (confidential)
    buyer_max_price = 82.0   # Maximum acceptable purchase price for buyer (confidential) - close to seller_min_price
    
    buyer = BuyerAgent(llm=llm, buyer_max_price=buyer_max_price)
    seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)
    
    # Configure reward weights
    reward_weights = {
        "buyer_savings": 1.0,      # 买方节省权重
        "seller_profit": 1.0,      # 卖方利润权重
        "time_cost": 0.1,          # 时间成本权重（降低影响）
    }
    
    # Method 1: Create environment using registration system (recommended)
    print("Creating negotiation environment using registration system...")
    print(f"Task2 Configuration:")
    print(f"  - Seller min price: ${seller_min_price:.2f}")
    print(f"  - Buyer max price: ${buyer_max_price:.2f}")
    print(f"  - Price difference: ${buyer_max_price - seller_min_price:.2f}")
    print(f"  - This tests if a deal can be reached when prices are very close")
    
    env = make(
        "Task2_close_price_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=20,
        initial_seller_price=100.0,  # Initial price offered by seller
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller_min_price=seller_min_price,  # Seller bottom price (confidential)
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=0.0,  # Tolerance for agreement
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Method 2: Direct instantiation (backward compatible, but not recommended)
    # env = Task2ClosePriceNegotiation(
    #     buyer_agent=buyer,
    #     seller_agent=seller,
    #     max_rounds=20,
    #     initial_seller_price=100.0,
    #     buyer_max_price=buyer_max_price,
    #     seller_min_price=seller_min_price,
    #     environment_info={
    #         "temperature": "warm",
    #         "season": "summer",
    #         "weather": "sunny",
    #     },
    #     price_tolerance=5.0,
    # )
    
    # Create user profile (text description of personal preferences)
    user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
    print(f"\nUser Profile: {user_profile}")
    
    # Get user requirement
    print("\n" + "="*60)
    print("Please enter the product requirement you want to purchase:")
    user_requirement = input("> ").strip()
    if not user_requirement:
        print("No requirement entered, using default requirement...")
        user_requirement = "I need a high-quality winter jacket for cold weather"
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new negotiation (Task2: Close Price Test)...")
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
        # Each round, both buyer and seller respond
        # Get buyer's response
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
            print(f"Final Prices: Seller=${info.get('seller_price', 0):.2f} | Buyer=${info.get('buyer_price', 0):.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Total Reward: {reward:.3f}")
            if 'seller_reward' in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
            if 'buyer_reward' in info:
                print(f"Buyer Reward: {info['buyer_reward']:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            
            # Task2 specific analysis
            if info.get('agreed_price'):
                print(f"\nTask2 Analysis:")
                print(f"  - Agreed Price: ${info['agreed_price']:.2f}")
                print(f"  - Seller Min Price: ${seller_min_price:.2f}")
                print(f"  - Buyer Max Price: ${buyer_max_price:.2f}")
                print(f"  - Deal Reached: {'YES' if terminated else 'NO'}")
                if terminated:
                    seller_profit = info['agreed_price'] - seller_min_price
                    buyer_savings = buyer_max_price - info['agreed_price']
                    print(f"  - Seller Profit: ${seller_profit:.2f}")
                    print(f"  - Buyer Savings: ${buyer_savings:.2f}")
            print("="*60)
            break
    
    # Close environment
    env.close()
    print("\nNegotiation completed!")


if __name__ == "__main__":
    main()

