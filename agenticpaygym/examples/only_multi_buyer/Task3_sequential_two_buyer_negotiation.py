"""Task3 Sequential Two-Buyer Negotiation Example

Demonstrates how to use the Task3SequentialTwoBuyerNegotiation to negotiate sequentially with two buyers,
where seller chooses one buyer per round to negotiate with.
"""

import os
import sys

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from agenticpaygym.envs.only_multi_buyer.Task3_sequential_two_buyer_negotiation import Task3SequentialTwoBuyerNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.models.custom_llm import CustomLLM
import re

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance


def extract_buyer_choice(seller_response: str, observation: dict) -> int:
    """Extract buyer choice from seller's response
    
    Seller should indicate which buyer they want to negotiate with.
    Look for patterns like "buyer 1", "buyer1", "first buyer", etc.
    
    Args:
        seller_response: Seller's response text
        observation: Current observation from environment
        
    Returns:
        1 or 2, indicating which buyer seller wants to negotiate with
    """
    response_lower = seller_response.lower()
    
    # Look for explicit buyer mentions
    if re.search(r'buyer\s*[12]|first\s+buyer|buyer\s*one', response_lower):
        if re.search(r'buyer\s*2|second\s+buyer|buyer\s*two', response_lower):
            return 2
        elif re.search(r'buyer\s*1|first\s+buyer|buyer\s*one', response_lower):
            return 1
    
    # If no explicit mention, try to infer from context
    # Check if seller mentions prices or other indicators
    buyer1_price = observation.get("buyer1_price")
    buyer2_price = observation.get("buyer2_price")
    seller_price_buyer1 = observation.get("seller_price_buyer1")
    seller_price_buyer2 = observation.get("seller_price_buyer2")
    
    # If seller mentions a specific price, try to match it
    price_match = re.search(r'\$?(\d+\.?\d*)', seller_response)
    if price_match:
        mentioned_price = float(price_match.group(1))
        if seller_price_buyer1 is not None and abs(mentioned_price - seller_price_buyer1) < 5:
            return 1
        elif seller_price_buyer2 is not None and abs(mentioned_price - seller_price_buyer2) < 5:
            return 2
    
    # Default: if no clear indication, check which buyer has been negotiated with more
    # or which has a better price (higher buyer price is better for seller)
    if buyer1_price is not None and buyer2_price is not None:
        # Choose the one with higher price if both available
        return 1 if buyer1_price >= buyer2_price else 2
    elif buyer1_price is not None:
        return 1
    elif buyer2_price is not None:
        return 2
    
    # Final default: buyer1
    return 1


def main():
    """Main function: Demonstrates sequential multi-buyer negotiation flow"""
    
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
    buyer2_max_price = 130.0  # Maximum acceptable purchase price for buyer2 (confidential, different from buyer1)
    seller_min_price = 80.0  # Minimum acceptable selling price for seller (confidential)
    
    buyer1 = BuyerAgent(llm=llm, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(llm=llm, buyer_max_price=buyer2_max_price)
    seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)
    
    # Create environment
    print("Creating sequential multi-buyer negotiation environment...")
    env = Task3SequentialTwoBuyerNegotiation(
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
    print("Starting new sequential negotiation with two buyers...")
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
        # Each round, seller chooses one buyer to negotiate with
        # Seller can see both buyers' information in the observation
        # Let seller decide which buyer to negotiate with and provide negotiation message
        # We'll use a combined conversation history that includes both buyers' conversations
        combined_history = []
        # Add buyer1 messages with prefix
        for msg in observation.get("conversation_history_buyer1", []):
            combined_history.append({
                **msg,
                "content": f"[Buyer 1] {msg['content']}"
            })
        # Add buyer2 messages with prefix
        for msg in observation.get("conversation_history_buyer2", []):
            combined_history.append({
                **msg,
                "content": f"[Buyer 2] {msg['content']}"
            })
        
        # Get seller's response - seller should indicate which buyer they want to negotiate with
        seller_response = seller.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": "You are negotiating with two buyers. Each round, you need to choose ONE buyer to negotiate with and provide your negotiation message. Please clearly indicate which buyer (1 or 2) you want to negotiate with, for example: 'I want to negotiate with buyer 1' or 'Let me talk to buyer 2'."
            }
        )
        
        # Extract buyer choice from seller's response
        selected_buyer = extract_buyer_choice(seller_response, observation)
        print(f"\n[Seller chooses to negotiate with Buyer {selected_buyer} this round]")
        
        # Use seller's full response as the negotiation message
        # The response may include the choice statement, which is fine as it's seller's natural expression
        seller_action = seller_response
        
        # Get the conversation history for the selected buyer
        if selected_buyer == 1:
            conversation_history = observation["conversation_history_buyer1"]
        else:
            conversation_history = observation["conversation_history_buyer2"]
        
        # Get the selected buyer's response
        if selected_buyer == 1:
            buyer_action = buyer1.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        else:
            buyer_action = buyer2.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        
        # Print conversation content for this round
        current_round = observation.get('current_round', 0)
        print(f"\n{'='*60}")
        print(f"Round {current_round} Conversation:")
        print(f"{'='*60}")
        print(f"[SELLER to Buyer {selected_buyer}]: {seller_action}")
        print(f"[BUYER {selected_buyer}]: {buyer_action}")
        print(f"{'='*60}")
        
        # Execute step with selected buyer and actions
        observation, reward, terminated, truncated, info = env.step(
            selected_buyer=selected_buyer,
            seller_action=seller_action,
            buyer_action=buyer_action
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
            if 'step_buyer1_reward' in info:
                buyer1_price = info.get('buyer1_price')
                if buyer1_price is not None and env.buyer1_max_price is not None:
                    buyer1_savings = env.buyer1_max_price - buyer1_price
                    weighted_savings = buyer1_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer1_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, buyer1_price={buyer1_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer1_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info:
                buyer2_price = info.get('buyer2_price')
                if buyer2_price is not None and env.buyer2_max_price is not None:
                    buyer2_savings = env.buyer2_max_price - buyer2_price
                    weighted_savings = buyer2_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer2_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer2_price={buyer2_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer2_price not specified, round={info['round']})")
            
            # Seller step reward details
            if 'step_seller_reward' in info:
                seller_price = None
                if info.get('current_selected_buyer') == 1:
                    seller_price = info.get('seller_price_buyer1')
                elif info.get('current_selected_buyer') == 2:
                    seller_price = info.get('seller_price_buyer2')
                
                if seller_price is not None and env.seller_min_price is not None:
                    seller_profit = seller_price - env.seller_min_price
                    weighted_seller_profit = seller_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = seller_profit({seller_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller_reward']:.2f} (seller_price={seller_price:.2f}, seller_min={env.seller_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer'):
                print(f"Final Selected Buyer: Buyer {info['selected_buyer']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
            buyer1_price = info.get('buyer1_price', 0) or 0
            seller_price_buyer1 = info.get('seller_price_buyer1', 0) or 0
            buyer2_price = info.get('buyer2_price', 0) or 0
            seller_price_buyer2 = info.get('seller_price_buyer2', 0) or 0
            print(f"Buyer1 Prices: Buyer=${buyer1_price:.2f} | Seller=${seller_price_buyer1:.2f}")
            print(f"Buyer2 Prices: Buyer=${buyer2_price:.2f} | Seller=${seller_price_buyer2:.2f}")
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

