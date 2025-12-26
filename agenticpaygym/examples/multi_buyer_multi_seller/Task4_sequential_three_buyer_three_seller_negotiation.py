"""Task4 Sequential Three-Buyer Three-Seller Negotiation Example

Demonstrates how to use the Task4SequentialThreeBuyerThreeSellerNegotiation to negotiate sequentially with
three buyers and three sellers, where each buyer chooses one seller per round to negotiate with.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.multi_buyer_multi_seller.Task4_sequential_three_buyer_three_seller_negotiation import Task4SequentialThreeBuyerThreeSellerNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM
import re

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, max_rounds, price_tolerance


def extract_seller_choice(buyer_response: str, observation: dict, buyer_id: int) -> int:
    """Extract seller choice from buyer's response
    
    Buyer should indicate which seller they want to negotiate with.
    Look for patterns like "seller 1", "seller1", "first seller", etc.
    
    Args:
        buyer_response: Buyer's response text
        observation: Current observation from environment
        buyer_id: Buyer ID (1, 2, or 3)
        
    Returns:
        1, 2, or 3, indicating which seller buyer wants to negotiate with
    """
    response_lower = buyer_response.lower()
    
    # Look for explicit seller mentions
    if re.search(r'seller\s*[123]|first\s+seller|seller\s*one|second\s+seller|seller\s*two|third\s+seller|seller\s*three', response_lower):
        if re.search(r'seller\s*3|third\s+seller|seller\s*three', response_lower):
            return 3
        elif re.search(r'seller\s*2|second\s+seller|seller\s*two', response_lower):
            return 2
        elif re.search(r'seller\s*1|first\s+seller|seller\s*one', response_lower):
            return 1
    
    # If no explicit mention, try to infer from context
    # Check if buyer mentions prices or other indicators
    # Get prices for this buyer
    seller1_price = observation.get(f"b{buyer_id}s1_seller_price")
    seller2_price = observation.get(f"b{buyer_id}s2_seller_price")
    seller3_price = observation.get(f"b{buyer_id}s3_seller_price")
    
    # If buyer mentions a specific price, try to match it
    price_match = re.search(r'\$?(\d+\.?\d*)', buyer_response)
    if price_match:
        mentioned_price = float(price_match.group(1))
        if seller1_price is not None and abs(mentioned_price - seller1_price) < 5:
            return 1
        elif seller2_price is not None and abs(mentioned_price - seller2_price) < 5:
            return 2
        elif seller3_price is not None and abs(mentioned_price - seller3_price) < 5:
            return 3
    
    # Default: if no clear indication, check which seller has been negotiated with more
    # or which has a better price
    prices = []
    if seller1_price is not None:
        prices.append((1, seller1_price))
    if seller2_price is not None:
        prices.append((2, seller2_price))
    if seller3_price is not None:
        prices.append((3, seller3_price))
    
    if prices:
        # Choose the one with lowest price if multiple available
        prices.sort(key=lambda x: x[1])
        return prices[0][0]
    
    # Final default: seller1
    return 1


def main():
    """Main function: Demonstrates sequential multi-buyer multi-seller negotiation flow"""
    
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
    buyer2_max_price = 130.0  # Maximum acceptable purchase price for buyer2 (confidential)
    buyer3_max_price = 140.0  # Maximum acceptable purchase price for buyer3 (confidential)
    seller1_min_price = 80.0  # Minimum acceptable selling price for seller1 (confidential)
    seller2_min_price = 85.0  # Minimum acceptable selling price for seller2 (confidential)
    seller3_min_price = 90.0  # Minimum acceptable selling price for seller3 (confidential)
    
    buyer1 = BuyerAgent(llm=llm, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(llm=llm, buyer_max_price=buyer2_max_price)
    buyer3 = BuyerAgent(llm=llm, buyer_max_price=buyer3_max_price)
    seller1 = SellerAgent(llm=llm, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(llm=llm, seller_min_price=seller2_min_price)
    seller3 = SellerAgent(llm=llm, seller_min_price=seller3_min_price)
    
    # Create environment
    print("Creating sequential multi-buyer multi-seller negotiation environment...")
    env = Task4SequentialThreeBuyerThreeSellerNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        buyer3_agent=buyer3,
        seller1_agent=seller1,
        seller2_agent=seller2,
        seller3_agent=seller3,
        max_rounds=max_rounds,
        initial_seller1_price=150.0,  # Initial price offered by seller1
        initial_seller2_price=160.0,  # Initial price offered by seller2 (higher)
        initial_seller3_price=170.0,  # Initial price offered by seller3 (highest)
        buyer1_max_price=buyer1_max_price,  # Buyer1 bottom price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 bottom price (confidential)
        buyer3_max_price=buyer3_max_price,  # Buyer3 bottom price (confidential)
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
    print("Starting new sequential negotiation with three buyers and three sellers...")
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
        # Each round, each buyer chooses one seller to negotiate with
        # Let buyers decide which seller to negotiate with and provide negotiation message
        
        # Build combined conversation history for each buyer (includes all sellers' conversations)
        combined_history_b1 = []
        combined_history_b2 = []
        combined_history_b3 = []
        
        for buyer_id in [1, 2, 3]:
            combined_history = []
            for seller_id in [1, 2, 3]:
                history_key = f"conversation_history_b{buyer_id}s{seller_id}"
                for msg in observation.get(history_key, []):
                    combined_history.append({
                        **msg,
                        "content": f"[Seller {seller_id}] {msg['content']}"
                    })
            
            if buyer_id == 1:
                combined_history_b1 = combined_history
            elif buyer_id == 2:
                combined_history_b2 = combined_history
            else:  # buyer_id == 3
                combined_history_b3 = combined_history
        
        # Get each buyer's response - buyer should indicate which seller they want to negotiate with
        buyer1_response = buyer1.respond(
            conversation_history=combined_history_b1,
            current_state={
                **observation,
                "instruction": "You are negotiating with three sellers. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1, 2, or 3) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'."
            }
        )
        
        buyer2_response = buyer2.respond(
            conversation_history=combined_history_b2,
            current_state={
                **observation,
                "instruction": "You are negotiating with three sellers. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1, 2, or 3) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'."
            }
        )
        
        buyer3_response = buyer3.respond(
            conversation_history=combined_history_b3,
            current_state={
                **observation,
                "instruction": "You are negotiating with three sellers. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1, 2, or 3) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'."
            }
        )
        
        # Extract seller choice from each buyer's response
        buyer1_selected_seller = extract_seller_choice(buyer1_response, observation, buyer_id=1)
        buyer2_selected_seller = extract_seller_choice(buyer2_response, observation, buyer_id=2)
        buyer3_selected_seller = extract_seller_choice(buyer3_response, observation, buyer_id=3)
        
        print(f"\n[Buyer 1 chooses to negotiate with Seller {buyer1_selected_seller} this round]")
        print(f"[Buyer 2 chooses to negotiate with Seller {buyer2_selected_seller} this round]")
        print(f"[Buyer 3 chooses to negotiate with Seller {buyer3_selected_seller} this round]")
        
        # Use buyer's full response as the negotiation message
        buyer1_action = buyer1_response
        buyer2_action = buyer2_response
        buyer3_action = buyer3_response
        
        # Get the conversation history for each buyer-seller pair
        conversation_history_b1s1 = observation.get("conversation_history_b1s1", [])
        conversation_history_b1s2 = observation.get("conversation_history_b1s2", [])
        conversation_history_b1s3 = observation.get("conversation_history_b1s3", [])
        conversation_history_b2s1 = observation.get("conversation_history_b2s1", [])
        conversation_history_b2s2 = observation.get("conversation_history_b2s2", [])
        conversation_history_b2s3 = observation.get("conversation_history_b2s3", [])
        conversation_history_b3s1 = observation.get("conversation_history_b3s1", [])
        conversation_history_b3s2 = observation.get("conversation_history_b3s2", [])
        conversation_history_b3s3 = observation.get("conversation_history_b3s3", [])
        
        # Get the selected sellers' responses
        seller1_action_buyer1 = None
        seller1_action_buyer2 = None
        seller1_action_buyer3 = None
        seller2_action_buyer1 = None
        seller2_action_buyer2 = None
        seller2_action_buyer3 = None
        seller3_action_buyer1 = None
        seller3_action_buyer2 = None
        seller3_action_buyer3 = None
        
        if buyer1_selected_seller == 1:
            seller1_action_buyer1 = seller1.respond(
                conversation_history=conversation_history_b1s1,
                current_state=observation
            )
        elif buyer1_selected_seller == 2:
            seller2_action_buyer1 = seller2.respond(
                conversation_history=conversation_history_b1s2,
                current_state=observation
            )
        elif buyer1_selected_seller == 3:
            seller3_action_buyer1 = seller3.respond(
                conversation_history=conversation_history_b1s3,
                current_state=observation
            )
        
        if buyer2_selected_seller == 1:
            seller1_action_buyer2 = seller1.respond(
                conversation_history=conversation_history_b2s1,
                current_state=observation
            )
        elif buyer2_selected_seller == 2:
            seller2_action_buyer2 = seller2.respond(
                conversation_history=conversation_history_b2s2,
                current_state=observation
            )
        elif buyer2_selected_seller == 3:
            seller3_action_buyer2 = seller3.respond(
                conversation_history=conversation_history_b2s3,
                current_state=observation
            )
        
        if buyer3_selected_seller == 1:
            seller1_action_buyer3 = seller1.respond(
                conversation_history=conversation_history_b3s1,
                current_state=observation
            )
        elif buyer3_selected_seller == 2:
            seller2_action_buyer3 = seller2.respond(
                conversation_history=conversation_history_b3s2,
                current_state=observation
            )
        elif buyer3_selected_seller == 3:
            seller3_action_buyer3 = seller3.respond(
                conversation_history=conversation_history_b3s3,
                current_state=observation
            )
        
        # Print conversation content for this round
        current_round = observation.get('current_round', 0)
        print(f"\n{'='*60}")
        print(f"Round {current_round} Conversation:")
        print(f"{'='*60}")
        
        for buyer_id in [1, 2, 3]:
            selected_seller = buyer1_selected_seller if buyer_id == 1 else (buyer2_selected_seller if buyer_id == 2 else buyer3_selected_seller)
            buyer_action = buyer1_action if buyer_id == 1 else (buyer2_action if buyer_id == 2 else buyer3_action)
            
            print(f"[BUYER {buyer_id} to Seller {selected_seller}]: {buyer_action}")
            
            if selected_seller == 1:
                seller_action = seller1_action_buyer1 if buyer_id == 1 else (seller1_action_buyer2 if buyer_id == 2 else seller1_action_buyer3)
            elif selected_seller == 2:
                seller_action = seller2_action_buyer1 if buyer_id == 1 else (seller2_action_buyer2 if buyer_id == 2 else seller2_action_buyer3)
            else:  # selected_seller == 3
                seller_action = seller3_action_buyer1 if buyer_id == 1 else (seller3_action_buyer2 if buyer_id == 2 else seller3_action_buyer3)
            
            if seller_action:
                print(f"[SELLER {selected_seller} to Buyer {buyer_id}]: {seller_action}")
        
        print(f"{'='*60}")
        
        # Execute step with selected sellers and actions
        observation, reward, terminated, truncated, info = env.step(
            buyer1_selected_seller=buyer1_selected_seller,
            buyer2_selected_seller=buyer2_selected_seller,
            buyer3_selected_seller=buyer3_selected_seller,
            buyer1_action=buyer1_action,
            buyer2_action=buyer2_action,
            buyer3_action=buyer3_action,
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
                if ('step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or
                    'step_seller1_reward' in info):
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            if 'step_seller3_reward' in info:
                if ('step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or
                    'step_seller1_reward' in info or 'step_seller2_reward' in info):
                    print(f" | ", end="")
                print(f"Seller3: {info['step_seller3_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer step reward details
            for buyer_id in [1, 2, 3]:
                reward_key = f'step_buyer{buyer_id}_reward'
                if reward_key in info:
                    buyer_price = None
                    selected_seller = info.get(f'buyer{buyer_id}_selected_seller')
                    if selected_seller is not None:
                        buyer_price = info.get(f'b{buyer_id}s{selected_seller}_buyer_price')
                    
                    buyer_max_price = getattr(env, f'buyer{buyer_id}_max_price')
                    if buyer_price is not None and buyer_max_price is not None:
                        buyer_savings = buyer_max_price - buyer_price
                        weighted_savings = buyer_savings * weights["buyer_savings"]
                        weighted_round_cost = round_cost * weights["time_cost"]
                        print(f"  Buyer{buyer_id} Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info[reward_key]:.2f} (buyer{buyer_id}_max={buyer_max_price}, buyer_price={buyer_price:.2f}, round={info['round']})")
                    else:
                        weighted_round_cost = round_cost * weights["time_cost"]
                        print(f"  Buyer{buyer_id} Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller step reward details
            for seller_id in [1, 2, 3]:
                reward_key = f'step_seller{seller_id}_reward'
                if reward_key in info:
                    seller_price = None
                    # Get price from whichever buyer(s) selected this seller
                    seller_prices = []
                    for buyer_id in [1, 2, 3]:
                        if info.get(f'buyer{buyer_id}_selected_seller') == seller_id:
                            price = info.get(f'b{buyer_id}s{seller_id}_seller_price')
                            if price is not None:
                                seller_prices.append(price)
                    
                    if seller_prices:
                        seller_price = max(seller_prices)  # Prefer higher price
                    
                    seller_min_price = getattr(env, f'seller{seller_id}_min_price')
                    if seller_price is not None and seller_min_price is not None:
                        seller_profit = seller_price - seller_min_price
                        weighted_seller_profit = seller_profit * weights["seller_profit"]
                        weighted_round_cost = round_cost * weights["time_cost"]
                        print(f"  Seller{seller_id} Step Reward = seller_profit({seller_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info[reward_key]:.2f} (seller{seller_id}_price={seller_price:.2f}, seller{seller_id}_min={seller_min_price}, round={info['round']})")
                    else:
                        weighted_round_cost = round_cost * weights["time_cost"]
                        print(f"  Seller{seller_id} Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller{seller_id}_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer') and info.get('selected_seller'):
                print(f"Selected Deal: Buyer {info['selected_buyer']} - Seller {info['selected_seller']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
            
            # Display prices for all buyer-seller pairs
            for buyer_id in [1, 2, 3]:
                for seller_id in [1, 2, 3]:
                    buyer_price = info.get(f'b{buyer_id}s{seller_id}_buyer_price', 0) or 0
                    seller_price = info.get(f'b{buyer_id}s{seller_id}_seller_price', 0) or 0
                    print(f"Buyer{buyer_id}-Seller{seller_id} Prices: Buyer=${buyer_price:.2f} | Seller=${seller_price:.2f}")
            
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
    print("\nSequential multi-buyer multi-seller negotiation completed!")


if __name__ == "__main__":
    main()

