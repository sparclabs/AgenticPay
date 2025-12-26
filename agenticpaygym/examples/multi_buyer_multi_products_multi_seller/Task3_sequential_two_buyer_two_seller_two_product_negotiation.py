"""Task3 Sequential Two-Buyer Two-Seller Two-Product Negotiation Example

Demonstrates how to use the Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation to negotiate sequentially with
two buyers and two sellers for two products, where each buyer chooses one seller per round to negotiate with.
Prices represent total price for both products.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.multi_buyer_multi_products_multi_seller.Task3_sequential_two_buyer_two_seller_two_product_negotiation import Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.models.custom_llm import CustomLLM
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
        buyer_id: Buyer ID (1 or 2)
        
    Returns:
        1 or 2, indicating which seller buyer wants to negotiate with
    """
    response_lower = buyer_response.lower()
    
    # Look for explicit seller mentions
    if re.search(r'seller\s*[12]|first\s+seller|seller\s*one', response_lower):
        if re.search(r'seller\s*2|second\s+seller|seller\s*two', response_lower):
            return 2
        elif re.search(r'seller\s*1|first\s+seller|seller\s*one', response_lower):
            return 1
    
    # If no explicit mention, try to infer from context
    # Check if buyer mentions prices or other indicators
    # Get prices for this buyer
    if buyer_id == 1:
        seller1_price = observation.get("b1s1_seller_price")
        seller2_price = observation.get("b1s2_seller_price")
    else:  # buyer_id == 2
        seller1_price = observation.get("b2s1_seller_price")
        seller2_price = observation.get("b2s2_seller_price")
    
    # If buyer mentions a specific price, try to match it
    price_match = re.search(r'\$?(\d+\.?\d*)', buyer_response)
    if price_match:
        mentioned_price = float(price_match.group(1))
        if seller1_price is not None and abs(mentioned_price - seller1_price) < 5:
            return 1
        elif seller2_price is not None and abs(mentioned_price - seller2_price) < 5:
            return 2
    
    # Default: if no clear indication, check which seller has been negotiated with more
    # or which has a better price
    if seller1_price is not None and seller2_price is not None:
        # Choose the one with lower price if both available
        return 1 if seller1_price <= seller2_price else 2
    elif seller1_price is not None:
        return 1
    elif seller2_price is not None:
        return 2
    
    # Final default: seller1
    return 1


def main():
    """Main function: Demonstrates sequential multi-buyer multi-seller multi-product negotiation flow"""
    
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
    # buyer_max_price and seller_min_price represent total expected cost for both products
    print("Creating agents...")
    buyer1_max_price = 200.0  # Maximum acceptable total purchase price for buyer1 (confidential, for both products)
    buyer2_max_price = 220.0  # Maximum acceptable total purchase price for buyer2 (confidential, for both products)
    seller1_min_price = 150.0  # Minimum acceptable total selling price for seller1 (confidential, for both products)
    seller2_min_price = 160.0  # Minimum acceptable total selling price for seller2 (confidential, for both products)
    
    buyer1 = BuyerAgent(llm=llm, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(llm=llm, buyer_max_price=buyer2_max_price)
    seller1 = SellerAgent(llm=llm, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(llm=llm, seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating sequential multi-buyer multi-seller multi-product negotiation environment...")
    env = Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=250.0,  # Initial total price offered by seller1 for both products
        initial_seller2_price=260.0,  # Initial total price offered by seller2 for both products (higher)
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential, for both products)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential, for both products)
        seller1_min_price=seller1_min_price,  # Seller1 total min price (confidential, for both products)
        seller2_min_price=seller2_min_price,  # Seller2 total min price (confidential, for both products)
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
    print("Starting new sequential negotiation with two buyers and two sellers for two products...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Each round, each buyer chooses one seller to negotiate with
        # Let buyers decide which seller to negotiate with and provide negotiation message
        
        # Build combined conversation history for buyer1 (includes both sellers' conversations)
        combined_history_b1 = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_b1s1", []):
            combined_history_b1.append({
                **msg,
                "content": f"[Seller 1] {msg['content']}"
            })
        # Add seller2 messages with prefix
        for msg in observation.get("conversation_history_b1s2", []):
            combined_history_b1.append({
                **msg,
                "content": f"[Seller 2] {msg['content']}"
            })
        
        # Build combined conversation history for buyer2 (includes both sellers' conversations)
        combined_history_b2 = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_b2s1", []):
            combined_history_b2.append({
                **msg,
                "content": f"[Seller 1] {msg['content']}"
            })
        # Add seller2 messages with prefix
        for msg in observation.get("conversation_history_b2s2", []):
            combined_history_b2.append({
                **msg,
                "content": f"[Seller 2] {msg['content']}"
            })
        
        # Get buyer1's response - buyer should indicate which seller they want to negotiate with
        buyer1_response = buyer1.respond(
            conversation_history=combined_history_b1,
            current_state={
                **observation,
                "instruction": "You are negotiating with two sellers for two products. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1 or 2) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'. Prices represent total price for both products."
            }
        )
        
        # Get buyer2's response - buyer should indicate which seller they want to negotiate with
        buyer2_response = buyer2.respond(
            conversation_history=combined_history_b2,
            current_state={
                **observation,
                "instruction": "You are negotiating with two sellers for two products. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1 or 2) you want to negotiate with, for example: 'I want to negotiate with seller 1' or 'Let me talk to seller 2'. Prices represent total price for both products."
            }
        )
        
        # Extract seller choice from each buyer's response
        buyer1_selected_seller = extract_seller_choice(buyer1_response, observation, buyer_id=1)
        buyer2_selected_seller = extract_seller_choice(buyer2_response, observation, buyer_id=2)
        
        print(f"\n[Buyer 1 chooses to negotiate with Seller {buyer1_selected_seller} this round]")
        print(f"[Buyer 2 chooses to negotiate with Seller {buyer2_selected_seller} this round]")
        
        # Use buyer's full response as the negotiation message
        buyer1_action = buyer1_response
        buyer2_action = buyer2_response
        
        # Get the conversation history for each buyer-seller pair
        if buyer1_selected_seller == 1:
            conversation_history_b1s1 = observation["conversation_history_b1s1"]
        else:
            conversation_history_b1s2 = observation["conversation_history_b1s2"]
        
        if buyer2_selected_seller == 1:
            conversation_history_b2s1 = observation["conversation_history_b2s1"]
        else:
            conversation_history_b2s2 = observation["conversation_history_b2s2"]
        
        # Get the selected sellers' responses
        seller1_action_buyer1 = None
        seller1_action_buyer2 = None
        seller2_action_buyer1 = None
        seller2_action_buyer2 = None
        
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
        
        # Print conversation content for this round
        current_round = observation.get('current_round', 0)
        print(f"\n{'='*60}")
        print(f"Round {current_round} Conversation:")
        print(f"{'='*60}")
        if buyer1_selected_seller == 1:
            print(f"[BUYER 1 to Seller 1]: {buyer1_action}")
            if seller1_action_buyer1:
                print(f"[SELLER 1 to Buyer 1]: {seller1_action_buyer1}")
        elif buyer1_selected_seller == 2:
            print(f"[BUYER 1 to Seller 2]: {buyer1_action}")
            if seller2_action_buyer1:
                print(f"[SELLER 2 to Buyer 1]: {seller2_action_buyer1}")
        
        if buyer2_selected_seller == 1:
            print(f"[BUYER 2 to Seller 1]: {buyer2_action}")
            if seller1_action_buyer2:
                print(f"[SELLER 1 to Buyer 2]: {seller1_action_buyer2}")
        elif buyer2_selected_seller == 2:
            print(f"[BUYER 2 to Seller 2]: {buyer2_action}")
            if seller2_action_buyer2:
                print(f"[SELLER 2 to Buyer 2]: {seller2_action_buyer2}")
        print(f"{'='*60}")
        
        # Execute step with selected sellers and actions
        observation, reward, terminated, truncated, info = env.step(
            buyer1_selected_seller=buyer1_selected_seller,
            buyer2_selected_seller=buyer2_selected_seller,
            buyer1_action=buyer1_action,
            buyer2_action=buyer2_action,
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
                buyer_price = None
                if info.get('buyer1_selected_seller') == 1:
                    buyer_price = info.get('b1s1_buyer_price')
                elif info.get('buyer1_selected_seller') == 2:
                    buyer_price = info.get('b1s2_buyer_price')
                
                if buyer_price is not None and env.buyer1_max_price is not None:
                    buyer_savings = env.buyer1_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info:
                buyer_price = None
                if info.get('buyer2_selected_seller') == 1:
                    buyer_price = info.get('b2s1_buyer_price')
                elif info.get('buyer2_selected_seller') == 2:
                    buyer_price = info.get('b2s2_buyer_price')
                
                if buyer_price is not None and env.buyer2_max_price is not None:
                    buyer_savings = env.buyer2_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info:
                seller1_price = None
                # Get price from whichever buyer selected seller1
                if info.get('buyer1_selected_seller') == 1 and info.get('b1s1_seller_price') is not None:
                    seller1_price = info.get('b1s1_seller_price')
                elif info.get('buyer2_selected_seller') == 1 and info.get('b2s1_seller_price') is not None:
                    seller1_price = info.get('b2s1_seller_price')
                # If both selected seller1, prefer higher price
                if (info.get('buyer1_selected_seller') == 1 and info.get('buyer2_selected_seller') == 1 and
                    info.get('b1s1_seller_price') is not None and info.get('b2s1_seller_price') is not None):
                    seller1_price = max(info.get('b1s1_seller_price'), info.get('b2s1_seller_price'))
                
                if seller1_price is not None and env.seller1_min_price is not None:
                    seller1_profit = seller1_price - env.seller1_min_price
                    weighted_seller1_profit = seller1_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_total_price={seller1_price:.2f}, seller1_min={env.seller1_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price not specified, round={info['round']})")
            
            # Seller2 step reward details
            if 'step_seller2_reward' in info:
                seller2_price = None
                # Get price from whichever buyer selected seller2
                if info.get('buyer1_selected_seller') == 2 and info.get('b1s2_seller_price') is not None:
                    seller2_price = info.get('b1s2_seller_price')
                elif info.get('buyer2_selected_seller') == 2 and info.get('b2s2_seller_price') is not None:
                    seller2_price = info.get('b2s2_seller_price')
                # If both selected seller2, prefer higher price
                if (info.get('buyer1_selected_seller') == 2 and info.get('buyer2_selected_seller') == 2 and
                    info.get('b1s2_seller_price') is not None and info.get('b2s2_seller_price') is not None):
                    seller2_price = max(info.get('b1s2_seller_price'), info.get('b2s2_seller_price'))
                
                if seller2_price is not None and env.seller2_min_price is not None:
                    seller2_profit = seller2_price - env.seller2_min_price
                    weighted_seller2_profit = seller2_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_total_price={seller2_price:.2f}, seller2_min={env.seller2_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer') and info.get('selected_seller'):
                print(f"Selected Deal: Buyer {info['selected_buyer']} - Seller {info['selected_seller']}")
                print(f"Final Deal Total Price: ${info.get('final_deal_price', 0):.2f}")
            print(f"Buyer1-Seller1 Total Prices: Buyer=${info.get('b1s1_buyer_price', 0) or 0:.2f} | Seller=${info.get('b1s1_seller_price', 0) or 0:.2f}")
            print(f"Buyer1-Seller2 Total Prices: Buyer=${info.get('b1s2_buyer_price', 0) or 0:.2f} | Seller=${info.get('b1s2_seller_price', 0) or 0:.2f}")
            print(f"Buyer2-Seller1 Total Prices: Buyer=${info.get('b2s1_buyer_price', 0) or 0:.2f} | Seller=${info.get('b2s1_seller_price', 0) or 0:.2f}")
            print(f"Buyer2-Seller2 Total Prices: Buyer=${info.get('b2s2_buyer_price', 0) or 0:.2f} | Seller=${info.get('b2s2_seller_price', 0) or 0:.2f}")
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
    print("\nSequential multi-buyer multi-seller multi-product negotiation completed!")


if __name__ == "__main__":
    main()
