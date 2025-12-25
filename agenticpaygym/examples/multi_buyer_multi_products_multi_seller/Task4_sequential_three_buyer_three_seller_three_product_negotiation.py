"""Task4 Sequential Three-Buyer Three-Seller Three-Product Negotiation Example

Demonstrates how to use the Task4SequentialThreeBuyerThreeSellerThreeProductNegotiation to negotiate sequentially with
three buyers and three sellers for three products, where each buyer chooses one seller per round to negotiate with.
Prices represent total price for all three products.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.multi_buyer_multi_products_multi_seller.Task4_sequential_three_buyer_three_seller_three_product_negotiation import Task4SequentialThreeBuyerThreeSellerThreeProductNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM
import re


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
    if re.search(r'seller\s*[123]|first\s+seller|second\s+seller|third\s+seller', response_lower):
        if re.search(r'seller\s*3|third\s+seller|seller\s*three', response_lower):
            return 3
        elif re.search(r'seller\s*2|second\s+seller|seller\s*two', response_lower):
            return 2
        elif re.search(r'seller\s*1|first\s+seller|seller\s*one', response_lower):
            return 1
    
    # If no explicit mention, try to infer from context
    if buyer_id == 1:
        seller1_price = observation.get("b1s1_seller_price")
        seller2_price = observation.get("b1s2_seller_price")
        seller3_price = observation.get("b1s3_seller_price")
    elif buyer_id == 2:
        seller1_price = observation.get("b2s1_seller_price")
        seller2_price = observation.get("b2s2_seller_price")
        seller3_price = observation.get("b2s3_seller_price")
    else:  # buyer_id == 3
        seller1_price = observation.get("b3s1_seller_price")
        seller2_price = observation.get("b3s2_seller_price")
        seller3_price = observation.get("b3s3_seller_price")
    
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
    
    # Default: if no clear indication, check which seller has a better price
    prices = [(1, seller1_price), (2, seller2_price), (3, seller3_price)]
    available_prices = [(sid, p) for sid, p in prices if p is not None]
    if available_prices:
        # Choose the one with lowest price if all available
        return min(available_prices, key=lambda x: x[1])[0]
    
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
    llm = OpenAILLM(api_key=api_key, model="gpt-4o-mini-2024-07-18")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price represent total expected cost for all three products
    print("Creating agents...")
    buyer1_max_price = 300.0  # Maximum acceptable total purchase price for buyer1 (confidential, for all three products)
    buyer2_max_price = 320.0  # Maximum acceptable total purchase price for buyer2 (confidential, for all three products)
    buyer3_max_price = 340.0  # Maximum acceptable total purchase price for buyer3 (confidential, for all three products)
    seller1_min_price = 220.0  # Minimum acceptable total selling price for seller1 (confidential, for all three products)
    seller2_min_price = 230.0  # Minimum acceptable total selling price for seller2 (confidential, for all three products)
    seller3_min_price = 240.0  # Minimum acceptable total selling price for seller3 (confidential, for all three products)
    
    buyer1 = BuyerAgent(llm=llm, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(llm=llm, buyer_max_price=buyer2_max_price)
    buyer3 = BuyerAgent(llm=llm, buyer_max_price=buyer3_max_price)
    seller1 = SellerAgent(llm=llm, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(llm=llm, seller_min_price=seller2_min_price)
    seller3 = SellerAgent(llm=llm, seller_min_price=seller3_min_price)
    
    # Configure reward weights
    reward_weights = {
        "buyer_savings": 1.0,
        "seller_profit": 1.0,
        "time_cost": 0.1,
    }
    
    # Create environment
    print("Creating sequential multi-buyer multi-seller multi-product negotiation environment...")
    env = Task4SequentialThreeBuyerThreeSellerThreeProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        buyer3_agent=buyer3,
        seller1_agent=seller1,
        seller2_agent=seller2,
        seller3_agent=seller3,
        max_rounds=20,
        initial_seller1_price=380.0,  # Initial total price offered by seller1 for all three products
        initial_seller2_price=390.0,  # Initial total price offered by seller2 for all three products
        initial_seller3_price=400.0,  # Initial total price offered by seller3 for all three products
        buyer1_max_price=buyer1_max_price,
        buyer2_max_price=buyer2_max_price,
        buyer3_max_price=buyer3_max_price,
        seller1_min_price=seller1_min_price,
        seller2_min_price=seller2_min_price,
        seller3_min_price=seller3_min_price,
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=5.0,
        reward_weights=reward_weights,
    )
    
    # Create user profile
    user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
    print(f"User Profile: {user_profile}")
    
    # Define three products with their individual prices
    product_info = {
        "products": [
            {
                "name": "Premium Winter Jacket",
                "brand": "Mountain Gear",
                "price": 120.0,
                "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                "condition": "New",
                "material": "Gore-Tex",
            },
            {
                "name": "Running Shoes",
                "brand": "SportMax",
                "price": 80.0,
                "features": ["Lightweight", "Cushioned", "Breathable", "Durable"],
                "condition": "New",
                "material": "Mesh and Synthetic",
            },
            {
                "name": "Hiking Backpack",
                "brand": "TrailBlazer",
                "price": 100.0,
                "features": ["Waterproof", "Lightweight", "Adjustable Straps", "Multiple Pockets"],
                "condition": "New",
                "material": "Nylon",
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts:")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Product Price: ${total_product_price:.2f}")
    
    # Get user requirement
    print("\n" + "="*60)
    print("Please enter the product requirement (should describe purchasing three products):")
    user_requirement = input("> ").strip()
    if not user_requirement:
        print("No requirement entered, using default requirement...")
        user_requirement = "I need a high-quality winter jacket, a pair of running shoes, and a hiking backpack for my outdoor activities"
        print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation with three buyers and three sellers for three products...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )
    
    # Start negotiation loop
    done = False
    
    while not done:
        # Build combined conversation history for each buyer
        for buyer_id in [1, 2, 3]:
            combined_history = []
            for seller_id in [1, 2, 3]:
                history_key = f"conversation_history_b{buyer_id}s{seller_id}"
                for msg in observation.get(history_key, []):
                    combined_history.append({
                        **msg,
                        "content": f"[Seller {seller_id}] {msg['content']}"
                    })
            
            # Get buyer's response
            if buyer_id == 1:
                buyer1_response = buyer1.respond(
                    conversation_history=combined_history,
                    current_state={
                        **observation,
                        "instruction": "You are negotiating with three sellers for three products. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1, 2, or 3) you want to negotiate with. Prices represent total price for all three products."
                    }
                )
            elif buyer_id == 2:
                buyer2_response = buyer2.respond(
                    conversation_history=combined_history,
                    current_state={
                        **observation,
                        "instruction": "You are negotiating with three sellers for three products. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1, 2, or 3) you want to negotiate with. Prices represent total price for all three products."
                    }
                )
            else:
                buyer3_response = buyer3.respond(
                    conversation_history=combined_history,
                    current_state={
                        **observation,
                        "instruction": "You are negotiating with three sellers for three products. Each round, you need to choose ONE seller to negotiate with and provide your negotiation message. Please clearly indicate which seller (1, 2, or 3) you want to negotiate with. Prices represent total price for all three products."
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
        conversation_histories = {}
        for buyer_id in [1, 2, 3]:
            for seller_id in [1, 2, 3]:
                key = f"b{buyer_id}s{seller_id}"
                history_key = f"conversation_history_{key}"
                conversation_histories[key] = observation.get(history_key, [])
        
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
                conversation_history=conversation_histories["b1s1"],
                current_state=observation
            )
        elif buyer1_selected_seller == 2:
            seller2_action_buyer1 = seller2.respond(
                conversation_history=conversation_histories["b1s2"],
                current_state=observation
            )
        else:
            seller3_action_buyer1 = seller3.respond(
                conversation_history=conversation_histories["b1s3"],
                current_state=observation
            )
        
        if buyer2_selected_seller == 1:
            seller1_action_buyer2 = seller1.respond(
                conversation_history=conversation_histories["b2s1"],
                current_state=observation
            )
        elif buyer2_selected_seller == 2:
            seller2_action_buyer2 = seller2.respond(
                conversation_history=conversation_histories["b2s2"],
                current_state=observation
            )
        else:
            seller3_action_buyer2 = seller3.respond(
                conversation_history=conversation_histories["b2s3"],
                current_state=observation
            )
        
        if buyer3_selected_seller == 1:
            seller1_action_buyer3 = seller1.respond(
                conversation_history=conversation_histories["b3s1"],
                current_state=observation
            )
        elif buyer3_selected_seller == 2:
            seller2_action_buyer3 = seller2.respond(
                conversation_history=conversation_histories["b3s2"],
                current_state=observation
            )
        else:
            seller3_action_buyer3 = seller3.respond(
                conversation_history=conversation_histories["b3s3"],
                current_state=observation
            )
        
        # Print conversation content for this round
        current_round = observation.get('current_round', 0)
        print(f"\n{'='*60}")
        print(f"Round {current_round} Conversation:")
        print(f"{'='*60}")
        for buyer_id, buyer_action, selected_seller in [(1, buyer1_action, buyer1_selected_seller),
                                                         (2, buyer2_action, buyer2_selected_seller),
                                                         (3, buyer3_action, buyer3_selected_seller)]:
            if selected_seller == 1:
                seller_action = seller1_action_buyer1 if buyer_id == 1 else (seller1_action_buyer2 if buyer_id == 2 else seller1_action_buyer3)
                print(f"[BUYER {buyer_id} to Seller 1]: {buyer_action}")
                if seller_action:
                    print(f"[SELLER 1 to Buyer {buyer_id}]: {seller_action}")
            elif selected_seller == 2:
                seller_action = seller2_action_buyer1 if buyer_id == 1 else (seller2_action_buyer2 if buyer_id == 2 else seller2_action_buyer3)
                print(f"[BUYER {buyer_id} to Seller 2]: {buyer_action}")
                if seller_action:
                    print(f"[SELLER 2 to Buyer {buyer_id}]: {seller_action}")
            else:
                seller_action = seller3_action_buyer1 if buyer_id == 1 else (seller3_action_buyer2 if buyer_id == 2 else seller3_action_buyer3)
                print(f"[BUYER {buyer_id} to Seller 3]: {buyer_action}")
                if seller_action:
                    print(f"[SELLER 3 to Buyer {buyer_id}]: {seller_action}")
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
        
        # Render current state
        env.render()
        
        # Display step rewards for each round with detailed calculation
        if ('step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_buyer3_reward' in info or
            'step_seller1_reward' in info or 'step_seller2_reward' in info or 'step_seller3_reward' in info):
            print(f"\n[Step Rewards] ", end="")
            rewards_list = []
            if 'step_buyer1_reward' in info:
                rewards_list.append(f"Buyer1: {info['step_buyer1_reward']:.3f}")
            if 'step_buyer2_reward' in info:
                rewards_list.append(f"Buyer2: {info['step_buyer2_reward']:.3f}")
            if 'step_buyer3_reward' in info:
                rewards_list.append(f"Buyer3: {info['step_buyer3_reward']:.3f}")
            if 'step_seller1_reward' in info:
                rewards_list.append(f"Seller1: {info['step_seller1_reward']:.3f}")
            if 'step_seller2_reward' in info:
                rewards_list.append(f"Seller2: {info['step_seller2_reward']:.3f}")
            if 'step_seller3_reward' in info:
                rewards_list.append(f"Seller3: {info['step_seller3_reward']:.3f}")
            print(" | ".join(rewards_list))
            
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
                elif info.get('buyer1_selected_seller') == 3:
                    buyer_price = info.get('b1s3_buyer_price')
                
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
                elif info.get('buyer2_selected_seller') == 3:
                    buyer_price = info.get('b2s3_buyer_price')
                
                if buyer_price is not None and env.buyer2_max_price is not None:
                    buyer_savings = env.buyer2_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Buyer3 step reward details
            if 'step_buyer3_reward' in info:
                buyer_price = None
                if info.get('buyer3_selected_seller') == 1:
                    buyer_price = info.get('b3s1_buyer_price')
                elif info.get('buyer3_selected_seller') == 2:
                    buyer_price = info.get('b3s2_buyer_price')
                elif info.get('buyer3_selected_seller') == 3:
                    buyer_price = info.get('b3s3_buyer_price')
                
                if buyer_price is not None and env.buyer3_max_price is not None:
                    buyer_savings = env.buyer3_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer3 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer3_reward']:.2f} (buyer3_max={env.buyer3_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer3 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info:
                seller1_price = None
                # Get price from whichever buyer selected seller1
                prices = []
                if info.get('buyer1_selected_seller') == 1 and info.get('b1s1_seller_price') is not None:
                    prices.append(info.get('b1s1_seller_price'))
                if info.get('buyer2_selected_seller') == 1 and info.get('b2s1_seller_price') is not None:
                    prices.append(info.get('b2s1_seller_price'))
                if info.get('buyer3_selected_seller') == 1 and info.get('b3s1_seller_price') is not None:
                    prices.append(info.get('b3s1_seller_price'))
                # If multiple buyers selected seller1, prefer higher price
                if prices:
                    seller1_price = max(prices)
                
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
                prices = []
                if info.get('buyer1_selected_seller') == 2 and info.get('b1s2_seller_price') is not None:
                    prices.append(info.get('b1s2_seller_price'))
                if info.get('buyer2_selected_seller') == 2 and info.get('b2s2_seller_price') is not None:
                    prices.append(info.get('b2s2_seller_price'))
                if info.get('buyer3_selected_seller') == 2 and info.get('b3s2_seller_price') is not None:
                    prices.append(info.get('b3s2_seller_price'))
                # If multiple buyers selected seller2, prefer higher price
                if prices:
                    seller2_price = max(prices)
                
                if seller2_price is not None and env.seller2_min_price is not None:
                    seller2_profit = seller2_price - env.seller2_min_price
                    weighted_seller2_profit = seller2_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_total_price={seller2_price:.2f}, seller2_min={env.seller2_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")
            
            # Seller3 step reward details
            if 'step_seller3_reward' in info:
                seller3_price = None
                # Get price from whichever buyer selected seller3
                prices = []
                if info.get('buyer1_selected_seller') == 3 and info.get('b1s3_seller_price') is not None:
                    prices.append(info.get('b1s3_seller_price'))
                if info.get('buyer2_selected_seller') == 3 and info.get('b2s3_seller_price') is not None:
                    prices.append(info.get('b2s3_seller_price'))
                if info.get('buyer3_selected_seller') == 3 and info.get('b3s3_seller_price') is not None:
                    prices.append(info.get('b3s3_seller_price'))
                # If multiple buyers selected seller3, prefer higher price
                if prices:
                    seller3_price = max(prices)
                
                if seller3_price is not None and env.seller3_min_price is not None:
                    seller3_profit = seller3_price - env.seller3_min_price
                    weighted_seller3_profit = seller3_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller3 Step Reward = seller_profit({seller3_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller3_reward']:.2f} (seller3_total_price={seller3_price:.2f}, seller3_min={env.seller3_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller3 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller3_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer') and info.get('selected_seller'):
                print(f"Selected Deal: Buyer {info['selected_buyer']} - Seller {info['selected_seller']}")
                print(f"Final Deal Total Price: ${info.get('final_deal_price', 0):.2f}")
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
    print("\nSequential multi-buyer multi-seller multi-product negotiation completed!")


if __name__ == "__main__":
    main()
