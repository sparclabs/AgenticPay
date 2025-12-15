"""Task4 Sequential Three-Buyer Negotiation Example

Demonstrates how to use the Task4SequentialThreeBuyerNegotiation to negotiate sequentially with three buyers,
where seller chooses one buyer per round to negotiate with.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym.envs.only_multi_buyer.Task4_sequential_three_buyer_negotiation import Task4SequentialThreeBuyerNegotiation
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM
import re


def extract_buyer_choice(seller_response: str, observation: dict) -> int:
    """Extract buyer choice from seller's response
    
    Seller should indicate which buyer they want to negotiate with.
    Look for patterns like "buyer 1", "buyer1", "first buyer", etc.
    
    Args:
        seller_response: Seller's response text
        observation: Current observation from environment
        
    Returns:
        1, 2, or 3, indicating which buyer seller wants to negotiate with
    """
    response_lower = seller_response.lower()
    
    # Look for explicit buyer mentions
    if re.search(r'buyer\s*[123]|first\s+buyer|buyer\s*one|second\s+buyer|buyer\s*two|third\s+buyer|buyer\s*three', response_lower):
        if re.search(r'buyer\s*3|third\s+buyer|buyer\s*three', response_lower):
            return 3
        elif re.search(r'buyer\s*2|second\s+buyer|buyer\s*two', response_lower):
            return 2
        elif re.search(r'buyer\s*1|first\s+buyer|buyer\s*one', response_lower):
            return 1
    
    # If no explicit mention, try to infer from context
    # Check if seller mentions prices or other indicators
    buyer1_price = observation.get("buyer1_price")
    buyer2_price = observation.get("buyer2_price")
    buyer3_price = observation.get("buyer3_price")
    seller_price_buyer1 = observation.get("seller_price_buyer1")
    seller_price_buyer2 = observation.get("seller_price_buyer2")
    seller_price_buyer3 = observation.get("seller_price_buyer3")
    
    # If seller mentions a specific price, try to match it
    price_match = re.search(r'\$?(\d+\.?\d*)', seller_response)
    if price_match:
        mentioned_price = float(price_match.group(1))
        if seller_price_buyer1 is not None and abs(mentioned_price - seller_price_buyer1) < 5:
            return 1
        elif seller_price_buyer2 is not None and abs(mentioned_price - seller_price_buyer2) < 5:
            return 2
        elif seller_price_buyer3 is not None and abs(mentioned_price - seller_price_buyer3) < 5:
            return 3
    
    # Default: if no clear indication, check which buyer has been negotiated with more
    # or which has a better price (higher buyer price is better for seller)
    prices = []
    if buyer1_price is not None:
        prices.append((1, buyer1_price))
    if buyer2_price is not None:
        prices.append((2, buyer2_price))
    if buyer3_price is not None:
        prices.append((3, buyer3_price))
    
    if prices:
        # Choose the one with highest price if available
        prices.sort(key=lambda x: x[1], reverse=True)
        return prices[0][0]
    
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
    llm = OpenAILLM(api_key=api_key, model="gpt-4o-mini-2024-07-18")  # gpt-4o-mini-2024-07-18, gpt-3.5-turbo
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    print("Creating agents...")
    buyer1_max_price = 120.0  # Maximum acceptable purchase price for buyer1 (confidential)
    buyer2_max_price = 130.0  # Maximum acceptable purchase price for buyer2 (confidential, different from buyer1)
    buyer3_max_price = 125.0  # Maximum acceptable purchase price for buyer3 (confidential, different from buyer1 and buyer2)
    seller_min_price = 80.0  # Minimum acceptable selling price for seller (confidential)
    
    buyer1 = BuyerAgent(llm=llm, buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(llm=llm, buyer_max_price=buyer2_max_price)
    buyer3 = BuyerAgent(llm=llm, buyer_max_price=buyer3_max_price)
    seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)
    
    # Create environment
    print("Creating sequential multi-buyer negotiation environment...")
    env = Task4SequentialThreeBuyerNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        buyer3_agent=buyer3,
        seller_agent=seller,
        max_rounds=20,
        initial_seller_price=150.0,  # Initial price offered by seller
        buyer1_max_price=buyer1_max_price,  # Buyer1 bottom price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 bottom price (confidential)
        buyer3_max_price=buyer3_max_price,  # Buyer3 bottom price (confidential)
        seller_min_price=seller_min_price,  # Seller bottom price (confidential)
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=5.0,
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
    print("Starting new sequential negotiation with three buyers...")
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
        # Seller can see all three buyers' information in the observation
        # Let seller decide which buyer to negotiate with and provide negotiation message
        # We'll use a combined conversation history that includes all three buyers' conversations
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
        # Add buyer3 messages with prefix
        for msg in observation.get("conversation_history_buyer3", []):
            combined_history.append({
                **msg,
                "content": f"[Buyer 3] {msg['content']}"
            })
        
        # Get seller's response - seller should indicate which buyer they want to negotiate with
        seller_response = seller.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": "You are negotiating with three buyers. Each round, you need to choose ONE buyer to negotiate with and provide your negotiation message. Please clearly indicate which buyer (1, 2, or 3) you want to negotiate with, for example: 'I want to negotiate with buyer 1' or 'Let me talk to buyer 2' or 'I'll negotiate with buyer 3'."
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
        elif selected_buyer == 2:
            conversation_history = observation["conversation_history_buyer2"]
        else:  # selected_buyer == 3
            conversation_history = observation["conversation_history_buyer3"]
        
        # Get the selected buyer's response
        if selected_buyer == 1:
            buyer_action = buyer1.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        elif selected_buyer == 2:
            buyer_action = buyer2.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        else:  # selected_buyer == 3
            buyer_action = buyer3.respond(
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
            buyer3_price = info.get('buyer3_price', 0) or 0
            seller_price_buyer3 = info.get('seller_price_buyer3', 0) or 0
            print(f"Buyer1 Prices: Buyer=${buyer1_price:.2f} | Seller=${seller_price_buyer1:.2f}")
            print(f"Buyer2 Prices: Buyer=${buyer2_price:.2f} | Seller=${seller_price_buyer2:.2f}")
            print(f"Buyer3 Prices: Buyer=${buyer3_price:.2f} | Seller=${seller_price_buyer3:.2f}")
            print(f"Total Rounds: {info['round']}")
            print(f"Reward: {reward:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            break
    
    # Close environment
    env.close()
    print("\nNegotiation completed!")


if __name__ == "__main__":
    main()

