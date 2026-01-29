"""Task2 Close Price Negotiation Example

Tests negotiation scenarios where buyer_max_price is close to seller_min_price
to see if a deal can be reached. This is useful for testing edge cases in negotiation.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add project path
# Script is at: agenticpay/examples/single_buyer_product_seller/Task2_close_price_negotiation.py
# Need to go up 4 levels to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from agenticpay import make, Task2ClosePriceNegotiation  # Use registration system
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.custom_llm import CustomLLM
from agenticpay.models.qwen3_vl import Qwen3VL
from agenticpay.models.vllm_lm import VLLMLLM
from agenticpay.models.sglang_vlm import SGLangVLM
from agenticpay.examples.config import reward_weights, max_rounds, price_tolerance


def get_model_name(model):
    """Extract model name from model object
    
    Args:
        model: Model object (CustomLLM, VLLMLLM, etc.)
    
    Returns:
        str: Model name
    """
    if hasattr(model, 'model'):
        return model.model
    elif hasattr(model, 'model_id'):
        return model.model_id
    elif hasattr(model, 'model_path'):
        # Extract model name from path
        model_path = model.model_path
        return os.path.basename(model_path) if model_path else str(model)
    else:
        # Fallback to string representation, but try to extract model name
        model_str = str(model)
        # Try to extract model name from string like "CustomLLM(model=qwen3-8b)"
        if "model=" in model_str:
            try:
                return model_str.split("model=")[1].split(")")[0]
            except:
                return model_str
        else:
            return model_str


def main(model_name=None):
    """Main function: Tests negotiation with buyer_max_price close to seller_min_price
    
    Args:
        model_name: Optional model name. If None, uses default model.
    """
    
    print("Initializing model...")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Use provided model name or default
    if model_name is None:
        model_name = "qwen3-8b"  # Default model
    
    model = CustomLLM(api_key=api_key, model=model_name) # claude-sonnet-4-5-20250929, gpt-5.2, gemini-3-pro-all, gpt-3.5-turbo, DeepSeek-R1

    # Build absolute path to model directory
    # model_path = os.path.join(project_root, "models", "download_models", "Qwen3-8B-Instruct")
    # model_path = os.path.abspath(model_path)

    # vLLM LLM Model
    # model = VLLMLLM(
    #     model_path=model_path,
    #     trust_remote_code=True,
    #     gpu_memory_utilization=0.9,
    #     tensor_parallel_size=4, # 4 GPUs
    # )

    # SGLang VLM Model
    # model = SGLangVLM(
    #     model_path=model_path,
    # )

    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # Task2: buyer_max_price is close to seller_min_price to test if deal can be reached
    print("Creating agents...")
    seller_min_price = 80.0  # Minimum acceptable selling price for seller (confidential)
    buyer_max_price = 85.0   # Maximum acceptable purchase price for buyer (confidential) - close to seller_min_price
    
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_price)
    
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
        max_rounds=max_rounds,
        initial_seller_price=100.0,  # Initial price offered by seller
        buyer_max_price=buyer_max_price,  # Buyer bottom price (confidential)
        seller_min_price=seller_min_price,  # Seller bottom price (confidential)
        environment_info={
            "temperature": "warm",
            "season": "summer",
            "weather": "sunny",
        },
        price_tolerance=price_tolerance,  # Tolerance for agreement
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
    # print("\n" + "="*60)
    # print("Please enter the product requirement you want to purchase:")
    # user_requirement = input("> ").strip()
    # if not user_requirement:
    #     print("No requirement entered, using default requirement...")
    #     user_requirement = "I need a high-quality winter jacket for cold weather"

    user_requirement = "I need a high-quality winter jacket for cold weather"
    print(f"Using default requirement: {user_requirement}")
    
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
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task2_close_price_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round: buyer responds first, then seller responds (seeing buyer's message)
        # Get buyer's response
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Create updated conversation history that includes buyer's response
        # So seller can see buyer's message before responding
        updated_conversation_history = observation["conversation_history"].copy()
        if buyer_action:
            current_round = observation.get("current_round", 0)
            updated_conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })
        
        # Get seller's response (seller can now see buyer's message)
        seller_action = seller.respond(
            conversation_history=updated_conversation_history,
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
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
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
        
        # If this is the final round (agreed or timeout), display score calculations after Step Rewards
        if done:
            # Print score calculations after Step Rewards
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()
            
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            seller_price = info.get('seller_price')
            buyer_price = info.get('buyer_price')
            seller_price_str = f"${seller_price:.2f}" if seller_price is not None else "Not specified"
            buyer_price_str = f"${buyer_price:.2f}" if buyer_price is not None else "Not specified"
            print(f"Final Prices: Seller={seller_price_str} | Buyer={buyer_price_str}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Total Reward: {reward:.3f}")
            if 'seller_reward' in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
            if 'buyer_reward' in info:
                print(f"Buyer Reward: {info['buyer_reward']:.3f}")
            if 'global_score' in info:
                print(f"GlobalScore: {info['global_score']:.3f}")
            if 'buyer_score' in info:
                print(f"BuyerScore: {info['buyer_score']:.3f}")
            if 'seller_score' in info:
                print(f"SellerScore: {info['seller_score']:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            
            # Task2 specific analysis
            agreed_price = info.get('agreed_price')
            if agreed_price:
                print(f"\nTask2 Analysis:")
                print(f"  - Agreed Price: ${agreed_price:.2f}")
                print(f"  - Seller Min Price: ${seller_min_price:.2f}")
                print(f"  - Buyer Max Price: ${buyer_max_price:.2f}")
                print(f"  - Deal Reached: {'YES' if terminated else 'NO'}")
                if terminated:
                    seller_profit = agreed_price - seller_min_price
                    buyer_savings = buyer_max_price - agreed_price
                    print(f"  - Seller Profit: ${seller_profit:.2f}")
                    print(f"  - Buyer Savings: ${buyer_savings:.2f}")
            print("="*60)
            
            # Collect results
            elapsed_time = time.time() - start_time
            # current_round has been incremented to reflect the completed round
            actual_rounds = info.get('round', 0)
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": agreed_price,
                "total_rounds": actual_rounds,
                "total_reward": float(reward) if reward is not None else None,
                "seller_reward": info.get('seller_reward'),
                "buyer_reward": info.get('buyer_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer_max_price": buyer_max_price,
                "seller_min_price": seller_min_price,
                "product_info": {
                    "name": "Premium Winter Jacket",
                    "brand": "Mountain Gear",
                    "price": 180.0,
                    "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                    "condition": "New",
                    "material": "Gore-Tex",
                },
                "model": get_model_name(model),
            })
            
            # Task2 specific analysis
            if agreed_price:
                results["seller_profit"] = agreed_price - seller_min_price
                results["buyer_savings"] = buyer_max_price - agreed_price
            
            break
    
    # Close environment
    env.close()
    print("\nNegotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "single_buyer_product_seller"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model name for directory (sanitize for filesystem)
        model_name = get_model_name(model)
        model_name_safe = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        model_dir = results_dir / model_name_safe
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"batch_evaluation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary JSON
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save output text
        output_file = run_dir / "Task2_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task2: Close Price Negotiation Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"User Requirement: {results['user_requirement']}\n")
            f.write(f"User Profile: {results['user_profile']}\n\n")
            f.write(f"Status: {results['status']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Total Rounds: {results['total_rounds']}\n")
            elapsed_time = results.get('elapsed_time', 0)
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Seller Price: ${results['seller_price']:.2f}" if results.get('seller_price') else "  Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Buyer Price: ${results['buyer_price']:.2f}" if results.get('buyer_price') else "  Buyer Price: Not specified")
            f.write("\n")
            if results.get('agreed_price'):
                f.write(f"  Agreed Price: ${results['agreed_price']:.2f}\n")
            f.write("\n")
            f.write("Rewards:\n")
            if results.get('total_reward') is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get('seller_reward') is not None:
                f.write(f"  Seller Reward: {results['seller_reward']:.3f}\n")
            if results.get('buyer_reward') is not None:
                f.write(f"  Buyer Reward: {results['buyer_reward']:.3f}\n")
            f.write("\n")
            f.write("Scores:\n")
            if results.get('global_score') is not None:
                f.write(f"  Global Score: {results['global_score']:.3f}\n")
            if results.get('buyer_score') is not None:
                f.write(f"  Buyer Score: {results['buyer_score']:.3f}\n")
            if results.get('seller_score') is not None:
                f.write(f"  Seller Score: {results['seller_score']:.3f}\n")
            f.write("\n")
            f.write("Task2 Analysis:\n")
            f.write(f"  Seller Min Price: ${results['seller_min_price']:.2f}\n")
            f.write(f"  Buyer Max Price: ${results['buyer_max_price']:.2f}\n")
            f.write(f"  Price Difference: ${results['buyer_max_price'] - results['seller_min_price']:.2f}\n")
            if results.get('agreed_price'):
                f.write(f"  Deal Reached: {'YES' if results['success'] else 'NO'}\n")
                if results.get('seller_profit') is not None:
                    f.write(f"  Seller Profit: ${results['seller_profit']:.2f}\n")
                if results.get('buyer_savings') is not None:
                    f.write(f"  Buyer Savings: ${results['buyer_savings']:.2f}\n")
            f.write("\n")
            if results.get('termination_reason'):
                f.write(f"Termination Reason: {results['termination_reason']}\n")
            if results.get('error'):
                f.write(f"\nError: {results['error']}\n")
        
        print(f"\nResults saved to: {run_dir}")
        print(f"  - Summary JSON: {summary_file}")
        print(f"  - Output Text: {output_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task2: Close Price Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

