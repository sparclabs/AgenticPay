"""Batch evaluation script for single buyer product seller tasks

This script runs all three negotiation tasks (Task1, Task2, Task3) in batch mode,
captures results, and saves them to the results directory.
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
# Script is at: agenticpaygym/examples/single_buyer_product_seller/batch_evaluate.py
# Need to go up 4 levels to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from agenticpaygym import make
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.models.custom_llm import CustomLLM
from agenticpaygym.examples.config import reward_weights, max_rounds, price_tolerance


# Model list configuration - models to evaluate in batch mode
# If --model is specified via command line, only that model will be used
# Otherwise, all models in this list will be evaluated
MODEL_LIST = [
    # "gpt-5.2",
    # "gemini-3-pro-all",
    # "claude-sonnet-4-5-20250929",
    "360/deepseek-r1",
    "qwen3-8b",
    "qwen3-14b"
    # Add more models as needed
]


def run_task1(model, user_requirement=None, output_file=None):
    """Run Task1: Basic Price Negotiation
    
    Args:
        model: The LLM model to use
        user_requirement: User requirement string (default if None)
        output_file: File to write output to (optional)
    
    Returns:
        dict: Results dictionary containing all relevant information
    """
    if user_requirement is None:
        user_requirement = "I need a high-quality winter jacket for cold weather"
    
    print(f"\n{'='*80}")
    print("Running Task1: Basic Price Negotiation")
    print(f"{'='*80}")
    
    # Capture stdout
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    results = {
        "task": "Task1_basic_price_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "status": "unknown",
        "success": False,
        "output": "",
        "error": None,
    }
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Create agents
            buyer_max_price = 150.0
            seller_min_price = 80.0
            
            buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
            seller = SellerAgent(model=model, seller_min_price=seller_min_price)
            
            # Create environment
            env = make(
                "Task1_basic_price_negotiation-v0",
                buyer_agent=buyer,
                seller_agent=seller,
                max_rounds=max_rounds,
                initial_seller_price=150.0,
                buyer_max_price=buyer_max_price,
                seller_min_price=seller_min_price,
                environment_info={
                    "temperature": "warm",
                    "season": "summer",
                    "weather": "sunny",
                },
                price_tolerance=price_tolerance,
                reward_weights=reward_weights,
            )
            
            # User profile
            user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
            
            # Reset environment
            observation, info = env.reset(
                user_requirement=user_requirement,
                product_info={
                    "name": "Premium Winter Jacket",
                    "brand": "Mountain Gear",
                    "price": 180.0,
                    "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                    "condition": "New",
                    "material": "Gore-Tex",
                },
                user_profile=user_profile,
            )
            
            # Negotiation loop
            done = False
            start_time = time.time()
            
            while not done:
                # Buyer responds
                buyer_action = buyer.respond(
                    conversation_history=observation["conversation_history"],
                    current_state=observation
                )
                
                # Update conversation history
                updated_conversation_history = observation["conversation_history"].copy()
                if buyer_action:
                    current_round = observation.get("current_round", 0)
                    updated_conversation_history.append({
                        "role": "buyer",
                        "content": buyer_action,
                        "round": current_round
                    })
                
                # Seller responds
                seller_action = seller.respond(
                    conversation_history=updated_conversation_history,
                    current_state=observation
                )
                
                # Execute step
                observation, reward, terminated, truncated, info = env.step(
                    buyer_action=buyer_action,
                    seller_action=seller_action
                )
                done = terminated or truncated
                
                if done:
                    break
            
            elapsed_time = time.time() - start_time
            
            # Extract results
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": info.get('agreed_price'),
                "total_rounds": info.get('round', 0),
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
            })
            
            env.close()
            
    except Exception as e:
        results["error"] = str(e)
        results["status"] = "error"
        import traceback
        results["traceback"] = traceback.format_exc()
    
    finally:
        # Get captured output
        results["output"] = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        if error_output:
            results["error_output"] = error_output
    
    # Write output to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results["output"])
    
    return results


def run_task2(model, user_requirement=None, output_file=None):
    """Run Task2: Close Price Negotiation
    
    Args:
        model: The LLM model to use
        user_requirement: User requirement string (default if None)
        output_file: File to write output to (optional)
    
    Returns:
        dict: Results dictionary containing all relevant information
    """
    if user_requirement is None:
        user_requirement = "I need a high-quality winter jacket for cold weather"
    
    print(f"\n{'='*80}")
    print("Running Task2: Close Price Negotiation")
    print(f"{'='*80}")
    
    # Capture stdout
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    results = {
        "task": "Task2_close_price_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "status": "unknown",
        "success": False,
        "output": "",
        "error": None,
    }
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Create agents - close prices
            seller_min_price = 80.0
            buyer_max_price = 85.0  # Close to seller_min_price
            
            buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
            seller = SellerAgent(model=model, seller_min_price=seller_min_price)
            
            # Create environment
            env = make(
                "Task2_close_price_negotiation-v0",
                buyer_agent=buyer,
                seller_agent=seller,
                max_rounds=max_rounds,
                initial_seller_price=100.0,
                buyer_max_price=buyer_max_price,
                seller_min_price=seller_min_price,
                environment_info={
                    "temperature": "warm",
                    "season": "summer",
                    "weather": "sunny",
                },
                price_tolerance=price_tolerance,
                reward_weights=reward_weights,
            )
            
            # User profile
            user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
            
            # Reset environment
            observation, info = env.reset(
                user_requirement=user_requirement,
                product_info={
                    "name": "Premium Winter Jacket",
                    "brand": "Mountain Gear",
                    "price": 180.0,
                    "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                    "condition": "New",
                    "material": "Gore-Tex",
                },
                user_profile=user_profile,
            )
            
            # Negotiation loop
            done = False
            start_time = time.time()
            
            while not done:
                # Buyer responds
                buyer_action = buyer.respond(
                    conversation_history=observation["conversation_history"],
                    current_state=observation
                )
                
                # Update conversation history
                updated_conversation_history = observation["conversation_history"].copy()
                if buyer_action:
                    current_round = observation.get("current_round", 0)
                    updated_conversation_history.append({
                        "role": "buyer",
                        "content": buyer_action,
                        "round": current_round
                    })
                
                # Seller responds
                seller_action = seller.respond(
                    conversation_history=updated_conversation_history,
                    current_state=observation
                )
                
                # Execute step
                observation, reward, terminated, truncated, info = env.step(
                    buyer_action=buyer_action,
                    seller_action=seller_action
                )
                done = terminated or truncated
                
                if done:
                    break
            
            elapsed_time = time.time() - start_time
            
            # Extract results
            agreed_price = info.get('agreed_price')
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": agreed_price,
                "total_rounds": info.get('round', 0),
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
            })
            
            # Task2 specific analysis
            if agreed_price:
                results["seller_profit"] = agreed_price - seller_min_price
                results["buyer_savings"] = buyer_max_price - agreed_price
            
            env.close()
            
    except Exception as e:
        results["error"] = str(e)
        results["status"] = "error"
        import traceback
        results["traceback"] = traceback.format_exc()
    
    finally:
        # Get captured output
        results["output"] = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        if error_output:
            results["error_output"] = error_output
    
    # Write output to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results["output"])
    
    return results


def run_task3(model, user_requirement=None, output_file=None):
    """Run Task3: Close to Market Price Negotiation
    
    Args:
        model: The LLM model to use
        user_requirement: User requirement string (default if None)
        output_file: File to write output to (optional)
    
    Returns:
        dict: Results dictionary containing all relevant information
    """
    if user_requirement is None:
        user_requirement = "I need a high-quality winter jacket for cold weather"
    
    print(f"\n{'='*80}")
    print("Running Task3: Close to Market Price Negotiation")
    print(f"{'='*80}")
    
    # Capture stdout
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    results = {
        "task": "Task3_close_to_market_price_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "status": "unknown",
        "success": False,
        "output": "",
        "error": None,
    }
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Market price
            market_price = 180.0
            
            # Create agents - seller_min_price close to market price
            seller_min_price = 175.0  # Close to market price
            buyer_max_price = 200.0
            
            buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
            seller = SellerAgent(model=model, seller_min_price=seller_min_price)
            
            # Create environment
            env = make(
                "Task3_close_to_market_price_negotiation-v0",
                buyer_agent=buyer,
                seller_agent=seller,
                max_rounds=max_rounds,
                initial_seller_price=185.0,
                buyer_max_price=buyer_max_price,
                seller_min_price=seller_min_price,
                environment_info={
                    "temperature": "warm",
                    "season": "summer",
                    "weather": "sunny",
                },
                price_tolerance=price_tolerance,
                reward_weights=reward_weights,
            )
            
            # User profile
            user_profile = "User prefers business/professional style and likes to compare prices before making purchases. In negotiations, they may mention comparing other options and seek better deals."
            
            # Reset environment
            observation, info = env.reset(
                user_requirement=user_requirement,
                product_info={
                    "name": "Premium Winter Jacket",
                    "brand": "Mountain Gear",
                    "price": market_price,
                    "features": ["Waterproof", "Insulated", "Windproof", "Breathable"],
                    "condition": "New",
                    "material": "Gore-Tex",
                },
                user_profile=user_profile,
            )
            
            # Negotiation loop
            done = False
            start_time = time.time()
            
            while not done:
                # Buyer responds
                buyer_action = buyer.respond(
                    conversation_history=observation["conversation_history"],
                    current_state=observation
                )
                
                # Update conversation history
                updated_conversation_history = observation["conversation_history"].copy()
                if buyer_action:
                    current_round = observation.get("current_round", 0)
                    updated_conversation_history.append({
                        "role": "buyer",
                        "content": buyer_action,
                        "round": current_round
                    })
                
                # Seller responds
                seller_action = seller.respond(
                    conversation_history=updated_conversation_history,
                    current_state=observation
                )
                
                # Execute step
                observation, reward, terminated, truncated, info = env.step(
                    buyer_action=buyer_action,
                    seller_action=seller_action
                )
                done = terminated or truncated
                
                if done:
                    break
            
            elapsed_time = time.time() - start_time
            
            # Extract results
            agreed_price = info.get('agreed_price')
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": agreed_price,
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "seller_reward": info.get('seller_reward'),
                "buyer_reward": info.get('buyer_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "market_price": market_price,
                "buyer_max_price": buyer_max_price,
                "seller_min_price": seller_min_price,
            })
            
            # Task3 specific analysis
            if agreed_price:
                results["seller_profit"] = agreed_price - seller_min_price
                results["buyer_savings"] = buyer_max_price - agreed_price
                results["price_vs_market"] = agreed_price - market_price
                results["seller_min_vs_market"] = seller_min_price - market_price
            
            env.close()
            
    except Exception as e:
        results["error"] = str(e)
        results["status"] = "error"
        import traceback
        results["traceback"] = traceback.format_exc()
    
    finally:
        # Get captured output
        results["output"] = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        if error_output:
            results["error_output"] = error_output
    
    # Write output to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results["output"])
    
    return results


def evaluate_model(model_name, api_key, user_requirement, results_dir, task_name):
    """Evaluate a single model on all tasks
    
    Args:
        model_name: Name of the model to evaluate
        api_key: API key for the model
        user_requirement: User requirement string
        results_dir: Base results directory
        task_name: Task name directory
    
    Returns:
        dict: Results dictionary for this model
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*80}")
    
    # Initialize model
    print("Initializing model...")
    try:
        model = CustomLLM(api_key=api_key, model=model_name)
        print(f"✓ Successfully initialized: {model}")
    except Exception as e:
        print(f"✗ Failed to initialize model {model_name}: {e}")
        return {
            "model": model_name,
            "error": f"Model initialization failed: {str(e)}",
            "status": "error"
        }
    
    # Create model directory
    # Sanitize model name for filesystem (replace special characters)
    model_name_safe = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    task_dir = results_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    model_dir = task_dir / model_name_safe
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / f"batch_evaluation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    
    # Run all tasks
    all_results = {}
    
    try:
        # Task 1
        print(f"\nRunning Task1...")
        task1_output_file = run_dir / "Task1_output.txt"
        task1_results = run_task1(model, user_requirement=user_requirement, output_file=task1_output_file)
        all_results["Task1"] = task1_results
        
        # Task 2
        print(f"\nRunning Task2...")
        task2_output_file = run_dir / "Task2_output.txt"
        task2_results = run_task2(model, user_requirement=user_requirement, output_file=task2_output_file)
        all_results["Task2"] = task2_results
        
        # Task 3
        print(f"\nRunning Task3...")
        task3_output_file = run_dir / "Task3_output.txt"
        task3_results = run_task3(model, user_requirement=user_requirement, output_file=task3_output_file)
        all_results["Task3"] = task3_results
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        all_results["error"] = str(e)
        all_results["traceback"] = traceback.format_exc()
    
    # Add metadata to results
    all_results["metadata"] = {
        "model": model_name,
        "user_requirement": user_requirement,
        "timestamp": datetime.now().isoformat(),
        "max_rounds": max_rounds,
        "price_tolerance": price_tolerance,
        "reward_weights": reward_weights,
    }
    
    # Save summary JSON
    summary_file = run_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print summary for this model
    print(f"\n{'='*80}")
    print(f"MODEL EVALUATION SUMMARY: {model_name}")
    print(f"{'='*80}")
    print(f"Results saved to: {run_dir}")
    print(f"\nTask Results:")
    
    for task_name_key, task_result in all_results.items():
        if task_name_key == "metadata" or task_name_key == "error" or task_name_key == "traceback":
            continue
        print(f"\n{task_name_key}:")
        print(f"  Status: {task_result.get('status', 'unknown')}")
        print(f"  Success: {task_result.get('success', False)}")
        print(f"  Total Rounds: {task_result.get('total_rounds', 0)}")
        print(f"  Elapsed Time: {task_result.get('elapsed_time', 0):.2f}s")
        if task_result.get('agreed_price'):
            print(f"  Agreed Price: ${task_result['agreed_price']:.2f}")
        if task_result.get('total_reward') is not None:
            print(f"  Total Reward: {task_result['total_reward']:.3f}")
        if task_result.get('error'):
            print(f"  Error: {task_result['error']}")
    
    if "error" in all_results:
        print(f"\n  Overall Error: {all_results['error']}")
    
    print(f"{'='*80}\n")
    
    return all_results


def main():
    """Main function: Run all tasks in batch mode and save results"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Batch evaluation script for single buyer product seller tasks"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (if not specified, all models in MODEL_LIST will be evaluated). Options: gpt-5.2, claude-sonnet-4-5-20250929, gemini-3-pro-all, gpt-3.5-turbo, DeepSeek-R1"
    )
    parser.add_argument(
        "--user-requirement",
        type=str,
        default="I need a high-quality winter jacket for cold weather",
        help="User requirement for the negotiation (default: 'I need a high-quality winter jacket for cold weather')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: agenticpaygym/results)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the model (default: uses OPENAI_API_KEY environment variable)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key not provided.")
        print("Please set OPENAI_API_KEY environment variable or use --api-key argument.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Determine which models to evaluate
    if args.model:
        # Single model specified via command line
        models_to_evaluate = [args.model]
        print(f"Single model mode: {args.model}")
    else:
        # Use model list from configuration
        models_to_evaluate = MODEL_LIST
        print(f"Batch mode: Evaluating {len(models_to_evaluate)} models")
        print(f"Models: {', '.join(models_to_evaluate)}")
    
    # Create results directory structure
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = Path(project_root) / "agenticpaygym" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Task name
    task_name = "single_buyer_product_seller"
    
    # User requirement
    user_requirement = args.user_requirement
    
    # Evaluate each model
    all_models_results = {}
    start_time = time.time()
    
    for i, model_name in enumerate(models_to_evaluate, 1):
        print(f"\n{'#'*80}")
        print(f"MODEL {i}/{len(models_to_evaluate)}: {model_name}")
        print(f"{'#'*80}")
        
        model_results = evaluate_model(
            model_name=model_name,
            api_key=api_key,
            user_requirement=user_requirement,
            results_dir=results_dir,
            task_name=task_name
        )
        all_models_results[model_name] = model_results
    
    total_time = time.time() - start_time
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL BATCH EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total models evaluated: {len(models_to_evaluate)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"\nModel Results:")
    
    for model_name, model_result in all_models_results.items():
        print(f"\n{model_name}:")
        if "error" in model_result and model_result["error"] and not any(
            k.startswith("Task") for k in model_result.keys()
        ):
            print(f"  Status: ERROR - {model_result['error']}")
        else:
            success_count = sum(
                1 for k, v in model_result.items()
                if k.startswith("Task") and isinstance(v, dict) and v.get("success", False)
            )
            print(f"  Successful tasks: {success_count}/3")
            for task_key in ["Task1", "Task2", "Task3"]:
                if task_key in model_result:
                    task_result = model_result[task_key]
                    status = task_result.get("status", "unknown")
                    success = task_result.get("success", False)
                    rounds = task_result.get("total_rounds", 0)
                    print(f"    {task_key}: {status} ({'✓' if success else '✗'}) - {rounds} rounds")
    
    print(f"\n{'='*80}")
    print("Batch evaluation completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
