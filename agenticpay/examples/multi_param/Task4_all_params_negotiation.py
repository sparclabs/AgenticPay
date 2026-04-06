"""Task4: Full 5-Parameter Negotiation Example

The most complex scenario — agents negotiate on all five parameters:
  price, quality, delivery, warranty, and payment schedule.

New tension vs Task3: Payment schedule requires exact-match agreement.
The seller wants upfront cash; the buyer wants installments for cash-flow
reasons. One party must yield completely, or both converge on "30-day net".

This task benchmarks whether agents can handle simultaneous trade-offs
across five dimensions and still reach a Pareto-efficient deal.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from agenticpay import make
from agenticpay.agents.multi_param_buyer_agent import MultiParamBuyerAgent
from agenticpay.agents.multi_param_seller_agent import MultiParamSellerAgent
from agenticpay.envs.multi_param.base_multi_param_negotiation import ParamPreferences
from agenticpay.examples.config import get_model, get_model_name


PRODUCT_INFO = {
    "name": "Atlas Copco GA 55 Industrial Air Compressor",
    "description": "55 kW rotary screw air compressor, 10 bar, 380 V — for manufacturing plant",
    "list_price": 28000.0,
    "features": ["Variable speed drive", "IE4 motor", "SmartLink remote monitoring"],
    "condition": "New",
}

SCENARIO_PARAMS = {
    # Tightest scenario: 5 parameters all under real pressure simultaneously.
    # Price ZOPA: $21k–$23.5k. Delivery: buyer needs ≤14 days, seller standard = 28.
    # Warranty: buyer needs 24mo, seller standard = 12mo.
    # Payment: exact-match required — seller wants upfront, buyer wants installments.
    # No single parameter can be conceded without affecting the others.
    "price_range":      (21000.0, 23500.0),
    "quality_options":  ["Standard", "Premium", "Luxury"],
    "delivery_range":   (7, 30),
    "warranty_range":   (12, 36),
    "payment_options":  ["upfront", "30-day", "installments"],
}

# Buyer: wants low price, high quality, fast delivery, long warranty, installments
BUYER_PREFERENCES = ParamPreferences(
    price_limit=23500.0,
    min_quality="Premium",
    max_delivery_days=14,
    min_warranty_months=24,
    preferred_payment="installments",
    price_weight=0.35,
    quality_weight=0.30,
    delivery_weight=0.20,
    warranty_weight=0.10,
    payment_weight=0.05,
)

# Seller: wants high price, standard quality, slow delivery, short warranty, upfront payment
SELLER_PREFERENCES = ParamPreferences(
    price_limit=21000.0,
    min_quality="Standard",
    max_delivery_days=28,
    min_warranty_months=12,
    preferred_payment="upfront",
    price_weight=0.40,
    quality_weight=0.20,
    delivery_weight=0.15,
    warranty_weight=0.10,
    payment_weight=0.15,
)

ACTIVE_PARAMS = ["price", "quality", "delivery_days", "warranty_months", "payment"]
TASK_NAME     = "Task4_all_params_negotiation"
ENV_ID        = "Task4_all_params_negotiation-v0"


def main(model_name=None, difficulty="normal"):
    print("Initializing model...")
    model = get_model()
    print(f"✓ Model: {model}")

    buyer = MultiParamBuyerAgent(
        model=model,
        preferences=BUYER_PREFERENCES,
        active_params=ACTIVE_PARAMS,
    )
    seller = MultiParamSellerAgent(
        model=model,
        preferences=SELLER_PREFERENCES,
        active_params=ACTIVE_PARAMS,
    )

    env = make(
        ENV_ID,
        buyer_agent=buyer,
        seller_agent=seller,
        buyer_preferences=BUYER_PREFERENCES,
        seller_preferences=SELLER_PREFERENCES,
        scenario_params=SCENARIO_PARAMS,
        max_rounds=20,
        environment_info={"sector": "manufacturing", "region": "North America"},
    )

    user_requirement = (
        "We are looking to procure a heavy-duty industrial air compressor for a new production line. "
        "We have preferences across price, quality, delivery, warranty, and payment terms, "
        "and are open to trade-offs across all five dimensions."
    )
    user_profile = (
        "Senior procurement manager at a large manufacturing firm. "
        "Evaluates trade-offs across price, quality, delivery, warranty, and payment "
        "to arrive at the best overall deal."
    )
    print(f"\nUser requirement: {user_requirement}")

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=PRODUCT_INFO,
        user_profile=user_profile,
    )

    print("\n" + "="*60)
    print(f"Starting {TASK_NAME}")
    print(f"Active parameters: {ACTIVE_PARAMS}")
    print("="*60)

    done = False
    start_time = time.time()
    results = {
        "task": TASK_NAME,
        "active_params": ACTIVE_PARAMS,
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "success": False,
        "error": None,
    }

    while not done:
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation,
        )

        updated_history = observation["conversation_history"].copy()
        if buyer_action:
            updated_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": observation.get("current_round", 0),
            })

        seller_action = seller.respond(
            conversation_history=updated_history,
            current_state=observation,
        )

        observation, reward, terminated, truncated, info = env.step(
            buyer_action=buyer_action,
            seller_action=seller_action,
        )
        done = terminated or truncated

        env.render()
        sys.stdout.flush()

        if "step_seller_reward" in info or "step_buyer_reward" in info:
            print(
                f"[Step Rewards] "
                f"Seller: {info.get('step_seller_reward', 0):.3f} | "
                f"Buyer: {info.get('step_buyer_reward', 0):.3f}"
            )

        if done:
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()

            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            print(f"Total Rounds: {info['round']}")
            print(f"Total Reward: {reward:.3f}")
            if "seller_reward" in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
            if "buyer_reward" in info:
                print(f"Buyer Reward: {info['buyer_reward']:.3f}")
            if "global_score" in info:
                print(f"GlobalScore: {info['global_score']:.3f}")
            if "buyer_score" in info:
                print(f"BuyerScore: {info['buyer_score']:.3f}")
            if "seller_score" in info:
                print(f"SellerScore: {info['seller_score']:.3f}")
            if info.get("agreed_offer"):
                print("\nAgreed Deal Terms:")
                for param, val in info["agreed_offer"].items():
                    print(f"  {param}: {val}")
            if info.get("termination_reason"):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)

            conversation_history = env.memory.get_history()
            results.update({
                "status":               info.get("status", "unknown"),
                "success":              terminated,
                "total_rounds":         info.get("round", 0),
                "total_reward":         float(reward) if reward is not None else None,
                "seller_reward":        info.get("seller_reward"),
                "buyer_reward":         info.get("buyer_reward"),
                "global_score":         info.get("global_score"),
                "buyer_score":          info.get("buyer_score"),
                "seller_score":         info.get("seller_score"),
                "agreed_offer":         info.get("agreed_offer"),
                "termination_reason":   info.get("termination_reason"),
                "elapsed_time":         time.time() - start_time,
                "product_info":         PRODUCT_INFO,
                "model":                get_model_name(model),
                "conversation_history": conversation_history,
                "buyer_prompt_log":     buyer.prompt_log,
                "seller_prompt_log":    seller.prompt_log,
            })
            break

    env.close()
    print("\nNegotiation completed!")
    _save_results(results, project_root, TASK_NAME, get_model_name(model))


def _save_results(results, project_root, task_name, model_name):
    try:
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_param"
        model_safe = model_name.replace("/", "_").replace(":", "_")
        run_dir = results_dir / model_safe / f"batch_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        exclude_keys = {"conversation_history", "buyer_prompt_log", "seller_prompt_log"}
        summary = {k: v for k, v in results.items() if k not in exclude_keys}
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        history         = results.get("conversation_history", [])
        buyer_prompts   = results.get("buyer_prompt_log", [])
        seller_prompts  = results.get("seller_prompt_log", [])

        buyer_prompt_by_round  = {e["round"]: e["prompt"] for e in buyer_prompts}
        seller_prompt_by_round = {e["round"]: e["prompt"] for e in seller_prompts}
        responses_by_round     = {}
        for msg in history:
            responses_by_round[(msg.get("round", 0), msg.get("role", "unknown"))] = msg.get("content", "")

        all_rounds = sorted(set(
            list(buyer_prompt_by_round) + list(seller_prompt_by_round) +
            [r for r, _ in responses_by_round]
        ))

        SEP = "=" * 80
        sep = "-" * 40

        with open(run_dir / "conversation_log.txt", "w", encoding="utf-8") as f:
            f.write(f"Task: {task_name}\nModel: {model_name}\nStatus: {results.get('status', 'unknown')}\n")
            f.write(SEP + "\n\n")
            for r in all_rounds:
                if r in buyer_prompt_by_round:
                    f.write(f"[Round {r}] BUYER PROMPT:\n{SEP}\n")
                    f.write(buyer_prompt_by_round[r].strip() + f"\n{SEP}\n\n")
                if (r, "buyer") in responses_by_round:
                    f.write(f"[Round {r}] BUYER:\n{responses_by_round[(r, 'buyer')]}\n\n{sep}\n\n")
                if r in seller_prompt_by_round:
                    f.write(f"[Round {r}] SELLER PROMPT:\n{SEP}\n")
                    f.write(seller_prompt_by_round[r].strip() + f"\n{SEP}\n\n")
                if (r, "seller") in responses_by_round:
                    f.write(f"[Round {r}] SELLER:\n{responses_by_round[(r, 'seller')]}\n\n{sep}\n\n")

        print(f"\nResults saved to: {run_dir}")
        if history:
            print(f"  - conversation_log.txt  ({len(history)} messages, with prompts)")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{TASK_NAME}")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--difficulty",
        choices=["normal", "hard", "no_deal"],
        default=os.environ.get("DIFFICULTY", "normal"),
    )
    args = parser.parse_args()
    main(model_name=args.model, difficulty=args.difficulty)
