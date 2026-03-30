"""Task15: Quantity / Bulk-Discount Negotiation — Two Buyers, One Seller

Scenario: Two procurement managers compete to buy printer paper from the same supplier.
The supplier has private tiered pricing. Both buyers negotiate on quantity AND per-unit price.
Seller picks the buyer offering the higher unit price.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.envs.only_multi_buyer.Task15_quantity_discount_negotiation import (
    Task15ParallelTwoBuyerQuantityDiscountNegotiation,
)
from agenticpay.examples.config import (
    reward_weights, max_rounds, price_tolerance,
    get_model, get_model_name, adjust_zopa, DIFFICULTY,
)


def main(model_name=None, difficulty=DIFFICULTY):
    print("Initialising model...")
    model = get_model()
    model_display = get_model_name(model)
    print(f"✓ Model: {model_display}")

    product_info = {
        "name": "Premium Printer Paper",
        "brand": "OfficePro",
        "price": 10.0,
        "features": ["500 sheets", "80 gsm", "Letter size", "Acid-free"],
        "condition": "New",
        "unit": "ream (500 sheets)",
    }

    seller_tiers = [
        (1,   10.0),
        (10,   9.0),
        (50,   8.0),
        (100,  7.0),
    ]
    initial_seller_unit_price = 10.0
    seller_min_unit_price = 6.0
    buyer_target_quantity = 50
    buyer1_max_unit_price = 8.50
    buyer2_max_unit_price = 9.00   # buyer2 has a slightly higher budget

    buyer1_max_unit_price, seller_min_unit_price = adjust_zopa(
        buyer1_max_unit_price, seller_min_unit_price, difficulty
    )
    buyer2_max_unit_price, _ = adjust_zopa(buyer2_max_unit_price, seller_min_unit_price, difficulty)
    print(
        f"Difficulty: {difficulty} | "
        f"buyer1_max=${buyer1_max_unit_price:.2f} | "
        f"buyer2_max=${buyer2_max_unit_price:.2f} | "
        f"seller_min=${seller_min_unit_price:.2f}"
    )

    buyer1 = BuyerAgent(model=model, buyer_max_price=buyer1_max_unit_price)
    buyer2 = BuyerAgent(model=model, buyer_max_price=buyer2_max_unit_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_unit_price)

    env = Task15ParallelTwoBuyerQuantityDiscountNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller_agent=seller,
        max_rounds=max_rounds,
        initial_seller_unit_price=initial_seller_unit_price,
        buyer1_max_unit_price=buyer1_max_unit_price,
        buyer2_max_unit_price=buyer2_max_unit_price,
        buyer_target_quantity=buyer_target_quantity,
        seller_min_unit_price=seller_min_unit_price,
        seller_tiers=seller_tiers,
        price_tolerance=price_tolerance,
        quantity_tolerance=5,
        reward_weights=reward_weights,
    )

    user_requirement = (
        f"I need to purchase approximately {buyer_target_quantity} reams of printer paper "
        "for our office. Looking for the best price per ream given our order volume."
    )
    user_profile = (
        "Procurement manager who compares supplier quotes. "
        "Willing to adjust order size for a better price tier."
    )

    print("\n" + "=" * 60)
    print("Starting two-buyer quantity/bulk-discount negotiation...")
    print("=" * 60)

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )

    results = {
        "difficulty": difficulty,
        "task": "Task15_quantity_discount_negotiation",
        "category": "only_multi_buyer",
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "success": False,
        "error": None,
        "buyer_target_quantity": buyer_target_quantity,
        "seller_tiers": seller_tiers,
        "buyer1_max_unit_price": buyer1_max_unit_price,
        "buyer2_max_unit_price": buyer2_max_unit_price,
        "seller_min_unit_price": seller_min_unit_price,
        "initial_seller_unit_price": initial_seller_unit_price,
    }

    done = False
    start_time = time.time()

    while not done:
        buyer1_action = buyer1.respond(
            conversation_history=observation.get("conversation_history_buyer1", []),
            current_state=observation,
        )
        buyer2_action = buyer2.respond(
            conversation_history=observation.get("conversation_history_buyer2", []),
            current_state=observation,
        )

        hist1 = observation.get("conversation_history_buyer1", []).copy()
        hist2 = observation.get("conversation_history_buyer2", []).copy()
        if buyer1_action:
            hist1.append({"role": "buyer", "content": buyer1_action, "round": observation.get("current_round", 0)})
        if buyer2_action:
            hist2.append({"role": "buyer", "content": buyer2_action, "round": observation.get("current_round", 0)})

        seller_action_b1 = seller.respond(
            conversation_history=hist1,
            current_state=observation,
        )
        seller_action_b2 = seller.respond(
            conversation_history=hist2,
            current_state=observation,
        )

        observation, reward, terminated, truncated, info = env.step(
            buyer1_action=buyer1_action,
            buyer2_action=buyer2_action,
            seller_action_buyer1=seller_action_b1,
            seller_action_buyer2=seller_action_b2,
        )
        done = terminated or truncated
        env.render()
        sys.stdout.flush()

        if done:
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()

            print("\n" + "=" * 60)
            print("Negotiation Ended")
            print("=" * 60)
            print(f"Status: {info['status']}")
            sel = info.get("selected_buyer")
            if sel:
                aq = info.get("agreed_quantity")
                aup = info.get("agreed_unit_price")
                tdv = info.get("total_deal_value")
                print(f"Selected Buyer: {sel}")
                if aq and aup:
                    print(f"Agreed: {aq} reams @ ${aup:.2f}/ream = ${tdv:.2f} total")
            else:
                print("No deal reached.")
            print(f"Total Rounds: {info['round']}")
            print(f"Total Reward: {reward:.3f}")
            for key in ("global_score", "buyer_score", "seller_score"):
                if key in info:
                    print(f"{key}: {info[key]:.3f}")
            print("=" * 60)

            elapsed = time.time() - start_time
            results.update({
                "status": info.get("status", "unknown"),
                "success": terminated,
                "selected_buyer": info.get("selected_buyer"),
                "agreed_quantity": info.get("agreed_quantity"),
                "agreed_unit_price": info.get("agreed_unit_price"),
                "total_deal_value": info.get("total_deal_value"),
                "total_rounds": info.get("round", 0),
                "total_reward": float(reward) if reward is not None else None,
                "global_score": info.get("global_score"),
                "buyer_score": info.get("buyer_score"),
                "seller_score": info.get("seller_score"),
                "elapsed_time": elapsed,
                "product_info": product_info,
                "model": model_display,
            })
            break

    env.close()
    print("\nNegotiation completed!")

    try:
        results_dir = (
            Path(project_root) / "agenticpay" / "results" / "only_multi_buyer"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        model_dir = results_dir / (str(model_display).replace("/", "_").replace(":", "_") or "default_model")
        model_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"batch_evaluation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_file = run_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {run_dir}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task15: Quantity Discount — Two Buyers")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--difficulty", choices=["normal", "hard", "no_deal"],
        default=os.environ.get("DIFFICULTY", "normal"),
    )
    args = parser.parse_args()
    main(model_name=args.model, difficulty=args.difficulty)
