"""Task15: Quantity / Bulk-Discount Negotiation — One Buyer, Two Sellers

Scenario: A procurement manager negotiates printer paper orders with two competing
suppliers. Each supplier has private tiered pricing. Buyer picks the better deal.
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
from agenticpay.envs.only_multi_seller.Task15_quantity_discount_negotiation import (
    Task15ParallelTwoSellerQuantityDiscountNegotiation,
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

    seller1_tiers = [(1, 10.0), (10, 9.0), (50, 8.0), (100, 7.0)]
    seller2_tiers = [(1, 10.5), (10, 9.2), (50, 7.8), (100, 6.8)]   # slightly different tiers
    initial_seller1_unit_price = 10.0
    initial_seller2_unit_price = 10.5
    seller1_min_unit_price = 6.0
    seller2_min_unit_price = 5.8   # seller2 has lower floor
    buyer_target_quantity = 50
    buyer_max_unit_price = 8.50

    buyer_max_unit_price, seller1_min_unit_price = adjust_zopa(
        buyer_max_unit_price, seller1_min_unit_price, difficulty
    )
    _, seller2_min_unit_price = adjust_zopa(buyer_max_unit_price, seller2_min_unit_price, difficulty)
    print(
        f"Difficulty: {difficulty} | buyer_max=${buyer_max_unit_price:.2f} | "
        f"seller1_min=${seller1_min_unit_price:.2f} | seller2_min=${seller2_min_unit_price:.2f}"
    )

    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_unit_price)
    seller1 = SellerAgent(model=model, seller_min_price=seller1_min_unit_price)
    seller2 = SellerAgent(model=model, seller_min_price=seller2_min_unit_price)

    env = Task15ParallelTwoSellerQuantityDiscountNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_unit_price=initial_seller1_unit_price,
        initial_seller2_unit_price=initial_seller2_unit_price,
        buyer_max_unit_price=buyer_max_unit_price,
        buyer_target_quantity=buyer_target_quantity,
        seller1_min_unit_price=seller1_min_unit_price,
        seller2_min_unit_price=seller2_min_unit_price,
        seller1_tiers=seller1_tiers,
        seller2_tiers=seller2_tiers,
        price_tolerance=price_tolerance,
        quantity_tolerance=5,
        reward_weights=reward_weights,
    )

    user_requirement = (
        f"I need approximately {buyer_target_quantity} reams of printer paper. "
        "Comparing two suppliers for the best per-unit price at volume."
    )
    user_profile = "Procurement manager seeking the lowest per-unit cost for bulk orders."

    print("\n" + "=" * 60)
    print("Starting two-seller quantity/bulk-discount negotiation...")
    print("=" * 60)

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )

    results = {
        "difficulty": difficulty,
        "task": "Task15_quantity_discount_negotiation",
        "category": "only_multi_seller",
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "success": False,
        "buyer_target_quantity": buyer_target_quantity,
        "seller1_tiers": seller1_tiers,
        "seller2_tiers": seller2_tiers,
        "buyer_max_unit_price": buyer_max_unit_price,
        "seller1_min_unit_price": seller1_min_unit_price,
        "seller2_min_unit_price": seller2_min_unit_price,
    }

    done = False
    start_time = time.time()

    while not done:
        hist_s1 = observation.get("conversation_history_seller1", [])
        hist_s2 = observation.get("conversation_history_seller2", [])

        buyer_action_s1 = buyer.respond(conversation_history=hist_s1, current_state=observation)
        buyer_action_s2 = buyer.respond(conversation_history=hist_s2, current_state=observation)

        hist_s1_updated = hist_s1.copy()
        hist_s2_updated = hist_s2.copy()
        rnd = observation.get("current_round", 0)
        if buyer_action_s1: hist_s1_updated.append({"role": "buyer", "content": buyer_action_s1, "round": rnd})
        if buyer_action_s2: hist_s2_updated.append({"role": "buyer", "content": buyer_action_s2, "round": rnd})

        seller1_action = seller1.respond(conversation_history=hist_s1_updated, current_state=observation)
        seller2_action = seller2.respond(conversation_history=hist_s2_updated, current_state=observation)

        observation, reward, terminated, truncated, info = env.step(
            buyer_action_seller1=buyer_action_s1,
            buyer_action_seller2=buyer_action_s2,
            seller1_action=seller1_action,
            seller2_action=seller2_action,
        )
        done = terminated or truncated
        env.render()
        sys.stdout.flush()

        if done:
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()

            print("\n" + "=" * 60)
            print(f"Status: {info['status']}")
            sel = info.get("selected_seller")
            if sel:
                aq = info.get("agreed_quantity")
                aup = info.get("agreed_unit_price")
                tdv = info.get("total_deal_value")
                if aq and aup:
                    print(f"Selected Seller: {sel} | {aq} reams @ ${aup:.2f}/ream = ${tdv:.2f} total")
            else:
                print("No deal reached.")
            print(f"Rounds: {info['round']} | Reward: {reward:.3f}")
            for k in ("global_score", "buyer_score", "seller_score"):
                if k in info: print(f"{k}: {info[k]:.3f}")
            print("=" * 60)

            elapsed = time.time() - start_time
            results.update({
                "status": info.get("status"), "success": terminated,
                "selected_seller": info.get("selected_seller"),
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
        results_dir = Path(project_root) / "agenticpay" / "results" / "only_multi_seller"
        results_dir.mkdir(parents=True, exist_ok=True)
        model_dir = results_dir / (str(model_display).replace("/", "_").replace(":", "_") or "default")
        model_dir.mkdir(parents=True, exist_ok=True)
        run_dir = model_dir / f"batch_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {run_dir}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task15: Quantity Discount — Two Sellers")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--difficulty", choices=["normal", "hard", "no_deal"],
                        default=os.environ.get("DIFFICULTY", "normal"))
    args = parser.parse_args()
    main(model_name=args.model, difficulty=args.difficulty)
