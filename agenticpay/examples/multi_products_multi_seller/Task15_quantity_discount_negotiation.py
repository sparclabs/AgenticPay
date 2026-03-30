"""Task15: Quantity / Bulk-Discount Negotiation — Multi-Product, Two Sellers

Scenario: One buyer negotiates in parallel with two suppliers, each selling a
different product. Seller 1 offers printer paper; Seller 2 offers toner cartridges.
The buyer picks the deal with the best combined surplus.
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
from agenticpay.envs.multi_products_multi_seller.Task15_quantity_discount_negotiation import (
    Task15MultiProductTwoSellerQuantityDiscountNegotiation,
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

    # Seller 1: printer paper
    buyer_max_s1 = 8.50
    seller1_min = 6.0
    buyer_max_s1, seller1_min = adjust_zopa(buyer_max_s1, seller1_min, difficulty)

    # Seller 2: toner cartridges
    buyer_max_s2 = 38.0
    seller2_min = 25.0
    buyer_max_s2, seller2_min = adjust_zopa(buyer_max_s2, seller2_min, difficulty)

    print(f"Difficulty: {difficulty}")
    print(f"  Paper:  buyer_max=${buyer_max_s1:.2f} | seller_min=${seller1_min:.2f}")
    print(f"  Toner:  buyer_max=${buyer_max_s2:.2f} | seller_min=${seller2_min:.2f}")

    seller1_tiers = [(1, 10.0), (10, 9.0), (50, 8.0), (100, 7.0)]
    seller2_tiers = [(1, 40.0), (5, 35.0), (10, 30.0)]

    seller1_product_info = {
        "name": "Premium Printer Paper",
        "brand": "OfficePro",
        "price": 10.0,
        "features": ["500 sheets", "80 gsm", "Letter size"],
        "condition": "New",
        "unit": "ream",
    }
    seller2_product_info = {
        "name": "Black Toner Cartridge",
        "brand": "OfficePro",
        "price": 40.0,
        "features": ["High yield", "Laser-compatible", "2500 page yield"],
        "condition": "New",
        "unit": "cartridge",
    }

    buyer = BuyerAgent(model=model, buyer_max_price=max(buyer_max_s1, buyer_max_s2))
    seller1 = SellerAgent(model=model, seller_min_price=seller1_min)
    seller2 = SellerAgent(model=model, seller_min_price=seller2_min)

    env = Task15MultiProductTwoSellerQuantityDiscountNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_unit_price=10.0,
        initial_seller2_unit_price=40.0,
        buyer_max_unit_price_s1=buyer_max_s1,
        buyer_max_unit_price_s2=buyer_max_s2,
        buyer_target_quantity_s1=50,
        buyer_target_quantity_s2=5,
        seller1_min_unit_price=seller1_min,
        seller2_min_unit_price=seller2_min,
        seller1_tiers=seller1_tiers,
        seller2_tiers=seller2_tiers,
        price_tolerance=price_tolerance,
        quantity_tolerance=5,
        reward_weights=reward_weights,
    )

    user_requirement = (
        "I need 50 reams of printer paper from one supplier and "
        "5 toner cartridges from another. Negotiate the best per-unit price for each."
    )
    user_profile = "Procurement manager sourcing office supplies from specialized suppliers."

    observation, info = env.reset(
        user_requirement=user_requirement,
        seller1_product_info=seller1_product_info,
        seller2_product_info=seller2_product_info,
        user_profile=user_profile,
    )

    results = {
        "difficulty": difficulty,
        "task": "Task15_quantity_discount_negotiation",
        "category": "multi_products_multi_seller",
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "success": False,
    }

    done = False
    start_time = time.time()

    while not done:
        s1_hist = observation.get("conversation_history_seller1", [])
        s2_hist = observation.get("conversation_history_seller2", [])

        b_action_s1 = buyer.respond(conversation_history=s1_hist, current_state=observation)
        b_action_s2 = buyer.respond(conversation_history=s2_hist, current_state=observation)

        rnd = observation.get("current_round", 0)

        def upd(hist, action):
            h = hist.copy()
            if action:
                h.append({"role": "buyer", "content": action, "round": rnd})
            return h

        s1_action = seller1.respond(conversation_history=upd(s1_hist, b_action_s1), current_state=observation)
        s2_action = seller2.respond(conversation_history=upd(s2_hist, b_action_s2), current_state=observation)

        observation, reward, terminated, truncated, info = env.step(
            buyer_action_seller1=b_action_s1,
            buyer_action_seller2=b_action_s2,
            seller1_action=s1_action,
            seller2_action=s2_action,
        )
        done = terminated or truncated
        env.render()
        sys.stdout.flush()

        if done:
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()
            print(f"\nStatus: {info['status']}")
            ss = info.get("selected_seller")
            aq = info.get("agreed_quantity")
            aup = info.get("agreed_unit_price")
            tdv = info.get("total_deal_value")
            prod = info.get("selected_product", "")
            if ss and aq:
                print(f"Deal: Seller{ss} ({prod}) | {aq} units @ ${aup:.2f}/unit = ${tdv:.2f}")
            for k in ("global_score", "buyer_score", "seller_score"):
                if k in info:
                    print(f"{k}: {info[k]:.3f}")
            results.update({
                "status": info.get("status"),
                "success": terminated,
                "selected_seller": ss,
                "selected_product": prod,
                "agreed_quantity": aq,
                "agreed_unit_price": aup,
                "total_deal_value": tdv,
                "total_rounds": info.get("round", 0),
                "total_reward": float(reward) if reward else None,
                "global_score": info.get("global_score"),
                "buyer_score": info.get("buyer_score"),
                "seller_score": info.get("seller_score"),
                "elapsed_time": time.time() - start_time,
                "model": model_display,
            })
            break

    env.close()
    try:
        d = Path(project_root) / "agenticpay" / "results" / "multi_products_multi_seller"
        d.mkdir(parents=True, exist_ok=True)
        md = d / (str(model_display).replace("/", "_").replace(":", "_") or "default")
        md.mkdir(parents=True, exist_ok=True)
        rd = md / f"batch_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        rd.mkdir(parents=True, exist_ok=True)
        with open(rd / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {rd}")
    except Exception as e:
        print(f"\nWarning: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task15: Quantity Discount — Multi-Product×2Sellers")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--difficulty", choices=["normal", "hard", "no_deal"],
                        default=os.environ.get("DIFFICULTY", "normal"))
    args = parser.parse_args()
    main(model_name=args.model, difficulty=args.difficulty)
