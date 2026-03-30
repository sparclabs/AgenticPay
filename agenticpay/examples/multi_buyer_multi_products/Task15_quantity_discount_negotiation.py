"""Task15: Quantity / Bulk-Discount Negotiation — Two Buyers, Multi-Product Bundle

Scenario: Two procurement departments compete to buy a bundle of printer paper
and toner cartridges. Seller picks the buyer offering the highest total value.
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
from agenticpay.envs.multi_buyer_multi_products.Task15_quantity_discount_negotiation import (
    Task15TwoBuyerMultiProductQuantityDiscountNegotiation,
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

    buyer1_max_unit_price = 8.50
    buyer2_max_unit_price = 9.00
    seller_min_unit_price = 6.0

    buyer1_max_unit_price, seller_min_unit_price = adjust_zopa(buyer1_max_unit_price, seller_min_unit_price, difficulty)
    buyer2_max_unit_price, _ = adjust_zopa(buyer2_max_unit_price, seller_min_unit_price, difficulty)
    print(f"Difficulty: {difficulty} | b1_max=${buyer1_max_unit_price:.2f} | b2_max=${buyer2_max_unit_price:.2f} | seller_min=${seller_min_unit_price:.2f}")

    seller_tiers = [(1, 10.0), (10, 9.0), (50, 8.0), (100, 7.0)]

    buyer1 = BuyerAgent(model=model, buyer_max_price=buyer1_max_unit_price)
    buyer2 = BuyerAgent(model=model, buyer_max_price=buyer2_max_unit_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_unit_price)

    products = [
        {
            "name": "Premium Printer Paper",
            "brand": "OfficePro",
            "price": 10.0,
            "features": ["500 sheets", "80 gsm", "Letter size"],
            "condition": "New",
            "unit": "ream",
            "target_quantity": 50,
            "seller_tiers": [(1, 10.0), (10, 9.0), (50, 8.0), (100, 7.0)],
        },
        {
            "name": "Black Toner Cartridge",
            "brand": "OfficePro",
            "price": 40.0,
            "features": ["High yield", "Laser-compatible", "2500 page yield"],
            "condition": "New",
            "unit": "cartridge",
            "target_quantity": 5,
            "seller_tiers": [(1, 40.0), (5, 35.0), (10, 30.0)],
        },
    ]
    total_bundle_qty = sum(p["target_quantity"] for p in products)

    env = Task15TwoBuyerMultiProductQuantityDiscountNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller_agent=seller,
        max_rounds=max_rounds,
        initial_seller_unit_price=10.0,
        buyer1_max_unit_price=buyer1_max_unit_price,
        buyer2_max_unit_price=buyer2_max_unit_price,
        buyer_target_quantity=total_bundle_qty,
        seller_min_unit_price=seller_min_unit_price,
        seller_tiers=seller_tiers,
        price_tolerance=price_tolerance,
        quantity_tolerance=5,
        reward_weights=reward_weights,
    )

    user_requirement = (
        f"I need a bundle of office supplies: {products[0]['target_quantity']} reams of "
        f"{products[0]['name']} and {products[1]['target_quantity']} {products[1]['name']}s. "
        "Please provide the best total price."
    )
    user_profile = "Procurement manager negotiating an annual office supplies bundle."

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=products[0],
        user_profile=user_profile,
    )

    results = {
        "difficulty": difficulty,
        "task": "Task15_quantity_discount_negotiation",
        "category": "multi_buyer_multi_products",
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "success": False,
    }

    done = False
    start_time = time.time()

    while not done:
        b1_hist = observation.get("conversation_history_buyer1", [])
        b2_hist = observation.get("conversation_history_buyer2", [])

        b1_action = buyer1.respond(conversation_history=b1_hist, current_state=observation)
        b2_action = buyer2.respond(conversation_history=b2_hist, current_state=observation)

        rnd = observation.get("current_round", 0)

        def upd(hist, action):
            h = hist.copy()
            if action:
                h.append({"role": "buyer", "content": action, "round": rnd})
            return h

        s_action_b1 = seller.respond(conversation_history=upd(b1_hist, b1_action), current_state=observation)
        s_action_b2 = seller.respond(conversation_history=upd(b2_hist, b2_action), current_state=observation)

        observation, reward, terminated, truncated, info = env.step(
            buyer1_action=b1_action,
            buyer2_action=b2_action,
            seller_action_buyer1=s_action_b1,
            seller_action_buyer2=s_action_b2,
        )
        done = terminated or truncated
        env.render()
        sys.stdout.flush()

        if done:
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()
            print(f"\nStatus: {info['status']}")
            sb = info.get("selected_buyer")
            aq = info.get("agreed_quantity")
            aup = info.get("agreed_unit_price")
            tdv = info.get("total_deal_value")
            if sb and aq:
                print(f"Deal: Buyer{sb} | {aq} units @ ${aup:.2f}/unit = ${tdv:.2f}")
            for k in ("global_score", "buyer_score", "seller_score"):
                if k in info:
                    print(f"{k}: {info[k]:.3f}")
            results.update({
                "status": info.get("status"),
                "success": terminated,
                "selected_buyer": sb,
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
        d = Path(project_root) / "agenticpay" / "results" / "multi_buyer_multi_products"
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
    parser = argparse.ArgumentParser(description="Task15: Quantity Discount — 2B×Multi-Product")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--difficulty", choices=["normal", "hard", "no_deal"],
                        default=os.environ.get("DIFFICULTY", "normal"))
    args = parser.parse_args()
    main(model_name=args.model, difficulty=args.difficulty)
