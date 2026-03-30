"""Task15: Quantity / Bulk-Discount Negotiation — Multiple Products

Scenario: Procurement manager negotiates two office supply products sequentially:
printer paper (50 reams) and printer toner cartridges (5 units).
Each product has its own volume-discount tiers. Conversation context is preserved.
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
from agenticpay.envs.only_multi_products.Task15_quantity_discount_negotiation import (
    Task15MultiProductQuantityDiscountNegotiation,
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

    # Shared per-unit ZOPA parameters (apply to all products in this scenario)
    buyer_max_unit_price = 8.50
    seller_min_unit_price = 6.0
    buyer_max_unit_price, seller_min_unit_price = adjust_zopa(
        buyer_max_unit_price, seller_min_unit_price, difficulty
    )
    print(f"Difficulty: {difficulty} | buyer_max=${buyer_max_unit_price:.2f} | seller_min=${seller_min_unit_price:.2f}")

    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_unit_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_unit_price)

    env = Task15MultiProductQuantityDiscountNegotiation(
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds_per_product=max_rounds,
        initial_seller_unit_price=10.0,
        buyer_max_unit_price=buyer_max_unit_price,
        buyer_target_quantity=50,
        seller_min_unit_price=seller_min_unit_price,
        price_tolerance=price_tolerance,
        quantity_tolerance=5,
        reward_weights=reward_weights,
    )

    # Define two products with per-product quantities and tiers
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

    user_profile = "Procurement manager negotiating office supplies for annual budget."
    all_results = []
    start_time = time.time()

    for i, product_info in enumerate(products):
        print(f"\n{'='*60}")
        print(f"Product {i+1}/{len(products)}: {product_info['name']}")
        print(f"Target quantity: {product_info['target_quantity']} {product_info['unit']}s")
        print(f"{'='*60}")

        clear_history = (i == 0)
        observation, info = env.reset(
            user_requirement=(
                f"I need {product_info['target_quantity']} {product_info['unit']}s of "
                f"{product_info['name']} at the best per-unit price."
            ),
            product_info=product_info,
            user_profile=user_profile,
            clear_history=clear_history,
            available_products=[p["name"] for p in products],
        )

        done = False
        while not done:
            buyer_action = buyer.respond(
                conversation_history=observation["conversation_history"],
                current_state=observation,
            )
            hist_updated = observation["conversation_history"].copy()
            if buyer_action:
                hist_updated.append({"role": "buyer", "content": buyer_action,
                                     "round": observation.get("current_round", 0)})
            seller_action = seller.respond(
                conversation_history=hist_updated,
                current_state=observation,
            )
            observation, reward, terminated, truncated, info = env.step(
                buyer_action=buyer_action,
                seller_action=seller_action,
            )
            done = terminated or truncated
            env.render()
            sys.stdout.flush()

            if done:
                env._print_global_score_details()
                env._print_buyer_score_details()
                env._print_seller_score_details()

                print(f"\nProduct {i+1} result: {info['status']}")
                aq = info.get("agreed_quantity")
                aup = info.get("agreed_unit_price")
                tdv = info.get("total_deal_value")
                if aq and aup:
                    print(f"  {aq} {product_info['unit']}s @ ${aup:.2f}/{product_info['unit']} = ${tdv:.2f} total")
                for k in ("global_score", "buyer_score", "seller_score"):
                    if k in info: print(f"  {k}: {info[k]:.3f}")
                all_results.append({
                    "product_name": product_info["name"],
                    "status": info.get("status"),
                    "agreed_quantity": aq,
                    "agreed_unit_price": aup,
                    "total_deal_value": tdv,
                    "rounds": info.get("round", 0),
                    "global_score": info.get("global_score"),
                    "buyer_score": info.get("buyer_score"),
                    "seller_score": info.get("seller_score"),
                })

    env.close()
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All products negotiated in {elapsed:.1f}s")
    print(f"{'='*60}")

    final_results = {
        "difficulty": difficulty,
        "task": "Task15_quantity_discount_negotiation",
        "category": "only_multi_products",
        "timestamp": datetime.now().isoformat(),
        "model": model_display,
        "elapsed_time": elapsed,
        "buyer_max_unit_price": buyer_max_unit_price,
        "seller_min_unit_price": seller_min_unit_price,
        "product_results": all_results,
    }

    try:
        results_dir = Path(project_root) / "agenticpay" / "results" / "only_multi_products"
        results_dir.mkdir(parents=True, exist_ok=True)
        model_dir = results_dir / (str(model_display).replace("/", "_").replace(":", "_") or "default")
        model_dir.mkdir(parents=True, exist_ok=True)
        run_dir = model_dir / f"batch_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "summary.json", "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to: {run_dir}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task15: Quantity Discount — Multi-Product")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--difficulty", choices=["normal", "hard", "no_deal"],
                        default=os.environ.get("DIFFICULTY", "normal"))
    args = parser.parse_args()
    main(model_name=args.model, difficulty=args.difficulty)
