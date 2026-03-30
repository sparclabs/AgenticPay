"""Task15: Quantity / Bulk-Discount Negotiation Example

Scenario: Office supply procurement (printer paper reams).

Buyer needs to purchase printer paper and wants a volume discount.
Seller has private tiered pricing that drops at order thresholds.

Both agents negotiate on both quantity (units) AND per-unit price.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path (script lives 4 levels deep from root)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.envs.single_buyer_product_seller.Task15_quantity_discount_negotiation import (
    Task15QuantityDiscountNegotiation,
)

from agenticpay.examples.config import (
    reward_weights,
    max_rounds,
    price_tolerance,
    get_model,
    get_model_name,
    adjust_zopa,
    DIFFICULTY,
)


def main(model_name=None, difficulty=DIFFICULTY):
    """Run a single quantity/bulk-discount negotiation episode.

    Args:
        model_name: Unused (kept for API parity with other Task runners).
                    Model is selected via MODEL_MODE / config.py.
        difficulty: ZOPA difficulty level: 'normal', 'hard', or 'no_deal'.
    """
    print("Initialising model...")
    model = get_model()
    model_display = get_model_name(model)
    print(f"✓ Model: {model_display}")

    # ------------------------------------------------------------------
    # Scenario parameters
    # ------------------------------------------------------------------
    # Printer paper (500-sheet reams)
    product_info = {
        "name": "Premium Printer Paper",
        "brand": "OfficePro",
        "price": 10.0,          # listed retail price per ream
        "features": ["500 sheets", "80 gsm", "Letter size", "Acid-free"],
        "condition": "New",
        "unit": "ream (500 sheets)",
    }

    # Seller's private tiered pricing (unit price drops with quantity)
    seller_tiers = [
        (1,   10.0),    #   1–9 reams:  $10.00/ream (listed price)
        (10,   9.0),    #  10–49 reams:  $9.00/ream
        (50,   8.0),    #  50–99 reams:  $8.00/ream
        (100,  7.0),    # 100+  reams:  $7.00/ream
    ]
    initial_seller_unit_price = 10.0     # publicly listed price
    seller_min_unit_price = 6.0          # private absolute floor (production cost)

    buyer_target_quantity = 50           # buyer needs 50 reams (public)
    buyer_max_unit_price = 8.50          # private max per-unit budget

    # Apply ZOPA difficulty (operates on the per-unit price ZOPA)
    buyer_max_unit_price, seller_min_unit_price = adjust_zopa(
        buyer_max_unit_price, seller_min_unit_price, difficulty
    )
    print(
        f"Difficulty: {difficulty} | "
        f"buyer_max=${buyer_max_unit_price:.2f}/ream | "
        f"seller_min=${seller_min_unit_price:.2f}/ream"
    )

    # ------------------------------------------------------------------
    # Create agents
    # ------------------------------------------------------------------
    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_unit_price)
    seller = SellerAgent(model=model, seller_min_price=seller_min_unit_price)

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    env = Task15QuantityDiscountNegotiation(
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=max_rounds,
        initial_seller_unit_price=initial_seller_unit_price,
        buyer_max_unit_price=buyer_max_unit_price,
        buyer_target_quantity=buyer_target_quantity,
        seller_min_unit_price=seller_min_unit_price,
        seller_tiers=seller_tiers,
        price_tolerance=price_tolerance,
        quantity_tolerance=5,           # allow ±5 reams tolerance
        reward_weights=reward_weights,
    )

    user_requirement = (
        f"I need to purchase approximately {buyer_target_quantity} reams of printer paper "
        "for our office. I'm looking for the best price per ream given our order volume."
    )
    user_profile = (
        "Procurement manager who compares supplier quotes. "
        "Willing to adjust order size if a better price tier is available."
    )

    print("\n" + "=" * 60)
    print("Starting quantity/bulk-discount negotiation...")
    print("=" * 60)

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )

    # ------------------------------------------------------------------
    # Negotiation loop
    # ------------------------------------------------------------------
    done = False
    start_time = time.time()

    results = {
        "difficulty": difficulty,
        "task": "Task15_quantity_discount_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
        # Quantity-discount specific
        "buyer_target_quantity": buyer_target_quantity,
        "seller_tiers": seller_tiers,
        "buyer_max_unit_price": buyer_max_unit_price,
        "seller_min_unit_price": seller_min_unit_price,
        "initial_seller_unit_price": initial_seller_unit_price,
    }

    while not done:
        # Buyer responds
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation,
        )

        # Give seller visibility of buyer's latest message
        updated_history = observation["conversation_history"].copy()
        if buyer_action:
            updated_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": observation.get("current_round", 0),
            })

        # Seller responds
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

        # Step reward display
        if "step_seller_reward" in info or "step_buyer_reward" in info:
            print(f"\n[Step Rewards] ", end="")
            if "step_seller_reward" in info:
                print(f"Seller: {info['step_seller_reward']:.3f}", end="")
            if "step_buyer_reward" in info:
                if "step_seller_reward" in info:
                    print(" | ", end="")
                print(f"Buyer: {info['step_buyer_reward']:.3f}", end="")
            print()

        if done:
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()

            print("\n" + "=" * 60)
            print("Negotiation Ended")
            print("=" * 60)
            print(f"Status: {info['status']}")

            seller_price = info.get("seller_price")
            buyer_price = info.get("buyer_price")
            # seller_price / buyer_price are total prices; show derived unit prices when available
            aup = info.get("agreed_unit_price")
            sq = info.get("seller_quantity") or env.buyer_target_quantity
            bq = info.get("buyer_quantity") or env.buyer_target_quantity
            s_unit = (seller_price / sq) if seller_price and sq else None
            b_unit = (buyer_price / bq) if buyer_price and bq else None
            print(
                f"Final Unit Prices: "
                f"Seller=${s_unit:.2f}/unit (total ${seller_price:.2f})" if s_unit else "Seller=N/A"
            )
            print(
                f"                   "
                f"Buyer=${b_unit:.2f}/unit (total ${buyer_price:.2f})" if b_unit else "Buyer=N/A"
            )

            aq = info.get("agreed_quantity")
            aup = info.get("agreed_unit_price")
            tdv = info.get("total_deal_value")
            if aq is not None and aup is not None:
                print(f"Agreed: {aq} reams @ ${aup:.2f}/ream = ${tdv:.2f} total")
            else:
                print("No deal reached.")

            print(f"Total Rounds: {info['round']}")
            print(f"Total Reward: {reward:.3f}")
            if "global_score" in info:
                print(f"GlobalScore: {info['global_score']:.3f}")
            if "buyer_score" in info:
                print(f"BuyerScore:  {info['buyer_score']:.3f}")
            if "seller_score" in info:
                print(f"SellerScore: {info['seller_score']:.3f}")
            print("=" * 60)

            elapsed_time = time.time() - start_time
            results.update({
                "status": info.get("status", "unknown"),
                "success": terminated,
                "seller_price": seller_price,
                "buyer_price": buyer_price,
                "agreed_price": info.get("agreed_price"),
                "agreed_unit_price": aup,
                "agreed_quantity": aq,
                "total_deal_value": tdv,
                "buyer_quantity": info.get("buyer_quantity"),
                "seller_quantity": info.get("seller_quantity"),
                "total_rounds": info.get("round", 0),
                "total_reward": float(reward) if reward is not None else None,
                "seller_reward": info.get("seller_reward"),
                "buyer_reward": info.get("buyer_reward"),
                "global_score": info.get("global_score"),
                "buyer_score": info.get("buyer_score"),
                "seller_score": info.get("seller_score"),
                "termination_reason": info.get("termination_reason"),
                "elapsed_time": elapsed_time,
                "product_info": product_info,
                "model": model_display,
            })
            break

    env.close()
    print("\nNegotiation completed!")

    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    try:
        results_dir = (
            Path(project_root)
            / "agenticpay"
            / "results"
            / "single_buyer_product_seller"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        model_name_safe = (
            (str(model_display).strip() or "default_model")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        model_dir = results_dir / model_name_safe
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"batch_evaluation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_file = run_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        output_file = run_dir / "Task15_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Task15: Quantity / Bulk-Discount Negotiation Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"Difficulty: {results['difficulty']}\n\n")
            f.write(f"Status: {results['status']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Total Rounds: {results['total_rounds']}\n")
            f.write(f"Elapsed Time: {results.get('elapsed_time', 0):.2f}s\n\n")
            f.write("Quantity / Price:\n")
            f.write(f"  Target Quantity: {results['buyer_target_quantity']} reams\n")
            f.write(f"  Agreed Quantity: {results.get('agreed_quantity')}\n")
            f.write(f"  Agreed Unit Price: ${results.get('agreed_unit_price')}\n")
            f.write(f"  Total Deal Value: ${results.get('total_deal_value')}\n\n")
            f.write("Scores:\n")
            for key in ("global_score", "buyer_score", "seller_score"):
                val = results.get(key)
                if val is not None:
                    f.write(f"  {key}: {val:.3f}\n")

        print(f"\nResults saved to: {run_dir}")
        print(f"  - Summary JSON: {summary_file}")
        print(f"  - Output Text:  {output_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task15: Quantity / Bulk-Discount Negotiation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Unused; model is selected via MODEL_MODE in config.py.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["normal", "hard", "no_deal"],
        default=os.environ.get("DIFFICULTY", "normal"),
        help=(
            "ZOPA difficulty: normal (default), hard (tight ZOPA ~5%% spread), "
            "no_deal (buyer max < seller min — no rational agreement possible)."
        ),
    )
    args = parser.parse_args()
    main(model_name=args.model, difficulty=args.difficulty)
