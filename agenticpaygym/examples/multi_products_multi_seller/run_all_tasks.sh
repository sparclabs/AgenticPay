#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

# ============================================
# Configuration: Model List
# ============================================
# Configure the model list to use here
# If the list is empty, each script will use its default model
# Example:
# MODELS=("gpt-5.2" "gemini-3-pro-all" "claude-sonnet-4-5-20250929")
MODELS=()

# ============================================
# Execute Tasks
# ============================================
if [ ${#MODELS[@]} -eq 0 ]; then
    # Model list is empty: use default behavior (each script uses its own default model)
    echo "Running all tasks (using default models)..."
    python Task1_parallel_two_seller_per_one_product_negotiation.py
    python Task2_parallel_three_seller_per_one_product_negotiation.py
    python Task3_sequential_two_seller_per_one_product_negotiation.py
    python Task4_sequential_three_seller_per_one_product_negotiation.py
else
    # Model list is provided: loop through each model and run all tasks for each model
    echo "Running all tasks with the following model list: ${MODELS[*]}"
    for model in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Using model: $model"
        echo "=========================================="
        echo ""
        
        echo "Running Task1 (model: $model)..."
        python Task1_parallel_two_seller_per_one_product_negotiation.py --model "$model"
        
        echo ""
        echo "Running Task2 (model: $model)..."
        python Task2_parallel_three_seller_per_one_product_negotiation.py --model "$model"
        
        echo ""
        echo "Running Task3 (model: $model)..."
        python Task3_sequential_two_seller_per_one_product_negotiation.py --model "$model"
        
        echo ""
        echo "Running Task4 (model: $model)..."
        python Task4_sequential_three_seller_per_one_product_negotiation.py --model "$model"
        
        echo ""
        echo "Completed all tasks for model: $model"
        echo ""
    done
    echo "All tasks for all models completed!"
fi
