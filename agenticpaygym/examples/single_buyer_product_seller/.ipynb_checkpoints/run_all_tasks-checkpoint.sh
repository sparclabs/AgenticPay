#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

# Get project root (4 levels up from script directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_BASE="$PROJECT_ROOT/agenticpaygym/results/single_buyer_product_seller"

# Function to find the latest result directory for a model
find_latest_result_dir() {
    local model_name="$1"
    local model_name_safe=$(echo "$model_name" | sed 's/[\/\\:]/_/g')
    local model_dir="$RESULTS_BASE/$model_name_safe"
    
    if [ -d "$model_dir" ]; then
        # Find the most recently created batch_evaluation_* directory
        # Use ls -t to sort by modification time (newest first)
        ls -td "$model_dir"/batch_evaluation_* 2>/dev/null | head -1
    fi
}

# Function to save run history to the latest result directory
save_run_history() {
    local log_file="$1"
    local model_name="$2"
    local task_name="$3"
    
    if [ ! -f "$log_file" ]; then
        return
    fi
    
    # Find the latest result directory
    local latest_dir=$(find_latest_result_dir "$model_name")
    
    if [ -n "$latest_dir" ] && [ -d "$latest_dir" ]; then
        # Save run history with task name prefix
        local history_file="$latest_dir/${task_name}_run_history.txt"
        cp "$log_file" "$history_file"
        echo "  Run history saved to: $history_file"
    fi
}

# ============================================
# Configuration: Model List
# ============================================
# Configure the model list to use here
# If the list is empty, each script will use its default model
# Example:
# MODELS=("gpt-5.2" "gemini-3-pro-all" "claude-sonnet-4-5-20250929")
MODELS=("gemini-3-pro-all" "qwen3-8b")

# ============================================
# Execute Tasks
# ============================================
if [ ${#MODELS[@]} -eq 0 ]; then
    # Model list is empty: use default behavior (each script uses its own default model)
    echo "Running all tasks (using default models)..."
    
    # Create temporary log file
    TEMP_LOG=$(mktemp)
    
    # Run each task with tee to capture output
    echo "Running Task1..."
    python Task1_basic_price_negotiation.py 2>&1 | tee "$TEMP_LOG"
    # Note: For default model case, we can't determine the exact result directory easily
    # So we'll save to a general location or skip this feature for default models
    
    echo ""
    echo "Running Task2..."
    python Task2_close_price_negotiation.py 2>&1 | tee "$TEMP_LOG"
    
    echo ""
    echo "Running Task3..."
    python Task3_close_to_market_price_negotiation.py 2>&1 | tee "$TEMP_LOG"
    
    # Clean up
    rm -f "$TEMP_LOG"
else
    # Model list is provided: loop through each model and run all tasks for each model
    echo "Running all tasks with the following model list: ${MODELS[*]}"
    for model in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Using model: $model"
        echo "=========================================="
        echo ""
        
        # Create temporary log file for this model's tasks
        TEMP_LOG=$(mktemp)
        
        echo "Running Task1 (model: $model)..."
        python Task1_basic_price_negotiation.py --model "$model" 2>&1 | tee "$TEMP_LOG"
        save_run_history "$TEMP_LOG" "$model" "Task1"
        
        echo ""
        echo "Running Task2 (model: $model)..."
        python Task2_close_price_negotiation.py --model "$model" 2>&1 | tee "$TEMP_LOG"
        save_run_history "$TEMP_LOG" "$model" "Task2"
        
        echo ""
        echo "Running Task3 (model: $model)..."
        python Task3_close_to_market_price_negotiation.py --model "$model" 2>&1 | tee "$TEMP_LOG"
        save_run_history "$TEMP_LOG" "$model" "Task3"
        
        # Clean up temporary log file
        rm -f "$TEMP_LOG"
        
        echo ""
        echo "Completed all tasks for model: $model"
        echo ""
    done
    echo "All tasks for all models completed!"
fi
