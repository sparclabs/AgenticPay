#!/bin/bash

# ============================================
# Configuration - Set your task and model here
# ============================================
# You can set these variables directly in the script,
# or pass them via command line arguments (--task and --model)
TASK_NAME="Task3"          # Options: Task1, Task2, Task3, Task4_s1, Task5_s2, Task6_s3, Task7_s4, Task8_s5, Task9_s6, Task10_s7, Task11_s8, Task12_s9, Task13_s10
MODEL_NAME="gpt-5.4"  # Model name, or leave empty for default

# ============================================
# Change to script directory
# ============================================
cd "$(dirname "$0")"

# Get project root (4 levels up from script directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_BASE="$PROJECT_ROOT/agenticpay/results/single_buyer_product_seller"

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
# Parse Arguments (Optional - overrides internal variables)
# ============================================
# Parse command line arguments (if provided, they override the internal variables above)
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--task <Task1|Task2|Task3|Task4_s1|Task5_s2|Task6_s3|Task7_s4|Task8_s5|Task9_s6|Task10_s7|Task11_s8|Task12_s9|Task13_s10>] [--model <model_name>]"
            echo ""
            echo "You can set TASK_NAME and MODEL_NAME directly in the script,"
            echo "or pass them via command line arguments."
            echo ""
            echo "Options:"
            echo "  --task <Task1|Task2|Task3|Task4_s1|Task5_s2|Task6_s3|Task7_s4|Task8_s5|Task9_s6|Task10_s7|Task11_s8|Task12_s9|Task13_s10>   Override task name (optional)"
            echo "  --model <model_name>                   Override model name (optional)"
            echo "  -h, --help                             Show this help message"
            echo ""
            echo "Available Tasks:"
            echo "  Task1    - Basic Price Negotiation (Winter Jacket)"
            echo "  Task2    - Close Price Negotiation"
            echo "  Task3    - Close to Market Price Negotiation"
            echo "  Task4_s1 - Beauty Product Negotiation (Maybelline Eyeshadow)"
            echo "  Task5_s2 - Toothpaste Negotiation (ARM & HAMMER Peroxicare)"
            echo "  Task6_s3 - Riflescope Negotiation (Crimson Trace Brushline Pro)"
            echo "  Task7_s4 - Kids Headphones Negotiation (NVRADCHUA)"
            echo "  Task8_s5 - Wall Lantern Negotiation (Sea Gull Lighting)"
            echo "  Task9_s6 - Bookshelf Negotiation (Kcelarec 4-Tier)"
            echo "  Task10_s7 - Men's Sandals Negotiation (N/C Flip Flops)"
            echo "  Task11_s8 - Women's Jeans Negotiation (myhehthw)"
            echo "  Task12_s9 - Beverage Negotiation (Belvoir Elderflower Rose)"
            echo "  Task13_s10 - Food Color Negotiation (AmeriColor AmeriMist)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Uses TASK_NAME and MODEL_NAME from script"
            echo "  $0 --task Task2                      # Override task, use MODEL_NAME from script"
            echo "  $0 --task Task4_s1 --model gpt-5.2  # Run beauty product scenario"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate task name
if [ -z "$TASK_NAME" ]; then
    echo "Error: TASK_NAME is not set"
    echo "Please set TASK_NAME in the script or use --task argument"
    echo "Use --help for usage information"
    exit 1
fi

# Map task name to script file
case "$TASK_NAME" in
    Task1|task1)
        TASK_SCRIPT="Task1_basic_price_negotiation.py"
        TASK_DISPLAY="Task1"
        ;;
    Task2|task2)
        TASK_SCRIPT="Task2_close_price_negotiation.py"
        TASK_DISPLAY="Task2"
        ;;
    Task3|task3)
        TASK_SCRIPT="Task3_close_to_market_price_negotiation.py"
        TASK_DISPLAY="Task3"
        ;;
    Task4_s1|task4_s1)
        TASK_SCRIPT="Task4_s1_beauty_product_negotiation.py"
        TASK_DISPLAY="Task4_s1"
        ;;
    Task5_s2|task5_s2)
        TASK_SCRIPT="Task5_s2_toothpaste_negotiation.py"
        TASK_DISPLAY="Task5_s2"
        ;;
    Task6_s3|task6_s3)
        TASK_SCRIPT="Task6_s3_riflescope_negotiation.py"
        TASK_DISPLAY="Task6_s3"
        ;;
    Task7_s4|task7_s4)
        TASK_SCRIPT="Task7_s4_headphones_negotiation.py"
        TASK_DISPLAY="Task7_s4"
        ;;
    Task8_s5|task8_s5)
        TASK_SCRIPT="Task8_s5_wall_lantern_negotiation.py"
        TASK_DISPLAY="Task8_s5"
        ;;
    Task9_s6|task9_s6)
        TASK_SCRIPT="Task9_s6_bookshelf_negotiation.py"
        TASK_DISPLAY="Task9_s6"
        ;;
    Task10_s7|task10_s7)
        TASK_SCRIPT="Task10_s7_sandals_negotiation.py"
        TASK_DISPLAY="Task10_s7"
        ;;
    Task11_s8|task11_s8)
        TASK_SCRIPT="Task11_s8_jeans_negotiation.py"
        TASK_DISPLAY="Task11_s8"
        ;;
    Task12_s9|task12_s9)
        TASK_SCRIPT="Task12_s9_beverage_negotiation.py"
        TASK_DISPLAY="Task12_s9"
        ;;
    Task13_s10|task13_s10)
        TASK_SCRIPT="Task13_s10_food_color_negotiation.py"
        TASK_DISPLAY="Task13_s10"
        ;;
    *)
        echo "Error: Invalid task name: $TASK_NAME"
        echo "Valid task names: Task1, Task2, Task3, Task4_s1, Task5_s2, Task6_s3, Task7_s4, Task8_s5, Task9_s6, Task10_s7, Task11_s8, Task12_s9, Task13_s10"
        exit 1
        ;;
esac

# Check if script file exists
if [ ! -f "$TASK_SCRIPT" ]; then
    echo "Error: Task script not found: $TASK_SCRIPT"
    exit 1
fi

# ============================================
# Execute Task
# ============================================
echo "=========================================="
echo "Running $TASK_DISPLAY"
if [ -n "$MODEL_NAME" ]; then
    echo "Model: $MODEL_NAME"
else
    echo "Model: (using default)"
fi
echo "=========================================="
echo ""

# Create temporary log file
TEMP_LOG=$(mktemp)

# Run the task
if [ -n "$MODEL_NAME" ]; then
    # Run with specified model
    python "$TASK_SCRIPT" --model "$MODEL_NAME" 2>&1 | tee "$TEMP_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    
    # Save run history if model is specified
    save_run_history "$TEMP_LOG" "$MODEL_NAME" "$TASK_DISPLAY"
else
    # Run with default model
    python "$TASK_SCRIPT" 2>&1 | tee "$TEMP_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    
    # Try to extract model name from output or use a default
    # Note: This is a fallback, ideally we'd know the model name
    if [ -n "$MODEL_NAME" ]; then
        save_run_history "$TEMP_LOG" "$MODEL_NAME" "$TASK_DISPLAY"
    fi
fi

# Clean up temporary log file
rm -f "$TEMP_LOG"

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ $TASK_DISPLAY completed successfully!"
else
    echo "✗ $TASK_DISPLAY failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
