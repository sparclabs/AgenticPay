#!/bin/bash

# ============================================
# Configuration - Set your task and model here
# ============================================
# You can set these variables directly in the script,
# or pass them via command line arguments (--task and --model)
TASK_NAME="Task4"          # Options: Task1, Task2, Task3, Task4
MODEL_NAME="gpt-5.2"  # Model name, or leave empty for default

# ============================================
# Change to script directory
# ============================================
cd "$(dirname "$0")"

# Get project root (4 levels up from script directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_BASE="$PROJECT_ROOT/agenticpaygym/results/only_multi_products"

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
            echo "Usage: $0 [--task <Task1|Task2|Task3|Task4>] [--model <model_name>]"
            echo ""
            echo "You can set TASK_NAME and MODEL_NAME directly in the script,"
            echo "or pass them via command line arguments."
            echo ""
            echo "Options:"
            echo "  --task <Task1|Task2|Task3|Task4>  Override task name (optional)"
            echo "  --model <model_name>              Override model name (optional)"
            echo "  -h, --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Uses TASK_NAME and MODEL_NAME from script"
            echo "  $0 --task Task2                      # Override task, use MODEL_NAME from script"
            echo "  $0 --task Task2 --model gemini-3-pro-all  # Override both"
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
        TASK_SCRIPT="Task1_multi_product_negotiation.py"
        TASK_DISPLAY="Task1"
        ;;
    Task2|task2)
        TASK_SCRIPT="Task2_two_product_negotiation.py"
        TASK_DISPLAY="Task2"
        ;;
    Task3|task3)
        TASK_SCRIPT="Task3_five_product_negotiation.py"
        TASK_DISPLAY="Task3"
        ;;
    Task4|task4)
        TASK_SCRIPT="Task4_select_three_from_five_negotiation.py"
        TASK_DISPLAY="Task4"
        ;;
    *)
        echo "Error: Invalid task name: $TASK_NAME"
        echo "Valid task names: Task1, Task2, Task3, Task4"
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
