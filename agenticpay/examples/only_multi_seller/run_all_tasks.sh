#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

# Get project root (4 levels up from script directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_BASE="$PROJECT_ROOT/agenticpay/results/only_multi_seller"

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


# Function to check if task results already exist for a given model
task_result_exists() {
    local model_name="$1"
    local script_name="$2"
    local model_name_safe=$(echo "$model_name" | sed 's/[\/\\:]/_/g')
    local model_dir="$RESULTS_BASE/$model_name_safe"

    if [ ! -d "$model_dir" ]; then
        return 1
    fi

    for summary in "$model_dir"/batch_evaluation_*/summary.json; do
        if [ -f "$summary" ]; then
            local task_in_file
            task_in_file=$(python3 -c "import json; d=json.load(open('$summary')); print(d.get('task',''))" 2>/dev/null)
            if [ "$task_in_file" = "$script_name" ]; then
                return 0
            fi
        fi
    done
    return 1
}
# ============================================
# Configuration: Model List
# ============================================
# Configure the model list to use here
# If the list is empty, each script will use its default model
# Example:
# MODELS=("gpt-5.2" "gemini-3-pro-all" "claude-sonnet-4-5-20250929")
MODELS=("google/gemma-4-31B-it")


# ============================================
# Configuration: Task List
# ============================================
# Configure which tasks to run
# If the list is empty, all available tasks will be run
# Format: Just specify task numbers (e.g., "Task1", "Task4", "Task5")
# Example:
# TASKS=("Task1" "Task4" "Task5")
TASKS=()

# ============================================
# Execute Tasks
# ============================================
# Define all available tasks mapping
declare -A TASK_SCRIPTS
TASK_SCRIPTS["Task1"]="Task1_parallel_two_seller_negotiation"
TASK_SCRIPTS["Task2"]="Task2_parallel_three_seller_negotiation"
TASK_SCRIPTS["Task3"]="Task3_sequential_two_seller_negotiation"
TASK_SCRIPTS["Task4"]="Task4_sequential_three_seller_negotiation"
TASK_SCRIPTS["Task5"]="Task5_s1_used_smartphone_negotiation"
TASK_SCRIPTS["Task6"]="Task6_s2_used_car_negotiation"
TASK_SCRIPTS["Task7"]="Task7_s3_short_term_rental_negotiation"
TASK_SCRIPTS["Task8"]="Task8_s4_website_development_negotiation"
TASK_SCRIPTS["Task9"]="Task9_s5_commercial_photography_negotiation"
TASK_SCRIPTS["Task10"]="Task10_s6_home_renovation_negotiation"
TASK_SCRIPTS["Task11"]="Task11_s7_saas_procurement_negotiation"
TASK_SCRIPTS["Task12"]="Task12_s8_raw_materials_procurement_negotiation"
TASK_SCRIPTS["Task13"]="Task13_s9_luxury_watch_negotiation"
TASK_SCRIPTS["Task14"]="Task14_s10_business_acquisition_negotiation"
TASK_SCRIPTS["Task15"]="Task15_quantity_discount_negotiation"

# Determine which tasks to run
if [ ${#TASKS[@]} -eq 0 ]; then
    # If TASKS is empty, run all available tasks
    echo "TASKS list is empty. Running all available tasks..."
    TASKS_TO_RUN=("Task1" "Task2" "Task3" "Task4" "Task5" "Task6" "Task7" "Task8" "Task9" "Task10" "Task11" "Task12" "Task13" "Task14" "Task15")
else
    # Use the specified task list
    TASKS_TO_RUN=("${TASKS[@]}")
fi

echo "Tasks to run: ${#TASKS_TO_RUN[@]}"
for task_name in "${TASKS_TO_RUN[@]}"; do
    echo "  - $task_name"
done
echo ""

if [ ${#MODELS[@]} -eq 0 ]; then
    # Model list is empty: use default behavior (each script uses its own default model)
    echo "Running selected tasks (using default models)..."
    
    # Create temporary log file
    TEMP_LOG=$(mktemp)
    
    # Run each task with tee to capture output
    for task_name in "${TASKS_TO_RUN[@]}"; do
        script_name="${TASK_SCRIPTS[$task_name]}"
        
        if [ -z "$script_name" ]; then
            echo "Warning: Unknown task '$task_name', skipping..."
            continue
        fi
        
        if [ -f "${script_name}.py" ]; then
            echo ""
            echo "Running ${task_name}..."
            python "${script_name}.py" 2>&1 | tee "$TEMP_LOG"
        else
            echo "Warning: ${script_name}.py not found, skipping..."
        fi
    done
    
    # Clean up
    rm -f "$TEMP_LOG"
else
    # Model list is provided: loop through each model and run selected tasks for each model
    echo "Running selected tasks with the following model list: ${MODELS[*]}"
    for model in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Using model: $model"
        echo "=========================================="
        echo ""
        
        # Create temporary log file for this model's tasks
        TEMP_LOG=$(mktemp)
        
        for task_name in "${TASKS_TO_RUN[@]}"; do
            script_name="${TASK_SCRIPTS[$task_name]}"
            
            if [ -z "$script_name" ]; then
                echo "Warning: Unknown task '$task_name', skipping..."
                continue
            fi
            
            if [ -f "${script_name}.py" ]; then
                if [ "${SKIP_EXISTING:-0}" = "1" ] && task_result_exists "$model" "$script_name"; then
                    echo "Skipping ${task_name} (model: $model) - results already exist"
                    continue
                fi
                echo ""
                echo "Running ${task_name} (model: $model)..."
                python "${script_name}.py" --model "$model" 2>&1 | tee "$TEMP_LOG"
                save_run_history "$TEMP_LOG" "$model" "$task_name"
            else
                echo "Warning: ${script_name}.py not found, skipping ${task_name}..."
            fi
        done
        
        # Clean up temporary log file
        rm -f "$TEMP_LOG"
        
        echo ""
        echo "Completed all tasks for model: $model"
        echo ""
    done
    echo "All tasks for all models completed!"
fi
