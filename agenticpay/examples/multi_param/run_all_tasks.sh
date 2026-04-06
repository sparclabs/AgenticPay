#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

# Get project root (4 levels up from script directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_BASE="$PROJECT_ROOT/agenticpay/results/multi_param"

# Function to find the latest result directory for a model
find_latest_result_dir() {
    local model_name="$1"
    local model_name_safe=$(echo "$model_name" | sed 's/[\/\\:]/_/g')
    local model_dir="$RESULTS_BASE/$model_name_safe"

    if [ -d "$model_dir" ]; then
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

    local latest_dir=$(find_latest_result_dir "$model_name")

    if [ -n "$latest_dir" ] && [ -d "$latest_dir" ]; then
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
MODELS=("google/gemma-4-31B-it")

# ============================================
# Configuration: Task List
# ============================================
# Leave empty to run all tasks, or specify e.g. TASKS=("Task1" "Task3")
TASKS=()

# ============================================
# Task definitions
# ============================================
declare -A TASK_SCRIPTS
TASK_SCRIPTS["Task1"]="Task1_price_quality_negotiation"
TASK_SCRIPTS["Task2"]="Task2_price_quality_delivery_negotiation"
TASK_SCRIPTS["Task3"]="Task3_price_quality_delivery_warranty_negotiation"
TASK_SCRIPTS["Task4"]="Task4_all_params_negotiation"

# Determine which tasks to run
if [ ${#TASKS[@]} -eq 0 ]; then
    echo "TASKS list is empty. Running all available tasks..."
    TASKS_TO_RUN=("Task1" "Task2" "Task3" "Task4")
else
    TASKS_TO_RUN=("${TASKS[@]}")
fi

echo "Tasks to run: ${#TASKS_TO_RUN[@]}"
for task_name in "${TASKS_TO_RUN[@]}"; do
    echo "  - $task_name: ${TASK_SCRIPTS[$task_name]}"
done
echo ""

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Running selected tasks (using default models)..."

    TEMP_LOG=$(mktemp)

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

    rm -f "$TEMP_LOG"
else
    echo "Running selected tasks with models: ${MODELS[*]}"

    for model in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Using model: $model"
        echo "=========================================="
        echo ""

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

        rm -f "$TEMP_LOG"

        echo ""
        echo "Completed all tasks for model: $model"
        echo ""
    done

    echo "All tasks for all models completed!"
fi
