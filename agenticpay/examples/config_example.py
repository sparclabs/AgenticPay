"""Configuration file for AgenticPayGym examples

This file contains common configuration variables used across different
negotiation examples, including reward weights, aggregation methods, and
environment parameters.
"""

# Reward weights configuration
# These weights control the relative importance of different reward components
reward_weights = {
    "buyer_savings": 1.0,      # Buyer savings weight
    "seller_profit": 1.0,      # Seller profit weight
    "time_cost": 0.1,          # Time cost weight (reduced impact)
}

# Reward aggregation methods
# Options: "average", "max", "min"
buyer_reward_aggregation = "average"  # Buyer reward aggregation method
seller_reward_aggregation = "average"  # Seller reward aggregation method

# Environment parameters
max_rounds = 20  # Maximum negotiation rounds
price_tolerance = 0.0  # Price tolerance (used to determine if prices match)

# Model configuration
# model_mode: "local" (local deployment) or "cloud" (cloud API)
MODEL_MODE = "local"
# MODEL_PATH: For local mode, use local model path; for cloud mode, use online model name (e.g. "gpt-4", "qwen-turbo")
MODEL_PATH = "/path/to/local/model"

# OpenAI API key and URL
OPENAI_API_KEY = "your-api-key-here"
OPENAI_URL = "your-url-here"
