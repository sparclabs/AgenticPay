"""Configuration file for AgenticPayGym examples

This file contains common configuration variables used across different
negotiation examples, including reward weights, aggregation methods, and
environment parameters.
"""

import os

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
# model_mode: "local"  - load model in-process via vLLM (requires GPU + vllm package)
#             "cloud"  - cloud API (OpenAI, etc.)
#             "vllm"   - connect to a running vLLM HTTP server (vllm serve)
# Can be overridden by the MODEL_MODE environment variable.
MODEL_MODE = os.environ.get("MODEL_MODE", "local")
# MODEL_PATH: For local mode, use local model path; for cloud mode, use online model name (e.g. "gpt-4", "qwen-turbo")
MODEL_PATH = "/path/to/local/model"

# OpenAI API key and URL
OPENAI_API_KEY = "your-api-key-here"
OPENAI_URL = "your-url-here"

# vLLM HTTP server configuration (used when MODEL_MODE = "vllm")
# Values are read from environment variables if set, otherwise fall back to the defaults below.
# Start the server first: vllm serve <model-name-or-path> --port 8000
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def get_model():
    """Return a model instance configured by MODEL_MODE.

    Modes:
        "vllm"  – connects to a running vLLM HTTP server (set VLLM_BASE_URL / VLLM_MODEL)
        "cloud" – cloud API via CustomLLM (set OPENAI_API_KEY / OPENAI_URL / MODEL_PATH)
        "local" – loads the model in-process via vLLM (set MODEL_PATH)
    """
    if MODEL_MODE == "vllm":
        from agenticpay.models.vllm_http_llm import VLLMHttpLLM
        return VLLMHttpLLM(model=VLLM_MODEL, base_url=VLLM_BASE_URL)

    if MODEL_MODE == "cloud":
        from agenticpay.models.custom_llm import CustomLLM
        _url = OPENAI_URL if OPENAI_URL != "your-url-here" else None
        _key = OPENAI_API_KEY if OPENAI_API_KEY != "your-api-key-here" else os.environ.get("OPENAI_API_KEY")
        if not _key:
            raise ValueError(
                "Cloud mode requires an API key. "
                "Set OPENAI_API_KEY in config.py or as an environment variable."
            )
        return CustomLLM(api_key=_key, model=MODEL_PATH, base_url=_url)

    if MODEL_MODE == "local":
        from agenticpay.models.vllm_lm import VLLMLLM
        return VLLMLLM(model_path=MODEL_PATH, trust_remote_code=True)

    raise ValueError(
        f"Unknown MODEL_MODE {MODEL_MODE!r}. Must be 'vllm', 'cloud', or 'local'."
    )


def get_model_name(model) -> str:
    """Extract a human-readable name from a model object."""
    for attr in ("model", "model_id"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    model_path = getattr(model, "model_path", None)
    if model_path:
        return os.path.basename(str(model_path).strip()) or str(model)
    model_str = str(model).strip()
    if "model=" in model_str:
        try:
            return model_str.split("model=")[1].split(")")[0]
        except Exception:
            pass
    return model_str or "unknown_model"
