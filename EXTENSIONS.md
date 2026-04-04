# Recent Extensions

## 1. vLLM HTTP Endpoint Support

A new `VLLMHttpLLM` model backend ([agenticpay/models/vllm_http_llm.py](agenticpay/models/vllm_http_llm.py)) connects to an already-running vLLM HTTP server via its OpenAI-compatible API. Unlike the existing in-process `VLLMLLM`, this requires no local GPU — inference is delegated to an external endpoint.

**Key details:**
- Configured via `VLLM_BASE_URL` and `VLLM_MODEL` environment variables (or `config.py`)
- Validates server reachability at startup with a lightweight `/v1/models` health check
- Supports Qwen3 chain-of-thought reasoning via `enable_thinking` flag

## 2. Anthropic / Claude Model Support

A new `AnthropicLLM` backend ([agenticpay/models/anthropic_llm.py](agenticpay/models/anthropic_llm.py)) wraps the official Anthropic SDK to run negotiations against Claude models (e.g. `claude-opus-4-5`, `claude-sonnet-4-5`).

**Key details:**
- Set `MODEL_MODE=anthropic` and `ANTHROPIC_API_KEY` to use it
- Model selected via `MODEL_PATH` config variable

## 3. Unified Model Configuration

`config.py` ([agenticpay/examples/config.py](agenticpay/examples/config.py)) was extended with a single `get_model()` factory that dispatches across all four backends based on the `MODEL_MODE` env var: `local`, `vllm`, `cloud`, `anthropic`. All existing task examples were updated to use this factory instead of inline model setup.

## 4. Difficulty Levels

A ZOPA difficulty system was added to `config.py`, controlled by the `DIFFICULTY` env var:

- `normal` — original buyer_max / seller_min, deal always possible
- `hard` — ZOPA tightened to 5% of original spread
- `no_deal` — buyer ceiling set 5% below seller floor, no rational deal exists

The `adjust_zopa(buyer_max, seller_min, difficulty)` helper computes scenario-specific price bounds. All existing task examples were updated to call it.

## 5. Quantity / Bulk-Discount Negotiation (Task 15)

A new task type where agents negotiate **both quantity and per-unit price** simultaneously, with seller-side tiered volume pricing.

Task 15 environments were added across all environment categories:

| Environment category | Class |
|---|---|
| Single buyer × single seller | `Task15QuantityDiscountNegotiation` |
| Multi-buyer | `Task15ParallelTwoBuyerQuantityDiscountNegotiation` |
| Multi-seller | `Task15ParallelTwoSellerQuantityDiscountNegotiation` |
| Multi-product | `Task15MultiProductQuantityDiscountNegotiation` |
| Multi-buyer × multi-seller | `Task15TwoBuyerTwoSellerQuantityDiscountNegotiation` |
| Multi-buyer × multi-product | `Task15TwoBuyerMultiProductQuantityDiscountNegotiation` |
| Multi-product × multi-seller | `Task15MultiProductTwoSellerQuantityDiscountNegotiation` |
| Multi-buyer × multi-seller × multi-product | `Task15TwoBuyerTwoSellerMultiProductQuantityDiscountNegotiation` |

All variants are registered in the environment registry (`agenticpay/envs/__init__.py`).

## 5. Evaluation Script

A new `agenticpay/experiments/evaluate.py` script aggregates results from batch runs.

**Metrics computed:**
- **Deal Rate** — percentage of episodes that reached agreement
- **Overflow Rate** — percentage where the agreed price violated a private constraint (buyer paid above their max or seller accepted below their floor)
- **GlobalScore**, **BuyerScore**, **SellerScore** — mean scores across runs

Results can be filtered by model name (`--model`) and exported to CSV (`--csv`). The script handles all environment topologies (single, multi-buyer, multi-seller, multi-product) by parsing the appropriate price fields from each `summary.json`.
