# AgenticPayGym Quick Start Guide

## Installation

```bash
cd AgenticPayGym
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Set API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or in code:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 2. Simplest Example

```python
from agenticpaygym import NegotiationEnv, BuyerAgent, SellerAgent
from agenticpaygym.models.custom_llm import CustomLLM
import os

# Initialize LLM
llm = CustomLLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# Create Agents
buyer = BuyerAgent(llm=llm)
seller = SellerAgent(llm=llm)

# Create environment
env = NegotiationEnv(
    buyer_agent=buyer,
    seller_agent=seller,
    max_rounds=10,
    initial_seller_price=150.0,
    buyer_max_price=100.0,
    environment_info={
        "temperature": "warm",
        "season": "summer",
    }
)

# Start negotiation
observation, info = env.reset(
    user_requirement="I need a high-quality winter jacket",
    product_info={
        "name": "Premium Winter Jacket",
        "brand": "Mountain Gear",
        "features": ["Waterproof", "Insulated"],
    }
)

# Negotiation loop
done = False
while not done:
    current_agent = observation["agent_role"]
    
    if current_agent == "buyer":
        action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
    else:
        action = seller.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
    
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

print(f"Final status: {info['status']}")
print(f"Final price: ${info.get('seller_price', 'N/A')}")
```

### 3. Run Example

```bash
python -m agenticpaygym.examples.simple_negotiation
```

### 4. Run Tests

```bash
python test_basic.py
```

## Configuration Options

### NegotiationEnv Parameters

- `max_rounds`: Maximum negotiation rounds (default: 10)
- `initial_seller_price`: Initial price offered by seller (default: 150.0)
- `buyer_max_price`: Maximum acceptable price for buyer (default: 100.0)
- `price_tolerance`: Price tolerance for determining agreement (default: 5.0)
- `environment_info`: Environment information dictionary (weather, season, etc.)

### Agent Parameters

- `name`: Agent name
- `role_description`: Role description (used in prompt)

### LLM Parameters

- `model`: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
- `api_key`: API key
- `temperature`: Generation temperature (set in `generate()` method)

## Frequently Asked Questions

### Q: How to not use OpenAI API?

A: Implement a custom LLM class, inherit from `BaseLLM`:

```python
from agenticpaygym.models.base_llm import BaseLLM

class MyCustomLLM(BaseLLM):
    def generate(self, prompt, **kwargs):
        # Your implementation
        return response
    
    def __repr__(self):
        return "MyCustomLLM()"
```

### Q: How to customize Agent behavior?

A: Inherit from `BaseAgent` or `BuyerAgent`/`SellerAgent`:

```python
from agenticpaygym.agents.buyer_agent import BuyerAgent

class CustomBuyerAgent(BuyerAgent):
    def respond(self, conversation_history, current_state):
        # Custom response logic
        return super().respond(conversation_history, current_state)
```

### Q: How to improve price extraction?

A: Override the `NegotiationEnv._extract_price()` method:

```python
class CustomNegotiationEnv(NegotiationEnv):
    def _extract_price(self, text: str) -> Optional[float]:
        # Your improved implementation
        # Can use more complex NLP methods
        return price
```

## Next Steps

- Check `README.md` for complete documentation
- Check `PROJECT_STRUCTURE.md` for project structure
- Run example code to learn usage
- Extend and customize the framework as needed
