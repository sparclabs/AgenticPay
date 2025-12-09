# AgenticPayGym

A Multi-Agent Negotiation Framework for Buyer-Seller Transactions using LLM-based Agents.

## Overview

AgenticPayGym is a framework for simulating multi-agent negotiations between buyers and sellers. It uses Large Language Models (LLMs) as the foundation for intelligent agents that can engage in realistic price negotiations. The framework is designed with a Gymnasium-like API for easy integration and extensibility.

## Features

- 🤖 **LLM-based Agents**: Buyer and Seller agents powered by LLMs (OpenAI, etc.)
- 💬 **Multi-turn Conversations**: Support for extended negotiation dialogues
- 🧠 **Memory System**: Conversation history management for context-aware negotiations
- 📊 **State Tracking**: Comprehensive tracking of prices, rounds, and negotiation status
- 🎯 **Flexible Configuration**: Customizable negotiation parameters and agent behaviors
- 🔌 **Extensible Design**: Easy to add new agent types or LLM providers

## Installation

```bash
cd AgenticPayGym
pip install -e .
```

### Dependencies

- Python 3.10+
- openai (for OpenAI LLM support)
- Other dependencies listed in `requirements.txt`

## Quick Start

```python
from agenticpaygym.core import NegotiationEnv
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM
import os

# Initialize LLM
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# Create agents
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

# Reset and start negotiation
observation, info = env.reset(
    user_requirement="I need a high-quality winter jacket",
    product_info={
        "name": "Premium Winter Jacket",
        "brand": "Mountain Gear",
        "features": ["Waterproof", "Insulated"],
    }
)

# Run negotiation
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

print(f"Negotiation ended: {info['status']}")
```

## Project Structure

```
AgenticPayGym/
├── agenticpaygym/
│   ├── __init__.py
│   ├── core.py                    # Core negotiation environment
│   ├── agents/
│   │   ├── base_agent.py          # Base agent class
│   │   ├── buyer_agent.py         # Buyer agent implementation
│   │   └── seller_agent.py        # Seller agent implementation
│   ├── memory/
│   │   └── conversation_memory.py  # Conversation history management
│   ├── llm/
│   │   ├── base_llm.py            # LLM interface
│   │   └── openai_llm.py          # OpenAI implementation
│   ├── utils/
│   │   └── negotiation_state.py   # State management
│   └── examples/
│       └── simple_negotiation.py  # Example usage
├── README.md
├── setup.py
└── requirements.txt
```

## Core Components

### NegotiationEnv

The main environment class that manages the negotiation process.

**Key Methods:**
- `reset()`: Initialize a new negotiation
- `step()`: Execute one negotiation turn
- `render()`: Display current negotiation state

### BaseAgent

Abstract base class for all agents.

**Subclasses:**
- `BuyerAgent`: Represents the buyer
- `SellerAgent`: Represents the seller

### ConversationMemory

Manages conversation history and context.

**Features:**
- Message storage with metadata
- History retrieval (full or recent)
- Role-based filtering

### BaseLLM

Abstract interface for LLM providers.

**Implementations:**
- `OpenAILLM`: OpenAI API integration

## Configuration

### Environment Parameters

- `max_rounds`: Maximum number of negotiation rounds
- `initial_seller_price`: Starting price from seller
- `buyer_max_price`: Maximum acceptable price for buyer
- `price_tolerance`: Price difference threshold for agreement
- `environment_info`: Contextual information (weather, season, etc.)

### Agent Configuration

Agents can be customized with:
- Role descriptions
- LLM model selection
- Temperature and other generation parameters

## Examples

See `agenticpaygym/examples/simple_negotiation.py` for a complete example.

Run the example:
```bash
export OPENAI_API_KEY="your-api-key"
python -m agenticpaygym.examples.simple_negotiation
```

## Extending the Framework

### Adding a New LLM Provider

1. Create a new class inheriting from `BaseLLM`
2. Implement the `generate()` method
3. Add any provider-specific configuration

### Creating Custom Agents

1. Inherit from `BaseAgent`
2. Implement the `respond()` method
3. Customize prompt building as needed

### Adding New Features

The framework is designed to be extensible. Key extension points:
- Custom reward functions
- Advanced price extraction
- Multi-product negotiations
- Learning-based strategies

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use AgenticPayGym in your research, please cite:

```bibtex
@software{agenticpaygym2024,
  title={AgenticPayGym: A Multi-Agent Negotiation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/AgenticPayGym}
}
```

