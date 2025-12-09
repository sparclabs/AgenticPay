# AgenticPayGym

A Multi-Agent Negotiation Framework for Buyer-Seller Transactions using LLM-based Agents.

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  * [Dependencies](#dependencies)
- [Quick Start](#quick-start)
  * [Basic Single-Product Negotiation](#basic-single-product-negotiation)
  * [Multi-Product Negotiation](#multi-product-negotiation)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
  * [Environments](#environments)
    + [NegotiationEnv](#negotiationenv)
    + [MultiProductNegotiationEnv](#multiproductnegotiationenv)
  * [Agents](#agents)
    + [BaseAgent](#baseagent)
  * [Environment Registration System](#environment-registration-system)
  * [ConversationMemory](#conversationmemory)
  * [UserProfile](#userprofile)
  * [BaseLLM](#basellm)
- [Configuration](#configuration)
  * [Environment Parameters](#environment-parameters)
  * [Agent Configuration](#agent-configuration)
    + [BuyerAgent](#buyeragent)
    + [SellerAgent](#selleragent)
    + [ProductSelectorAgent](#productselectoragent)
  * [User Profile Configuration](#user-profile-configuration)
  * [LLM Configuration](#llm-configuration)
- [Examples](#examples)
  * [Available Examples](#available-examples)
  * [Running Examples](#running-examples)
- [Extending the Framework](#extending-the-framework)
  * [Adding a New LLM Provider](#adding-a-new-llm-provider)
  * [Creating Custom Agents](#creating-custom-agents)
  * [Registering New Environments](#registering-new-environments)
  * [Adding New Features](#adding-new-features)
- [License](#license)
- [Contributing](#contributing)
- [Citation](#citation)




## Overview

AgenticPayGym is a framework for simulating multi-agent negotiations between buyers and sellers. It uses Large Language Models (LLMs) as the foundation for intelligent agents that can engage in realistic price negotiations. The framework is designed with a Gymnasium-like API for easy integration and extensibility.

## Features

- 🤖 **LLM-based Agents**: Buyer and Seller agents powered by LLMs (OpenAI, etc.)
- 💬 **Multi-turn Conversations**: Support for extended negotiation dialogues
- 🧠 **Memory System**: Conversation history management for context-aware negotiations
- 📊 **State Tracking**: Comprehensive tracking of prices, rounds, and negotiation status
- 🎯 **Flexible Configuration**: Customizable negotiation parameters and agent behaviors
- 🔌 **Extensible Design**: Easy to add new agent types or LLM providers
- 🏪 **Environment Registration System**: Gymnasium-like environment registration for easy environment management
- 🛍️ **Multi-Product Negotiations**: Support for negotiating multiple products with context preservation
- 👤 **User Profiles**: Personal preference system that influences agent negotiation behavior
- 🔄 **Product Selection**: Intelligent product matching based on user requirements

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

### Basic Single-Product Negotiation

```python
from agenticpaygym import make  # Recommended: use registration system
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM
from agenticpaygym.utils.user_profile import UserProfile, StylePreference, ShoppingHabit
import os

# Initialize LLM
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# Create agents with bottom prices (confidential)
buyer_max_price = 120.0  # Maximum acceptable price for buyer
seller_min_price = 80.0   # Minimum acceptable price for seller

buyer = BuyerAgent(llm=llm, buyer_max_price=buyer_max_price)
seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)

# Create environment using registration system (recommended)
env = make(
    "Negotiation-v0",
    buyer_agent=buyer,
    seller_agent=seller,
    max_rounds=20,
    initial_seller_price=150.0,
    buyer_max_price=buyer_max_price,
    seller_min_price=seller_min_price,
    environment_info={
        "temperature": "warm",
        "season": "summer",
    },
    price_tolerance=5.0,
)

# Create user profile (optional)
user_profile = UserProfile(
    style_preference=StylePreference.BUSINESS,
    shopping_habit=ShoppingHabit.COMPARE,
)

# Reset and start negotiation
observation, info = env.reset(
    user_requirement="I need a high-quality winter jacket",
    product_info={
        "name": "Premium Winter Jacket",
        "brand": "Mountain Gear",
        "price": 180.0,
        "features": ["Waterproof", "Insulated", "Windproof"],
        "condition": "New",
        "material": "Gore-Tex",
    },
    user_profile=user_profile,  # Optional
)

# Run negotiation
done = False
while not done:
    # Get responses from both agents
    seller_action = seller.respond(
        conversation_history=observation["conversation_history"],
        current_state=observation
    )
    
    buyer_action = buyer.respond(
        conversation_history=observation["conversation_history"],
        current_state=observation
    )
    
    # Execute step with both actions
    observation, reward, terminated, truncated, info = env.step(
        buyer_action=buyer_action,
        seller_action=seller_action
    )
    done = terminated or truncated
    env.render()

print(f"Negotiation ended: {info['status']}")
print(f"Final price: ${info.get('seller_price', 'N/A')}")
env.close()
```

### Multi-Product Negotiation

```python
from agenticpaygym import make
from agenticpaygym.agents.product_selector_agent import ProductSelectorAgent

# Create product selector agent
product_selector = ProductSelectorAgent(llm=llm)

# Create multi-product environment
env = make(
    "MultiProductNegotiation-v0",
    buyer_agent=buyer,
    seller_agent=seller,
    max_rounds_per_product=20,
    initial_seller_price=150.0,
    buyer_max_price=buyer_max_price,
    seller_min_price=seller_min_price,
)

# Define available products
products = [
    {
        "name": "Premium Winter Jacket",
        "brand": "Mountain Gear",
        "price": 180.0,
        "features": ["Waterproof", "Insulated"],
    },
    {
        "name": "Running Shoes",
        "brand": "SportMax",
        "price": 120.0,
        "features": ["Lightweight", "Cushioned"],
    },
]

# First product negotiation
observation, info = env.reset(
    user_requirement="I need a winter jacket",
    product_info=products[0],
    available_products=products,
)

# ... negotiation loop ...

# Second product (preserves context)
selected_product = product_selector.select_product("I need running shoes", products)
observation, info = env.reset(
    user_requirement="I need running shoes",
    product_info=selected_product,
    clear_history=False,  # Preserve previous context
    available_products=products,
)

# ... continue negotiation ...
```

## Project Structure

```
AgenticPayGym/
├── agenticpaygym/
│   ├── __init__.py
│   ├── core.py                    # Core base environment class
│   ├── agents/
│   │   ├── base_agent.py          # Base agent class
│   │   ├── buyer_agent.py         # Buyer agent implementation
│   │   ├── seller_agent.py        # Seller agent implementation
│   │   └── product_selector_agent.py  # Product selector agent
│   ├── envs/                      # Environment implementations
│   │   ├── __init__.py            # Environment registration
│   │   ├── registration.py        # Registration system
│   │   ├── negotiation_env.py     # Single-product negotiation
│   │   └── multi_product_negotiation_env.py  # Multi-product negotiation
│   ├── memory/
│   │   └── conversation_memory.py  # Conversation history management
│   ├── llm/
│   │   ├── base_llm.py            # LLM interface
│   │   └── openai_llm.py          # OpenAI implementation
│   ├── utils/
│   │   ├── negotiation_state.py   # State management
│   │   └── user_profile.py        # User profile data structures
│   ├── spaces/                    # Action/observation spaces (for future use)
│   └── examples/
│       ├── simple_negotiation.py  # Basic single-product example
│       ├── multi_product_negotiation.py  # Multi-product example
│       └── registration_example.py  # Registration system example
├── README.md                      # This file
├── QUICKSTART.md                  # Quick start guide
├── PROJECT_STRUCTURE.md            # Detailed project structure
├── ENV_REGISTRATION.md            # Environment registration guide
├── setup.py
└── requirements.txt
```

## Core Components

### Environments

#### NegotiationEnv

Single-product negotiation environment that manages the negotiation process for one product.

**Key Methods:**
- `reset()`: Initialize a new negotiation
- `step()`: Execute one negotiation turn (accepts `buyer_action` and `seller_action`)
- `render()`: Display current negotiation state
- `close()`: Close environment and clean up

#### MultiProductNegotiationEnv

Multi-product negotiation environment that supports negotiating multiple products while preserving conversation context.

**Key Features:**
- Context preservation across products
- Product result tracking
- Flexible product switching

### Agents

#### BaseAgent

Abstract base class for all agents.

**Subclasses:**
- `BuyerAgent`: Represents the buyer, negotiates based on user requirements and budget
- `SellerAgent`: Represents the seller, negotiates based on product information and market conditions
- `ProductSelectorAgent`: Selects the most appropriate product from available products based on user requirements

### Environment Registration System

Gymnasium-like environment registration system for easy environment management.

**Key Functions:**
- `make()`: Create environment instance by ID
- `register()`: Register new environment
- `spec()`: Get environment specification
- `pprint_registry()`: List all registered environments

**Usage:**
```python
from agenticpaygym import make

env = make("Negotiation-v0", buyer_agent=buyer, seller_agent=seller, max_rounds=20)
```

### ConversationMemory

Manages conversation history and context.

**Features:**
- Message storage with metadata
- History retrieval (full or recent)
- Role-based filtering

### UserProfile

User preference system that influences buyer agent behavior.

**Attributes:**
- `style_preference`: Simple, Business, or Traditional
- `shopping_habit`: Compare prices or Direct purchase

### BaseLLM

Abstract interface for LLM providers.

**Implementations:**
- `OpenAILLM`: OpenAI API integration

## Configuration

### Environment Parameters

#### NegotiationEnv / Negotiation-v0

- `max_rounds`: Maximum number of negotiation rounds
- `initial_seller_price`: Starting price from seller
- `buyer_max_price`: Maximum acceptable price for buyer (confidential)
- `seller_min_price`: Minimum acceptable price for seller (confidential)
- `price_tolerance`: Price difference threshold for agreement
- `environment_info`: Contextual information (weather, season, etc.)

#### MultiProductNegotiationEnv / MultiProductNegotiation-v0

- `max_rounds_per_product`: Maximum rounds per product negotiation
- `initial_seller_price`: Starting price from seller
- `buyer_max_price`: Maximum acceptable price for buyer
- `seller_min_price`: Minimum acceptable price for seller
- `price_tolerance`: Price difference threshold for agreement
- `environment_info`: Contextual information

### Agent Configuration

#### BuyerAgent

- `buyer_max_price`: Maximum acceptable purchase price (confidential)
- `name`: Agent name
- `role_description`: Custom role description

#### SellerAgent

- `seller_min_price`: Minimum acceptable selling price (confidential)
- `name`: Agent name
- `role_description`: Custom role description

#### ProductSelectorAgent

- `name`: Agent name
- `role_description`: Custom role description

### User Profile Configuration

```python
from agenticpaygym.utils.user_profile import UserProfile, StylePreference, ShoppingHabit

user_profile = UserProfile(
    style_preference=StylePreference.BUSINESS,  # SIMPLE, BUSINESS, TRADITIONAL
    shopping_habit=ShoppingHabit.COMPARE,     # COMPARE, DIRECT
)
```

### LLM Configuration

- `model`: Model name (e.g., "gpt-4", "gpt-4o-mini-2024-07-18", "gpt-3.5-turbo")
- `api_key`: API key for the LLM provider
- `temperature`: Generation temperature (set in `generate()` method)

## Examples

### Available Examples

1. **Simple Negotiation** (`simple_negotiation.py`)
   - Basic single-product negotiation
   - Demonstrates environment registration system
   - Shows user profile usage

2. **Multi-Product Negotiation** (`multi_product_negotiation.py`)
   - Negotiate multiple products sequentially
   - Context preservation across products
   - Product selection using ProductSelectorAgent

3. **Registration Example** (`registration_example.py`)
   - Detailed guide on environment registration
   - Custom environment creation
   - Namespace and version management

### Running Examples

```bash
# Set API key
export OPENAI_API_KEY="your-api-key"

# Run simple negotiation
python -m agenticpaygym.examples.simple_negotiation

# Run multi-product negotiation
python -m agenticpaygym.examples.multi_product_negotiation

# Run registration example
python -m agenticpaygym.examples.registration_example
```

## Extending the Framework

### Adding a New LLM Provider

1. Create a new class inheriting from `BaseLLM`
2. Implement the `generate()` method
3. Add any provider-specific configuration
4. Export in `llm/__init__.py` (optional)

Example:
```python
from agenticpaygym.llm.base_llm import BaseLLM

class MyCustomLLM(BaseLLM):
    def generate(self, prompt, **kwargs):
        # Your implementation
        return response
```

### Creating Custom Agents

1. Inherit from `BaseAgent` or existing agent classes
2. Implement the `respond()` method
3. Customize prompt building as needed

Example:
```python
from agenticpaygym.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def respond(self, conversation_history, current_state):
        # Custom response logic
        return response
```

### Registering New Environments

1. Create a new environment class inheriting from `BaseEnv`
2. Implement `reset()` and `step()` methods
3. Register using the registration system

Example:
```python
from agenticpaygym.core import BaseEnv
from agenticpaygym.envs import register

class MyCustomEnv(BaseEnv):
    def reset(self, **kwargs):
        # Implementation
        return observation, info
    
    def step(self, action):
        # Implementation
        return observation, reward, terminated, truncated, info

# Register environment
register(
    id="MyCustomEnv-v0",
    entry_point="agenticpaygym.envs.my_custom_env:MyCustomEnv",
    max_episode_steps=100,
)
```

### Adding New Features

The framework is designed to be extensible. Key extension points:
- Custom reward functions
- Advanced price extraction
- Custom negotiation strategies
- Learning-based agent behaviors
- Additional agent types
- Custom memory systems

For detailed guides, see:
- `ENV_REGISTRATION.md` - Environment registration system
- `PROJECT_STRUCTURE.md` - Project structure and extension points
- `QUICKSTART.md` - Quick start guide

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use AgenticPayGym in your research, please cite:

```bibtex
@software{agenticpaygym2025,
  title={AgenticPayGym: A Multi-Agent Negotiation Framework},
  author={The AgenticPayGym Team},
  year={2025},
  url={https://github.com/username/AgenticPayGym}
}
```

