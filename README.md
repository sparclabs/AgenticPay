

<h1 align="center" style="font-size: 30px;"><strong><em>AgenticPayGym</em></strong>: A Multi-Agent LLM Negotiation System for Buyer–Seller Transactions</h1>

**A Multi-Agent Negotiation Framework for Buyer-Seller Transactions using LLM-based Agents.**

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
  * [Agents](#agents)
    + [BaseAgent](#baseagent)
  * [Environment Registration System](#environment-registration-system)
  * [ConversationMemory](#conversationmemory)
  * [UserProfile](#userprofile)
  * [BaseLLM and BaseVLM](#basellm-and-basevlm)
- [Configuration](#configuration)
  * [Environment Parameters](#environment-parameters)
  * [Agent Configuration](#agent-configuration)
    + [BuyerAgent](#buyeragent)
    + [SellerAgent](#selleragent)
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

- 🤖 **LLM/VLM-based Agents**: Buyer and Seller agents powered by LLMs and Vision Language Models (OpenAI, HuggingFace, etc.) supporting both text and image-based negotiations
- 💬 **Multi-turn Conversations**: Support for extended negotiation dialogues
- 🧠 **Memory System**: Conversation history management for context-aware negotiations
- 📊 **State Tracking**: Comprehensive tracking of prices, rounds, and negotiation status
- 🎯 **Flexible Configuration**: Customizable negotiation parameters and agent behaviors
- 🔌 **Extensible Design**: Easy to add new agent types or LLM/VLM providers
- 🏪 **Environment Registration System**: Gymnasium-like environment registration for easy environment management
- 🛍️ **Multi-Product Negotiations**: Support for negotiating multiple products with context preservation
- 👥 **Multi-Agent Scenarios**: Support for multiple buyers, sellers, and products in various combinations
- 🔄 **Parallel & Sequential Negotiations**: Support for both parallel and sequential negotiation modes
- 👤 **User Profiles**: Personal preference system that influences agent negotiation behavior

## Installation

```bash
cd AgenticPayGym
pip install -e .
```

### Dependencies

- Python 3.10+
- openai (for OpenAI LLM/VLM support)
- transformers (for HuggingFace LLM/VLM support, optional)
- torch (for HuggingFace models, optional)
- Other dependencies listed in `requirements.txt`

## Quick Start

### Basic Single-Product Negotiation

```python
from agenticpaygym import make  # Recommended: use registration system
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.models.openai_llm import OpenAILLM
import os

# Initialize LLM/VLM (supports local, online, and API modes)
# Option 1: OpenAI API (online)
from agenticpaygym.models.openai_llm import OpenAILLM
from agenticpaygym.models.openai_vlm import OpenAIVLM
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
vlm = OpenAIVLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-vision-preview")

# Option 2: HuggingFace (local/online)
# from agenticpaygym.models.huggingface_llm import HuggingFaceLLM
# from agenticpaygym.models.huggingface_vlm import HuggingFaceVLM
# llm = HuggingFaceLLM(model_name="meta-llama/Llama-2-7b-chat-hf", device="cuda")
# vlm = HuggingFaceVLM(model_name="llava-hf/llava-1.5-7b-hf", device="cuda")

# Create agents with bottom prices (confidential)
buyer_max_price = 120.0  # Maximum acceptable price for buyer
seller_min_price = 80.0   # Minimum acceptable price for seller

buyer = BuyerAgent(llm=llm, buyer_max_price=buyer_max_price)
seller = SellerAgent(llm=llm, seller_min_price=seller_min_price)

# Create environment using registration system (recommended)
env = make(
    "Task1_basic_price_negotiation-v0",
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

# User description (optional)
user_description = "A business professional who prefers comparing prices before making decisions"

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
    user_description=user_description,  # Optional
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

# Create multi-product environment
env = make(
    "Task1_multi_product_negotiation-v0",
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
observation, info = env.reset(
    user_requirement="I need running shoes",
    product_info=products[1],
    clear_history=False,  # Preserve previous context
    available_products=products,
)

# ... continue negotiation ...
```

## Project Structure

```
AgenticPayGym/
├── agenticpaygym/
│   ├── agents/                    # Agent implementations (buyer, seller)
│   ├── envs/                      # Environment implementations
│   │   ├── single_buyer_product_seller/  # Basic negotiation
│   │   ├── only_multi_products/   # Multi-product scenarios
│   │   ├── only_multi_seller/     # Multi-seller scenarios
│   │   ├── only_multi_buyer/      # Multi-buyer scenarios
│   │   └── multi_*/               # Complex multi-agent scenarios
│   ├── models/                    # LLM/VLM implementations (OpenAI, HuggingFace, Custom)
│   ├── memory/                    # Conversation history management
│   ├── utils/                     # Utilities (state, user profile)
│   └── examples/                   # Example scripts organized by scenario
├── README.md
├── setup.py
└── requirements.txt
```

## Core Components

### Environments

The framework provides a comprehensive set of negotiation environments organized by complexity:

#### Single Buyer + Product + Seller (`single_buyer_product_seller/`)

Basic negotiation scenarios with one buyer, one product, and one seller.

- **Task1: Basic Price Negotiation** - Fundamental price negotiation environment
- **Task2: Close Price Negotiation** - Tests edge cases with narrow price ranges
- **Task3: Close to Market Price Negotiation** - Tests scenarios near market price

#### Only Multi-Products (`only_multi_products/`)

Environments for negotiating multiple products with a single buyer and seller.

- **Task1: Multi-Product Negotiation** - General multi-product negotiation
- **Task2: Two Product Negotiation** - Two products negotiation
- **Task3: Five Product Negotiation** - Five products negotiation
- **Task4: Select Three from Five Negotiation** - Product selection and negotiation

#### Only Multi-Seller (`only_multi_seller/`)

Environments with multiple sellers competing for a single buyer.

- **Task1-2: Parallel Multi-Seller** - Parallel negotiations with multiple sellers
- **Task3-4: Sequential Multi-Seller** - Sequential negotiations with multiple sellers

#### Only Multi-Buyer (`only_multi_buyer/`)

Environments with multiple buyers competing for products.

- **Task1-2: Parallel Multi-Buyer** - Parallel negotiations with multiple buyers
- **Task3-4: Sequential Multi-Buyer** - Sequential negotiations with multiple buyers

#### Multi-Buyer Multi-Seller (`multi_buyer_multi_seller/`)

Complex environments with multiple buyers and multiple sellers.

#### Multi-Products Multi-Seller (`multi_products_multi_seller/`)

Environments with multiple products and multiple sellers.

#### Multi-Buyer Multi-Products (`multi_buyer_multi_products/`)

Environments with multiple buyers and multiple products.

#### Multi-Buyer Multi-Products Multi-Seller (`multi_buyer_multi_products_multi_seller/`)

Most complex environments with multiple buyers, products, and sellers.

**Common Environment Methods:**
- `reset()`: Initialize a new negotiation
- `step()`: Execute one negotiation turn (accepts agent actions)
- `render()`: Display current negotiation state
- `close()`: Close environment and clean up

### Agents

#### BaseAgent

Abstract base class for all agents.

**Subclasses:**
- `BuyerAgent`: Represents the buyer, negotiates based on user requirements and budget
- `SellerAgent`: Represents the seller, negotiates based on product information and market conditions

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

# Single buyer/product/seller
env = make("Task1_basic_price_negotiation-v0", buyer_agent=buyer, seller_agent=seller, max_rounds=20)

# Multi-product
env = make("Task1_multi_product_negotiation-v0", buyer_agent=buyer, seller_agent=seller, max_rounds_per_product=20)

# Multi-seller
env = make("Task1_parallel_two_seller_negotiation-v0", buyer_agent=buyer, seller_agents=[seller1, seller2], max_rounds=20)
```

### ConversationMemory

Manages conversation history and context.

**Features:**
- Message storage with metadata
- History retrieval (full or recent)
- Role-based filtering

### UserProfile

User description system that influences agent negotiation behavior. Currently uses a simple string description passed to agents.

### BaseLLM and BaseVLM

Abstract interfaces for LLM and Vision Language Model (VLM) providers.

**LLM Implementations:**
- `OpenAILLM`: OpenAI API integration for text models
- `HuggingFaceLLM`: HuggingFace model integration

**VLM Implementations:**
- `OpenAIVLM`: OpenAI API integration for vision-language models
- `HuggingFaceVLM`: HuggingFace vision-language model integration

## Configuration

### Environment Parameters

Common parameters across environments:
- `max_rounds`: Maximum number of negotiation rounds
- `initial_seller_price`: Starting price from seller
- `buyer_max_price`: Maximum acceptable price for buyer (confidential)
- `seller_min_price`: Minimum acceptable price for seller (confidential)
- `price_tolerance`: Price difference threshold for agreement
- `environment_info`: Contextual information (weather, season, etc.)

### Agent Configuration

- **BuyerAgent**: `buyer_max_price` (maximum acceptable purchase price)
- **SellerAgent**: `seller_min_price` (minimum acceptable selling price)

### User Profile

User description is passed as a string to agents during negotiation initialization.

### LLM/VLM Configuration

Supports multiple providers:
- **OpenAI** (API): `OpenAILLM`, `OpenAIVLM` - requires API key
- **HuggingFace** (local/online): `HuggingFaceLLM`, `HuggingFaceVLM` - requires model name and device

## Examples

### Available Examples

Examples are organized by environment category:

1. **Single Buyer + Product + Seller** (`examples/single_buyer_product_seller/`)
   - `Task1_basic_price_negotiation.py` - Basic price negotiation
   - `Task2_close_price_negotiation.py` - Close price negotiation
   - `Task3_close_to_market_price_negotiation.py` - Market price negotiation
   - `registration_example.py` - Registration system demonstration

2. **Multi-Product Negotiations** (`examples/only_multi_products/`)
   - Multiple products negotiation examples
   - Product selection scenarios

3. **Multi-Seller Negotiations** (`examples/only_multi_seller/`)
   - Parallel and sequential multi-seller scenarios

4. **Multi-Buyer Negotiations** (`examples/only_multi_buyer/`)
   - Parallel and sequential multi-buyer scenarios

5. **Complex Multi-Agent Scenarios**
   - `examples/multi_buyer_multi_seller/` - Multiple buyers and sellers
   - `examples/multi_products_multi_seller/` - Multiple products and sellers
   - `examples/multi_buyer_multi_products/` - Multiple buyers and products
   - `examples/multi_buyer_multi_products_multi_seller/` - Full multi-agent scenarios

### Running Examples

```bash
# Set API key
export OPENAI_API_KEY="your-api-key"

# Run single buyer/product/seller examples
python -m agenticpaygym.examples.single_buyer_product_seller.Task1_basic_price_negotiation
python -m agenticpaygym.examples.single_buyer_product_seller.Task2_close_price_negotiation
python -m agenticpaygym.examples.single_buyer_product_seller.Task3_close_to_market_price_negotiation

# Run multi-product examples
python -m agenticpaygym.examples.only_multi_products.Task1_multi_product_negotiation

# Run multi-seller examples
python -m agenticpaygym.examples.only_multi_seller.Task1_parallel_two_seller_negotiation

# Run registration example
python -m agenticpaygym.examples.single_buyer_product_seller.registration_example
```

## Extending the Framework

### Adding a New LLM Provider

1. Create a new class inheriting from `BaseLLM` or `BaseVLM`
2. Implement the `generate()` method
3. Add any provider-specific configuration
4. Export in `models/__init__.py` (optional)

Example for LLM:
```python
from agenticpaygym.models.base_llm import BaseLLM

class MyCustomLLM(BaseLLM):
    def generate(self, prompt, **kwargs):
        # Your implementation
        return response
```

Example for VLM:
```python
from agenticpaygym.models.base_vlm import BaseVLM

class MyCustomVLM(BaseVLM):
    def generate(self, prompt, images=None, **kwargs):
        # Your implementation with image support
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
@misc{agenticpaygym2025,
    title={AgenticPayGym: A Multi-Agent LLM Negotiation System for Buyer–Seller Transactions},
    author={The AgenticPayGym Team},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/SafeRL-Lab/AgenticPayGym}},
}
```

