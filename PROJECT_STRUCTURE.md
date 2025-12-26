# AgenticPayGym Project Structure

## Directory Structure

```
AgenticPayGym/
├── agenticpaygym/              # Main package directory
│   ├── __init__.py             # Package initialization, exports main interfaces
│   ├── core.py                 # Core negotiation environment class
│   │
│   ├── agents/                 # Agent module
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Agent base class
│   │   ├── buyer_agent.py      # Buyer Agent implementation
│   │   └── seller_agent.py     # Seller Agent implementation
│   │
│   ├── memory/                 # Memory system
│   │   ├── __init__.py
│   │   └── conversation_memory.py  # Conversation memory management
│   │
│   ├── models/                    # LLM interface
│   │   ├── __init__.py
│   │   ├── base_llm.py         # LLM base class interface
│   │   └── custom_llm.py       # Custom LLM implementation
│   │
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py
│   │   └── negotiation_state.py  # Negotiation state management
│   │
│   └── examples/               # Example code
│       ├── __init__.py
│       └── simple_negotiation.py  # Simple negotiation example
│
├── README.md                   # Project documentation
├── PROJECT_STRUCTURE.md        # Project structure documentation (this file)
├── setup.py                    # Installation configuration
├── requirements.txt            # Dependency list
├── .gitignore                  # Git ignore file
└── test_basic.py               # Basic functionality tests

```

## Core Module Descriptions

### 1. core.py - Core Environment

**NegotiationEnv**: Main negotiation environment class
- `reset()`: Initialize new negotiation
- `step()`: Execute one negotiation step
- `render()`: Render current state
- `close()`: Close environment

**NegotiationStatus**: Negotiation status enumeration
- ONGOING: In progress
- AGREED: Agreement reached
- FAILED: Failed
- TIMEOUT: Timeout

### 2. agents/ - Agent Module

**BaseAgent**: Base class for all Agents
- `initialize()`: Initialize Agent context
- `respond()`: Generate response (abstract method)
- `_build_prompt()`: Build prompt

**BuyerAgent**: Buyer Agent
- Negotiates based on user requirements and budget

**SellerAgent**: Seller Agent
- Negotiates based on product information and market conditions

### 3. memory/ - Memory System

**ConversationMemory**: Conversation memory management
- `add_message()`: Add message
- `get_history()`: Get conversation history
- `get_history_by_role()`: Get messages by role
- `clear()`: Clear memory

### 4. models/ - LLM Interface

**BaseLLM**: LLM interface base class
- `generate()`: Generate text (abstract method)

**OpenAILLM**: Custom LLM implementation
- Uses OpenAI API for text generation

### 5. utils/ - Utility Modules

**NegotiationState**: Negotiation state dataclass
- Manages negotiation state information
- `update()`: Update state
- `to_dict()`: Convert to dictionary

## Usage Flow

1. **Initialize LLM**
   ```python
   from agenticpaygym.models.custom_llm import CustomLLM
   llm = CustomLLM(api_key="your-key", model="gpt-4")
   ```

2. **Create Agents**
   ```python
   from agenticpaygym.agents import BuyerAgent, SellerAgent
   buyer = BuyerAgent(llm=llm)
   seller = SellerAgent(llm=llm)
   ```

3. **Create Environment**
   ```python
   from agenticpaygym import NegotiationEnv
   env = NegotiationEnv(
       buyer_agent=buyer,
       seller_agent=seller,
       max_rounds=10,
       initial_seller_price=150.0,
       buyer_max_price=100.0,
   )
   ```

4. **Run Negotiation**
   ```python
   observation, info = env.reset(
       user_requirement="...",
       product_info={...}
   )
   
   while not done:
       action = agent.respond(...)
       observation, reward, terminated, truncated, info = env.step(action)
       done = terminated or truncated
   ```

## Extension Points

### Adding New LLM Providers

1. Create new class inheriting from `BaseLLM`
2. Implement `generate()` method
3. Export in `models/__init__.py`

### Creating Custom Agents

1. Inherit from `BaseAgent`
2. Implement `respond()` method
3. Optional: Override `_build_prompt()` to customize prompt

### Adding New Features

- Custom reward function: Modify `NegotiationEnv._calculate_reward()`
- Improve price extraction: Modify `NegotiationEnv._extract_price()`
- Multi-product support: Extend `NegotiationEnv` to support multi-product scenarios

## Design Principles

1. **Modularity**: Clear responsibilities for each module, low coupling
2. **Extensibility**: Easy to add new features and implementations
3. **Type Safety**: Use type hints to improve code quality
4. **Gymnasium Style**: API design references Gymnasium, easy to understand
5. **Complete Documentation**: Each module has clear documentation
