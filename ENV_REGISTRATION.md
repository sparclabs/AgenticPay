# Environment Registration System Usage Guide

AgenticPayGym now supports a Gymnasium-like environment registration system for quick registration and use of new tasks.

## Directory Structure

```
agenticpaygym/
├── core.py                    # Core abstract base class BaseEnv
├── envs/                      # Environment directory
│   ├── __init__.py           # Environment registration module
│   ├── registration.py       # Registration system implementation
│   └── negotiation_env.py    # Negotiation environment implementation (example)
└── examples/                 # Example files
    ├── simple_negotiation.py  # Basic usage example
    └── registration_example.py # Registration system example
```

## Quick Start

### 1. Using Registered Environments

```python
from agenticpaygym import make
from agenticpaygym.agents import BuyerAgent, SellerAgent

# Create agents
buyer = BuyerAgent(llm=llm, buyer_max_price=120.0)
seller = SellerAgent(llm=llm, seller_min_price=80.0)

# Create environment using registration system
env = make(
    "Negotiation-v0",
    buyer_agent=buyer,
    seller_agent=seller,
    max_rounds=20,
    initial_seller_price=150.0,
)
```

### 2. Registering New Environments

#### Method 1: Using Class Object

```python
from agenticpaygym.envs import register
from agenticpaygym.envs.negotiation_env import NegotiationEnv

register(
    id="my_env/CustomNegotiation-v0",
    entry_point=NegotiationEnv,
    max_episode_steps=30,
    kwargs={"price_tolerance": 2.0},  # Default parameters
)
```

#### Method 2: Using String Path

```python
register(
    id="my_env/CustomNegotiation-v1",
    entry_point="agenticpaygym.envs.negotiation_env:NegotiationEnv",
    max_episode_steps=25,
)
```

### 3. Using Namespaces

```python
from agenticpaygym.envs.registration import namespace

with namespace("my_company"):
    register(
        id="ProductNegotiation-v0",
        entry_point=NegotiationEnv,
    )
    # The actual registered ID is "my_company/ProductNegotiation-v0"
```

### 4. Version Management

```python
# Register multiple versions
register(id="MyEnv-v1", entry_point=MyEnv)
register(id="MyEnv-v2", entry_point=MyEnv)
register(id="MyEnv-v3", entry_point=MyEnv)

# Use latest version (automatically selects v3)
env = make("MyEnv")

# Use specific version
env = make("MyEnv-v2")
```

## Creating Custom Environments

### 1. Inherit from BaseEnv

```python
from agenticpaygym.core import BaseEnv
from typing import Dict, Any, Tuple

class MyCustomEnv(BaseEnv):
    def __init__(self, param1: str, param2: int = 10):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Implement reset logic
        observation = {"state": "initial"}
        info = {}
        return observation, info
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # Implement step logic
        observation = {"state": "updated"}
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
```

### 2. Register Environment

Add to `agenticpaygym/envs/__init__.py`:

```python
from agenticpaygym.envs.my_custom_env import MyCustomEnv

register(
    id="MyCustomEnv-v0",
    entry_point="agenticpaygym.envs.my_custom_env:MyCustomEnv",
    max_episode_steps=100,
    kwargs={"param2": 20},  # Default parameters
)
```

### 3. Use Environment

```python
from agenticpaygym import make

env = make("MyCustomEnv-v0", param1="value")
```

## API Reference

### register()

Register a new environment.

```python
register(
    id: str,                    # Environment ID: [namespace/](env_name)[-v(version)]
    entry_point: EnvCreator | str | None,  # Environment class or string path
    max_episode_steps: int | None = None,  # Maximum steps
    kwargs: dict | None = None,  # Default parameters
)
```

### make()

Create environment instance.

```python
env = make(
    id: str | EnvSpec,  # Environment ID or specification
    max_episode_steps: int | None = None,  # Override maximum steps
    **kwargs: Any,  # Parameters passed to environment
)
```

### spec()

Get environment specification.

```python
env_spec = spec("Negotiation-v0")
print(env_spec.id)
print(env_spec.max_episode_steps)
```

### pprint_registry()

Print all registered environments.

```python
from agenticpaygym import pprint_registry

pprint_registry()
pprint_registry(exclude_namespaces=["ALE"])  # Exclude specific namespaces
```

## Environment ID Format

Environment IDs follow this format:

```
[namespace/](env_name)[-v(version)]
```

Examples:
- `Negotiation-v0` - No namespace, version 0
- `my_env/Negotiation-v1` - With namespace, version 1
- `Negotiation` - No version number

## Backward Compatibility

For backward compatibility, `NegotiationEnv` can still be directly imported and used:

```python
from agenticpaygym import NegotiationEnv

env = NegotiationEnv(
    buyer_agent=buyer,
    seller_agent=seller,
    max_rounds=20,
)
```

But using the registration system is recommended:

```python
from agenticpaygym import make

env = make("Negotiation-v0", buyer_agent=buyer, seller_agent=seller, max_rounds=20)
```

## Examples

See the following files for more examples:

- `examples/simple_negotiation.py` - Basic usage example
- `examples/registration_example.py` - Detailed registration system example
