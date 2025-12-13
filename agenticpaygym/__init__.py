"""AgenticPayGym: A Multi-Agent Negotiation Framework for Buyer-Seller Transactions."""

from agenticpaygym.core import NegotiationStatus, NegotiationInfo, BaseEnv
from agenticpaygym.agents.base_agent import BaseAgent
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.memory.conversation_memory import ConversationMemory
from agenticpaygym.llm.base_llm import BaseLLM

# Import environment registration system
from agenticpaygym.envs import (
    register,
    make,
    spec,
    pprint_registry,
    registry,
    EnvSpec,
    Task1BasicPriceNegotiation,  # Backward compatibility
    Task2ClosePriceNegotiation,  # Backward compatibility
    Task3CloseToMarketPriceNegotiation,  # Backward compatibility
)

__version__ = "0.1.0"

__all__ = [
    # Core types
    "BaseEnv",
    "NegotiationStatus",
    "NegotiationInfo",
    # Agents
    "BaseAgent",
    "BuyerAgent",
    "SellerAgent",
    # Memory
    "ConversationMemory",
    # LLM
    "BaseLLM",
    # Environment registration system
    "register",
    "make",
    "spec",
    "pprint_registry",
    "registry",
    "EnvSpec",
    # Environment classes (backward compatibility)
    "Task1BasicPriceNegotiation",
    "Task2ClosePriceNegotiation",
    "Task3CloseToMarketPriceNegotiation",
]

# Try to import OpenAI LLM if available
try:
    from agenticpaygym.llm.openai_llm import OpenAILLM
    __all__.append("OpenAILLM")
except ImportError:
    pass
