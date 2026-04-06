"""AgenticPayGym: A Multi-Agent Negotiation Framework for Buyer-Seller Transactions."""

from agenticpay.core import NegotiationStatus, NegotiationInfo, BaseEnv
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.agents.multi_param_buyer_agent import MultiParamBuyerAgent
from agenticpay.agents.multi_param_seller_agent import MultiParamSellerAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.models.base_llm import BaseLLM

# Import environment registration system
from agenticpay.envs import (
    register,
    make,
    spec,
    pprint_registry,
    registry,
    EnvSpec,
    Task1BasicPriceNegotiation,  # Backward compatibility
    Task2ClosePriceNegotiation,  # Backward compatibility
    Task3CloseToMarketPriceNegotiation,  # Backward compatibility
    Task1PriceQualityNegotiation,
    Task2PriceQualityDeliveryNegotiation,
    Task3PriceQualityDeliveryWarrantyNegotiation,
    Task4AllParamsNegotiation,
)
from agenticpay.envs.multi_param.base_multi_param_negotiation import ParamPreferences, MultiParamOffer

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
    "MultiParamBuyerAgent",
    "MultiParamSellerAgent",
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
    # Multi-parameter negotiation
    "Task1PriceQualityNegotiation",
    "Task2PriceQualityDeliveryNegotiation",
    "Task3PriceQualityDeliveryWarrantyNegotiation",
    "Task4AllParamsNegotiation",
    "ParamPreferences",
    "MultiParamOffer",
]

# Try to import OpenAI LLM if available
try:
    from agenticpay.models.openai_llm import OpenAILLM
    __all__.append("OpenAILLM")
except ImportError:
    pass

# Try to import Custom LLM if available
try:
    from agenticpay.models.custom_llm import CustomLLM
    __all__.append("CustomLLM")
except ImportError:
    pass
