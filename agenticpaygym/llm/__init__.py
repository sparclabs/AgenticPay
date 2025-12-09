"""LLM module"""

from agenticpaygym.llm.base_llm import BaseLLM

# Try to import OpenAI implementation, but don't fail if not available
try:
    from agenticpaygym.llm.openai_llm import OpenAILLM
    __all__ = ["BaseLLM", "OpenAILLM"]
except ImportError:
    __all__ = ["BaseLLM"]
