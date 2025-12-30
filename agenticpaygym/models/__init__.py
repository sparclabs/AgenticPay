"""Models module"""

from agenticpaygym.models.base_llm import BaseLLM
from agenticpaygym.models.base_vlm import BaseVLM

# Try to import OpenAI LLM implementation, but don't fail if not available
try:
    from agenticpaygym.models.openai_llm import OpenAILLM
    __all__ = ["BaseLLM", "BaseVLM", "OpenAILLM"]
except ImportError:
    __all__ = ["BaseLLM", "BaseVLM"]

# Try to import OpenAI VLM implementation, but don't fail if not available
try:
    from agenticpaygym.models.openai_vlm import OpenAIVLM
    if len(__all__) > 2:
        __all__.append("OpenAIVLM")
    else:
        __all__ = ["BaseLLM", "BaseVLM", "OpenAIVLM"]
except ImportError:
    pass  # OpenAIVLM is optional

# Try to import Custom LLM implementation, but don't fail if not available
try:
    from agenticpaygym.models.custom_llm import CustomLLM
    if "OpenAILLM" in __all__:
        __all__.append("CustomLLM")
    else:
        __all__ = ["BaseLLM", "BaseVLM", "CustomLLM"]
except ImportError:
    pass  # CustomLLM is optional

# Try to import Qwen3 VL implementation, but don't fail if not available
try:
    from agenticpaygym.models.qwen3_vl import Qwen3VL
    if len(__all__) > 2:
        __all__.append("Qwen3VL")
    else:
        __all__ = ["BaseLLM", "BaseVLM", "Qwen3VL"]
except ImportError:
    pass  # Qwen3VL is optional

# Try to import vLLM VLM implementation, but don't fail if not available
try:
    from agenticpaygym.models.vllm_vlm import VLLMVLM
    if len(__all__) > 2:
        __all__.append("VLLMVLM")
    else:
        __all__ = ["BaseLLM", "BaseVLM", "VLLMVLM"]
except ImportError:
    pass  # VLLMVLM is optional
