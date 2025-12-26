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

# Try to import Hugging Face LLM implementation, but don't fail if not available
try:
    from agenticpaygym.models.huggingface_llm import HuggingFaceLLM
    if len(__all__) > 2:  # At least BaseLLM and BaseVLM
        __all__.append("HuggingFaceLLM")
    else:
        __all__ = ["BaseLLM", "BaseVLM", "HuggingFaceLLM"]
except ImportError:
    pass  # HuggingFaceLLM is optional

# Try to import Hugging Face VLM implementation, but don't fail if not available
try:
    from agenticpaygym.models.huggingface_vlm import HuggingFaceVLM
    if len(__all__) > 2:
        __all__.append("HuggingFaceVLM")
    else:
        __all__ = ["BaseLLM", "BaseVLM", "HuggingFaceVLM"]
except ImportError:
    pass  # HuggingFaceVLM is optional
