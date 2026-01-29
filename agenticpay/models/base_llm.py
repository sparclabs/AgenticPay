"""LLM Interface Abstract"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLM(ABC):
    """LLM Interface Base Class
    
    Defines a unified LLM interface, supporting different LLM implementations (OpenAI, Anthropic, local models, etc.).
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text
        
        Args:
            prompt: Input prompt
            temperature: Temperature parameter, controls randomness (0-1)
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of LLM"""
        pass

