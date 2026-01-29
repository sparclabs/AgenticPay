"""VLM (Vision Language Model) Interface Abstract"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any
from pathlib import Path


class BaseVLM(ABC):
    """VLM Interface Base Class
    
    Defines a unified VLM interface, supporting different VLM implementations 
    (OpenAI GPT-4V, Anthropic Claude Vision, local vision models, etc.).
    
    VLM can process both text prompts and images to generate text responses.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, Path, List[Union[str, Path]], Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text from text prompt and optional images
        
        Args:
            prompt: Input text prompt
            images: Image input(s). Can be:
                - str: Path to image file or image URL
                - Path: Path object to image file
                - List[str/Path]: List of image paths/URLs
                - PIL.Image: PIL Image object
                - numpy.ndarray: Image as numpy array
                - bytes: Image as bytes (base64 encoded or raw)
                - List of any above types: Multiple images
                - None: No images (text-only generation)
            temperature: Temperature parameter, controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters (e.g., top_p, frequency_penalty, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If image format is not supported
            FileNotFoundError: If image file path does not exist
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of VLM"""
        pass
    
    def _validate_image_input(
        self,
        images: Optional[Union[str, Path, List[Union[str, Path]], Any]]
    ) -> bool:
        """Validate image input format
        
        Args:
            images: Image input to validate
            
        Returns:
            True if image input is valid, False otherwise
            
        Note:
            This is a helper method that can be overridden by subclasses
            to implement custom validation logic.
        """
        if images is None:
            return True
        
        # Check if it's a list
        if isinstance(images, list):
            return len(images) > 0
        
        # Check if it's a string path or URL
        if isinstance(images, (str, Path)):
            return True
        
        # Check if it's a PIL Image
        try:
            from PIL import Image
            if isinstance(images, Image.Image):
                return True
        except ImportError:
            pass
        
        # Check if it's a numpy array
        try:
            import numpy as np
            if isinstance(images, np.ndarray):
                return True
        except ImportError:
            pass
        
        # Check if it's bytes
        if isinstance(images, bytes):
            return True
        
        return False

