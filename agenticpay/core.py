"""Core Environment Base Class

Provides base abstract class and common type definitions for all environments.
All concrete environments should inherit from BaseEnv and implement abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class NegotiationStatus(Enum):
    """Negotiation status enumeration"""
    ONGOING = "ongoing"
    AGREED = "agreed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class NegotiationInfo:
    """Negotiation information dataclass
    
    Used to store state information during the negotiation process.
    """
    current_price: Optional[float] = None
    buyer_price: Optional[float] = None
    seller_price: Optional[float] = None
    round_count: int = 0
    status: NegotiationStatus = NegotiationStatus.ONGOING
    conversation_history: list = field(default_factory=list)


class BaseEnv(ABC):
    """Multi-agent interaction environment base class
    
    References Gymnasium's design pattern, provides standard reset and step interfaces.
    All concrete environments should inherit from this class and implement abstract methods.
    
    Subclasses need to implement:
    - reset(): Reset environment, start new interaction
    - step(): Execute one interaction step
    
    Optional implementations:
    - render(): Render current state
    - close(): Close environment, cleanup resources
    """
    
    def __init__(self):
        """Initialize environment
        
        Subclasses can override this method to add custom initialization logic.
        """
        # Used to store environment specification (set by registration system)
        self.spec = None
        self.unwrapped = self
    
    @abstractmethod
    def reset(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new interaction
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            (observation, info) Initial observation and info
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one interaction step
        
        Args:
            action: Current agent's action
            
        Returns:
            (observation, reward, terminated, truncated, info)
            - observation: New observation state
            - reward: Reward value
            - terminated: Whether normally terminated
            - truncated: Whether truncated (e.g., reached max steps)
            - info: Additional information dictionary
        """
        pass
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        pass
    
    def close(self):
        """Close environment, cleanup resources
        
        Subclasses can override this method to add cleanup logic.
        """
        pass

