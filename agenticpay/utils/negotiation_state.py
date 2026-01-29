"""Negotiation State Management"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class NegotiationState:
    """Negotiation state dataclass"""
    
    round: int = 0
    seller_price: Optional[float] = None
    buyer_price: Optional[float] = None
    agreed_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, **kwargs):
        """Update state"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "round": self.round,
            "seller_price": self.seller_price,
            "buyer_price": self.buyer_price,
            "agreed_price": self.agreed_price,
            "metadata": self.metadata,
        }

