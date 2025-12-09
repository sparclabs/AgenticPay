"""User profile data structure"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class StylePreference(Enum):
    """Style preference enumeration"""
    SIMPLE = "Simple"  # Simple style
    BUSINESS = "Business"  # Business style
    TRADITIONAL = "Traditional"  # Traditional style


class ShoppingHabit(Enum):
    """Shopping habit enumeration"""
    COMPARE = "Compare prices"  # Likes to compare prices
    DIRECT = "Direct purchase"  # Likes to buy directly


@dataclass
class UserProfile:
    """User profile dataclass
    
    Contains user's personal preference information, used to influence Buyer Agent's negotiation behavior.
    """
    
    style_preference: Optional[StylePreference] = None
    """Style preference: Simple, Business, or Traditional"""
    
    shopping_habit: Optional[ShoppingHabit] = None
    """Shopping habit: Likes to compare prices or likes to buy directly"""
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "style_preference": self.style_preference.value if self.style_preference else None,
            "shopping_habit": self.shopping_habit.value if self.shopping_habit else None,
        }
    
    def get_description(self) -> str:
        """Get text description of user preferences"""
        descriptions = []
        
        if self.style_preference:
            descriptions.append(f"Style preference: {self.style_preference.value}")
        
        if self.shopping_habit:
            descriptions.append(f"Shopping habit: {self.shopping_habit.value}")
        
        if descriptions:
            return "; ".join(descriptions)
        return "No special preferences"

