"""Conversation Memory System"""

from typing import List, Dict, Any, Optional


class ConversationMemory:
    """Conversation Memory Management
    
    Manages multi-turn conversation history, supports context retrieval and summarization.
    """
    
    def __init__(self, max_length: Optional[int] = None):
        """Initialize memory
        
        Args:
            max_length: Maximum memory length, None means unlimited
        """
        self.messages: List[Dict[str, Any]] = []
        self.max_length = max_length
    
    def add_message(
        self,
        role: str,
        content: str,
        round: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add message to memory
        
        Args:
            role: Speaker role ("buyer" or "seller")
            content: Message content
            round: Round number
            metadata: Additional metadata
        """
        message = {
            "role": role,
            "content": content,
            "round": round,
            "metadata": metadata or {},
        }
        self.messages.append(message)
        
        # If exceeds maximum length, remove oldest messages
        if self.max_length and len(self.messages) > self.max_length:
            self.messages = self.messages[-self.max_length:]
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history
        
        Args:
            last_n: Only return last n messages, None means return all
            
        Returns:
            Conversation history list
        """
        if last_n is None:
            return self.messages.copy()
        return self.messages[-last_n:] if last_n > 0 else []
    
    def get_history_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get messages from specific role
        
        Args:
            role: Role name
            
        Returns:
            List of messages from that role
        """
        return [msg for msg in self.messages if msg.get("role") == role]
    
    def clear(self):
        """Clear memory"""
        self.messages = []
    
    def get_summary(self) -> str:
        """Get conversation summary
        
        Returns:
            Conversation summary string
        """
        total_messages = len(self.messages)
        buyer_messages = len(self.get_history_by_role("buyer"))
        seller_messages = len(self.get_history_by_role("seller"))
        
        return (
            f"Total messages: {total_messages}, "
            f"Buyer: {buyer_messages}, "
            f"Seller: {seller_messages}"
        )
    
    def __len__(self) -> int:
        """Return number of messages"""
        return len(self.messages)

