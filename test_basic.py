"""Basic functionality test script (no LLM API required)"""

import sys
import os

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all module imports"""
    print("Testing imports...")
    
    try:
        from agenticpaygym import (
            NegotiationEnv,
            NegotiationStatus,
            NegotiationInfo,
            BaseAgent,
            BuyerAgent,
            SellerAgent,
            ConversationMemory,
            BaseLLM,
        )
        print("✓ Core imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    try:
        from agenticpaygym.core import NegotiationEnv, NegotiationStatus
        print("✓ Core module import successful")
    except ImportError as e:
        print(f"✗ Core module import failed: {e}")
        return False
    
    try:
        from agenticpaygym.agents import BaseAgent, BuyerAgent, SellerAgent
        print("✓ Agents module import successful")
    except ImportError as e:
        print(f"✗ Agents module import failed: {e}")
        return False
    
    try:
        from agenticpaygym.memory import ConversationMemory
        print("✓ Memory module import successful")
    except ImportError as e:
        print(f"✗ Memory module import failed: {e}")
        return False
    
    try:
        from agenticpaygym.models import BaseLLM
        print("✓ LLM module import successful")
    except ImportError as e:
        print(f"✗ LLM module import failed: {e}")
        return False
    
    return True


def test_memory():
    """Test Memory system"""
    print("\nTesting ConversationMemory...")
    
    from agenticpaygym.memory import ConversationMemory
    
    memory = ConversationMemory(max_length=10)
    
    # Add messages
    memory.add_message("buyer", "Hello, I'm interested in this product.", 0)
    memory.add_message("seller", "Great! The price is $100.", 0)
    memory.add_message("buyer", "Can you do $80?", 1)
    
    # Test getting history
    history = memory.get_history()
    assert len(history) == 3, f"Expected 3 messages, got {len(history)}"
    print("✓ Memory add/get operations work")
    
    # Test getting by role
    buyer_msgs = memory.get_history_by_role("buyer")
    assert len(buyer_msgs) == 2, f"Expected 2 buyer messages, got {len(buyer_msgs)}"
    print("✓ Memory role filtering works")
    
    # Test clearing
    memory.clear()
    assert len(memory) == 0, "Memory should be empty after clear"
    print("✓ Memory clear works")
    
    return True


def test_negotiation_state():
    """Test negotiation state"""
    print("\nTesting NegotiationState...")
    
    from agenticpaygym.utils import NegotiationState
    
    state = NegotiationState()
    state.update(round=1, seller_price=100.0, buyer_price=80.0)
    
    assert state.round == 1
    assert state.seller_price == 100.0
    assert state.buyer_price == 80.0
    print("✓ NegotiationState update works")
    
    state_dict = state.to_dict()
    assert "round" in state_dict
    assert "seller_price" in state_dict
    print("✓ NegotiationState to_dict works")
    
    return True


def test_core_env_structure():
    """Test core environment structure (without running actual negotiation)"""
    print("\nTesting NegotiationEnv structure...")
    
    from agenticpaygym.core import NegotiationEnv, NegotiationStatus
    from agenticpaygym.agents.base_agent import BaseAgent
    from agenticpaygym.models.base_llm import BaseLLM
    
    # Create a simple Mock LLM
    class MockLLM(BaseLLM):
        def generate(self, prompt, **kwargs):
            return "Mock response"
        
        def __repr__(self):
            return "MockLLM()"
    
    # Create Mock Agents
    class MockAgent(BaseAgent):
        def respond(self, conversation_history, current_state):
            return "Mock agent response"
    
    mock_llm = MockLLM()
    buyer = MockAgent(mock_llm, "buyer", "a buyer")
    seller = MockAgent(mock_llm, "seller", "a seller")
    
    # Create environment
    env = NegotiationEnv(
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=5,
        initial_seller_price=100.0,
        buyer_max_price=80.0,
    )
    
    assert env.max_rounds == 5
    assert env.initial_seller_price == 100.0
    assert env.buyer_max_price == 80.0
    print("✓ NegotiationEnv initialization works")
    
    # Test reset (without actually running)
    try:
        observation, info = env.reset(
            user_requirement="Test requirement",
            product_info={"name": "Test Product"}
        )
        assert "conversation_history" in observation
        assert "current_round" in observation
        print("✓ NegotiationEnv reset structure works")
    except Exception as e:
        print(f"✗ NegotiationEnv reset failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("AgenticPayGym Basic Tests")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Memory System", test_memory),
        ("Negotiation State", test_negotiation_state),
        ("Core Environment", test_core_env_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

