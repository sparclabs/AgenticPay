"""Environment Registration System Usage Example

Demonstrates how to use the new environment registration system to create and manage environments.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agenticpaygym import make, register, spec, pprint_registry
from agenticpaygym.envs.single_buyer_product_seller.negotiation_env import NegotiationEnv
from agenticpaygym.agents.buyer_agent import BuyerAgent
from agenticpaygym.agents.seller_agent import SellerAgent
from agenticpaygym.llm.openai_llm import OpenAILLM


def example_basic_usage():
    """Example 1: Basic Usage - Using Registered Environments"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # View all registered environments
    print("\nRegistered environments:")
    pprint_registry()
    
    # Use make() to create environment
    # Note: Required parameters (buyer_agent, seller_agent, etc.) need to be provided here
    print("\nUsing make() to create environment:")
    print("env = make('Negotiation-v0', buyer_agent=..., seller_agent=...)")
    

def example_custom_registration():
    """Example 2: Register Custom Environment"""
    print("\n" + "=" * 60)
    print("Example 2: Register Custom Environment")
    print("=" * 60)
    
    # Method 1: Register using class object
    register(
        id="my_custom/Negotiation-v1",
        entry_point=NegotiationEnv,
        max_episode_steps=30,
        kwargs={"price_tolerance": 2.0},  # Default parameters
    )
    
    # Method 2: Register using string path
    register(
        id="my_custom/Negotiation-v2",
        entry_point="agenticpaygym.envs.negotiation_env:NegotiationEnv",
        max_episode_steps=25,
    )
    
    print("\nEnvironment list after registration:")
    pprint_registry()
    
    # Get environment specification
    env_spec = spec("my_custom/Negotiation-v1")
    print(f"\nEnvironment specification: {env_spec.id}")
    print(f"Max episode steps: {env_spec.max_episode_steps}")
    print(f"Default parameters: {env_spec.kwargs}")


def example_namespace():
    """Example 3: Using Namespace"""
    print("\n" + "=" * 60)
    print("Example 3: Using Namespace")
    print("=" * 60)
    
    from agenticpaygym.envs.registration import namespace
    
    # Use namespace context manager
    with namespace("my_company"):
        register(
            id="ProductNegotiation-v0",
            entry_point=NegotiationEnv,
        )
        # The actual registered ID is "my_company/ProductNegotiation-v0"
    
    print("\nEnvironment list after using namespace:")
    pprint_registry()


def example_version_management():
    """Example 4: Version Management"""
    print("\n" + "=" * 60)
    print("Example 4: Version Management")
    print("=" * 60)
    
    # Register multiple versions
    register(id="VersionedEnv-v1", entry_point=NegotiationEnv)
    register(id="VersionedEnv-v2", entry_point=NegotiationEnv)
    register(id="VersionedEnv-v3", entry_point=NegotiationEnv)
    
    # When creating environment, if version is not specified, the latest version will be used automatically
    print("\nEnvironments with multiple versions registered:")
    print("- VersionedEnv-v1")
    print("- VersionedEnv-v2")
    print("- VersionedEnv-v3")
    print("\nUsing make('VersionedEnv') will automatically use v3")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("AgenticPayGym Environment Registration System Examples")
    print("=" * 60)
    
    example_basic_usage()
    example_custom_registration()
    example_namespace()
    example_version_management()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

