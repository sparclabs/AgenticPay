"""OpenAI LLM Implementation"""

from typing import Optional
import os
from agenticpaygym.models.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI LLM Implementation
    
    Uses official OpenAI API for text generation.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize OpenAI LLM
        
        Args:
            model: Model name, e.g., "gpt-4", "gpt-3.5-turbo", "gpt-4o"
            api_key: API key, if not provided, will be obtained from environment variable
            base_url: API base URL (optional, defaults to official OpenAI API)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set it via api_key parameter or OPENAI_API_KEY environment variable."
            )
        
        # Lazy import openai to avoid errors when not installed
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install it with: pip install openai"
            )
    
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
            temperature: Temperature parameter (controls randomness, 0-2)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters (e.g., top_p, frequency_penalty, presence_penalty, etc.)
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def __repr__(self) -> str:
        """Return string representation of LLM"""
        return f"OpenAILLM(model={self.model})"


if __name__ == "__main__":
    """Test OpenAI LLM API calls"""
    import sys
    
    print("=" * 50)
    print("Testing OpenAI LLM API calls")
    print("=" * 50)
    
    try:
        # Initialize LLM
        print("\n1. Initializing OpenAI LLM...")
        llm = OpenAILLM(model="gpt-3.5-turbo")
        print(f"✓ Successfully initialized: {llm}")
        
        # Test generation
        print("\n2. Testing text generation...")
        test_prompt = "Please introduce artificial intelligence in one sentence."
        print(f"Input prompt: {test_prompt}")
        
        response = llm.generate(
            prompt=test_prompt,
            temperature=0.7,
            max_tokens=1024
        )
        
        print(f"\n✓ Generation successful!")
        print(f"Response content: {response}")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("Hint: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Hint: Install OpenAI package with: pip install openai")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ API call error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

