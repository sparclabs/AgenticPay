"""Custom LLM Implementation"""

from typing import Optional
import os
import http.client
import json
from agenticpaygym.models.base_llm import BaseLLM


class CustomLLM(BaseLLM):
    """Custom LLM Implementation
    
    Uses custom API for text generation.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize Custom LLM
        
        Args:
            model: Model name, e.g., "gpt-4", "gpt-3.5-turbo"
            api_key: API key, if not provided, will be obtained from environment variable
            base_url: API base URL (for compatibility with OpenAI-format APIs)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "API key is required. "
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
    
    def model_api(self, input_str: str, key: str, temperature: float = 0.7, max_tokens: int = 4096, model_name: str = "gpt-4o-mini-2024-07-18") -> str:
        conn = http.client.HTTPSConnection("api2.aigcbest.top")
        payload = json.dumps({
           "model": model_name,
           "messages": [
              {
                 "role": "user",
                 "content": input_str
              }
           ],
            "temperature": 0,
            "max_tokens": max_tokens
        
        })
        headers = {
           'Accept': 'application/json',
           'Authorization': f'Bearer {key}',
           'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        # print(data.decode("utf-8"))
        data = json.loads(data.decode("utf-8"))
        # logger.info(f'Bearer {key}')
        # print(data)
    
        return data["choices"][0]["message"]["content"]

    
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
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Returns:
            Generated text
        """
        try:
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     **kwargs,
            # )
            # return response.choices[0].message.content.strip()

            response = self.model_api(prompt, self.api_key, temperature, max_tokens, self.model)
            return response

        except Exception as e:
            raise RuntimeError(f"Custom API error: {e}")
    
    def __repr__(self) -> str:
        """Return string representation of LLM"""
        return f"CustomLLM(model={self.model})"


if __name__ == "__main__":
    """Test Custom LLM API calls"""
    import sys
    
    print("=" * 50)
    print("Testing Custom LLM API calls")
    print("=" * 50)
    
    try:
        # Initialize LLM
        print("\n1. Initializing Custom LLM...")
        llm = CustomLLM(model="gpt-3.5-turbo", api_key="", base_url="https://api2.aigcbest.top")
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
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ API call error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

