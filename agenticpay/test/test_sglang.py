"""Simple test script for SGLang using OpenAI-compatible API

This script demonstrates a simple single question-answer test using SGLang
with OpenAI-compatible client interface.
"""

import os
import openai
from sglang import Runtime

# Disable SGLang's CuDNN compatibility check (since vLLM works fine)
os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"


def test_sglang_simple():
    """Simple test: single question-answer using SGLang with OpenAI-compatible API"""
    
    print("=" * 60)
    print("SGLang Simple Test - Single Q&A (OpenAI-compatible API)")
    print("=" * 60)
    
    # Model path - adjust this to your local model path
    model_path = "/root/autodl-tmp/AgenticPayGym/agenticpay/models/download_models/Qwen3-VL-8B-Instruct"
    
    # Extract model name from model_path for API consistency
    model_name = os.path.basename(model_path)  # Gets "Qwen3-VL-8B-Instruct"
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"\n✗ Error: Model path not found: {model_path}")
        print("Please update the model_path variable to point to your local model directory.")
        return
    
    print(f"\nLoading model from: {model_path}")
    
    try:
        # 1. Initialize SGLang Runtime (this will start a local server)
        runtime = Runtime(model_path=model_path)
        print("✓ Model loaded successfully")
        
        # 2. Get server address (SGLang default port is 30000)
        # Try to get URL from runtime (may be endpoint object or url string)
        if hasattr(runtime, 'url'):
            base_url = runtime.url
        elif hasattr(runtime, 'endpoint'):
            endpoint = runtime.endpoint
            # RuntimeEndpoint object may have url attribute or can be converted to string
            if hasattr(endpoint, 'url'):
                base_url = endpoint.url
            else:
                base_url = str(endpoint)
        else:
            base_url = 'http://127.0.0.1:30000'
        
        # Ensure base_url is a string and in correct format
        if not isinstance(base_url, str):
            base_url = str(base_url)
        if not base_url.startswith("http"):
            base_url = f"http://127.0.0.1:{base_url}"
        
        print(f"✓ Server endpoint: {base_url}")
        
        # 3. Create OpenAI-compatible client (connects to SGLang server via base_url)
        client = openai.OpenAI(
            base_url=f"{base_url}/v1",
            api_key="None"  # SGLang doesn't require API key
        )
        print("✓ Client connected successfully")
        
        # 4. Use OpenAI API (actually connects to SGLang server)
        print("\nTesting single question-answer...")
        question = "你好，请介绍一下你自己。"
        print(f"Question: {question}")
        
        response = client.chat.completions.create(
            model=model_name,  # Use model name from local path for consistency
            messages=[
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        
        # 5. Extract result
        answer = response.choices[0].message.content
        print(f"Answer: {answer}")
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            if 'runtime' in locals() and hasattr(runtime, 'shutdown'):
                runtime.shutdown()
        except:
            pass


if __name__ == "__main__":
    test_sglang_simple()
