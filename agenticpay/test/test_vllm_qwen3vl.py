"""Test script for Qwen2.5-VL-2B-Instruct using vLLM

This script demonstrates how to use vLLM to load and run Qwen2.5-VL-2B-Instruct model
from Hugging Face for text and vision-language tasks.
"""

import os
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal.utils import encode_image_base64
except ImportError:
    print("Error: vLLM is not installed.")
    print("Please install vLLM with: pip install vllm")
    sys.exit(1)


def test_vllm_qwen3vl():
    """Test Qwen2.5-VL-2B-Instruct model using vLLM"""
    
    print("=" * 60)
    print("Testing Qwen2.5-VL-2B-Instruct with vLLM")
    print("=" * 60)
    
    # Model path (local)
    model_id = "/root/autodl-tmp/AgenticPayGym/agenticpay/models/download_models/Qwen3-VL-2B-Instruct"
    
    print(f"\n1. Loading model from local path: {model_id}")
    print("   Loading local model...")
    
    try:
        # Initialize vLLM with Qwen2.5-VL model from Hugging Face
        llm = LLM(
            model=model_id,
            trust_remote_code=True,  # Required for Qwen models
            # Increase GPU memory utilization to use more available memory
            gpu_memory_utilization=0.9,
            # Reduce max_model_len to fit available KV cache memory
            # 8192 should be sufficient for most use cases
            # max_model_len=8192,
            # Optional: specify tensor parallel size for multi-GPU
            tensor_parallel_size=2,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # Test 1: Text-only generation
    print("\n2. Testing text-only generation...")
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
    )
    
    text_prompt = "你好，请介绍一下你自己。你是什么模型？"
    
    print(f"   Prompt: {text_prompt}")
    print("   Generating response...")
    
    try:
        outputs = llm.generate([text_prompt], sampling_params)
        response = outputs[0].outputs[0].text
        print(f"   Response: {response}")
        print("✓ Text-only generation test completed")
    except Exception as e:
        print(f"✗ Error in text generation: {e}")
    
    # Test 2: Vision-language generation with local image
    print("\n3. Testing vision-language generation with local image...")
    
    # # Get test image path
    # test_image_path = Path(__file__).parent / "image.png"
    
    # if not test_image_path.exists():
    #     print(f"   Warning: Test image not found at {test_image_path}")
    #     print("   Skipping vision-language test")
    # else:
    #     print(f"   Using test image: {test_image_path}")
        
    #     try:
    #         # Encode image to base64
    #         image_base64 = encode_image_base64(str(test_image_path))
            
    #         # Prepare multimodal prompt
    #         # Qwen2.5-VL uses a specific format for multimodal inputs
    #         vision_prompt = "请详细描述这张图片的内容。"
            
    #         # For Qwen2.5-VL, we need to format the input correctly
    #         # The format depends on the model's chat template
    #         # Let's try a simple approach first
    #         multimodal_input = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "image",
    #                         "image": image_base64,
    #                     },
    #                     {
    #                         "type": "text",
    #                         "text": vision_prompt,
    #                     }
    #                 ]
    #             }
    #         ]
            
    #         print(f"   Prompt: {vision_prompt}")
    #         print("   Generating response...")
            
        #         # Note: vLLM's multimodal support may vary by model
        #         # For Qwen2.5-VL, we might need to use a different approach
    #         outputs = llm.generate(multimodal_input, sampling_params)
    #         response = outputs[0].outputs[0].text
    #         print(f"   Response: {response}")
    #         print("✓ Vision-language generation test completed")
            
    #     except Exception as e:
    #         print(f"✗ Error in vision-language generation: {e}")
        #         print("   Note: vLLM's multimodal support for Qwen2.5-VL may require specific configuration")
        #         print("   You may need to check vLLM documentation for Qwen2.5-VL specific usage")
    
    # # Test 3: Multiple prompts
    # print("\n4. Testing batch generation with multiple prompts...")
    
    # prompts = [
    #     "什么是人工智能？",
    #     "请解释一下机器学习的基本概念。",
    #     "深度学习有哪些应用？",
    # ]
    
    # print(f"   Processing {len(prompts)} prompts in batch...")
    
    # try:
    #     outputs = llm.generate(prompts, sampling_params)
        
    #     for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
    #         response = output.outputs[0].text
    #         print(f"\n   Prompt {i}: {prompt}")
    #         print(f"   Response {i}: {response[:100]}..." if len(response) > 100 else f"   Response {i}: {response}")
        
    #     print("✓ Batch generation test completed")
    # except Exception as e:
    #     print(f"✗ Error in batch generation: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_vllm_qwen3vl()

