"""vLLM LLM (Large Language Model) Implementation

This module provides a LLM implementation using vLLM for efficient inference.
Supports text-only generation (no vision/multimodal support).
"""

from typing import Optional
from pathlib import Path
from agenticpaygym.models.base_llm import BaseLLM


class VLLMLLM(BaseLLM):
    """vLLM LLM Implementation
    
    Uses vLLM for efficient language model inference.
    Supports loading models from Hugging Face Hub or local path.
    Text-only generation (no multimodal/vision support).
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        **llm_kwargs,
    ):
        """Initialize vLLM LLM Model
        
        Args:
            model_id: Hugging Face Hub model ID (e.g., "Qwen/Qwen3-8B-Instruct")
            model_path: Local model path (e.g., "/path/to/local/model")
            trust_remote_code: Whether to trust remote code (required for some models)
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum model length (KV cache size)
            **llm_kwargs: Additional parameters passed to vLLM LLM initialization
            
        Note:
            Either model_id or model_path must be provided.
            If both are provided, model_path takes priority.
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM package is required. Install it with: pip install vllm"
            )
        
        if not model_id and not model_path:
            raise ValueError(
                "Either model_id (Hugging Face Hub) or model_path (local) must be provided."
            )
        
        self.model_id = model_id
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.llm_kwargs = llm_kwargs
        
        # Lazy loading: model is loaded on first call
        self._llm = None
    
    def _load_model(self):
        """Load vLLM model"""
        if self._llm is not None:
            return
        
        from vllm import LLM
        
        model_name_or_path = self.model_path or self.model_id
        
        print(f"Loading vLLM LLM model from: {model_name_or_path}")
        print(f"GPU memory utilization: {self.gpu_memory_utilization}")
        print(f"Tensor parallel size: {self.tensor_parallel_size}")
        if self.max_model_len:
            print(f"Max model length: {self.max_model_len}")
        
        # Prepare LLM initialization parameters
        llm_params = {
            "model": model_name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            **self.llm_kwargs
        }
        
        if self.max_model_len is not None:
            llm_params["max_model_len"] = self.max_model_len
        
        # Initialize vLLM
        self._llm = LLM(**llm_params)
        print("✓ vLLM LLM model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        seed: int = 0,
        **kwargs,
    ) -> str:
        """Generate text from text prompt using vLLM
        
        Args:
            prompt: Input text prompt
            temperature: Temperature parameter (controls randomness, 0.0-2.0), default 0.0 for deterministic
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            seed: Random seed for reproducibility (default 0)
            **kwargs: Other parameters passed to SamplingParams
            
        Returns:
            Generated text response
        """
        from vllm import SamplingParams
        
        self._load_model()
        
        try:
            # Prepare sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens or 1024,
                top_p=top_p,
                seed=seed,
                **kwargs
            )
            
            # Generate text
            outputs = self._llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            return response.strip()
                
        except Exception as e:
            raise RuntimeError(f"vLLM generation error: {e}")
    
    def __repr__(self) -> str:
        """Return string representation of LLM"""
        model_source = self.model_path if self.model_path else self.model_id
        return f"VLLMLLM(model={model_source}, gpu_memory_utilization={self.gpu_memory_utilization}, tensor_parallel_size={self.tensor_parallel_size})"


if __name__ == "__main__":
    """Test vLLM LLM Model"""
    import sys
    
    print("=" * 50)
    print("Testing vLLM LLM Model")
    print("=" * 50)
    
    try:
        # Test loading model
        print("\n1. Testing loading model...")
        print("Note: This will load the model from local path.")
        print("Note: GPU is required for vLLM.")
        
        # Use Qwen3 model (text-only)
        # Build relative path starting from AgenticPayGym
        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        # Search up the directory tree to find AgenticPayGym directory
        while project_root.name != "AgenticPayGym" and project_root.parent != project_root:
            project_root = project_root.parent
        
        # Build path from AgenticPayGym root
        if project_root.name == "AgenticPayGym":
            model_path = project_root / "agenticpaygym" / "models" / "download_models" / "Qwen3-8B-Instruct"
        else:
            # Fallback: use relative path from current file
            model_path = current_file.parent / "download_models" / "Qwen3-8B-Instruct"
        
        llm = VLLMLLM(
            model_path=str(model_path),
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=2,
        )
        print(f"✓ Successfully initialized: {llm}")
        
        # Test text generation
        print("\n2. Testing text generation...")
        response = llm.generate(
            prompt="你好，请介绍一下你自己。你是什么模型？",
            temperature=0.0,
            max_tokens=256,
            seed=0
        )
        print(f"Response: {response}")
        print("✓ Text generation test completed")
        
        # Test reproducibility (same seed should give same output)
        print("\n3. Testing reproducibility with seed=0...")
        response2 = llm.generate(
            prompt="你好，请介绍一下你自己。你是什么模型？",
            temperature=0.0,
            max_tokens=256,
            seed=0
        )
        print(f"Response (2nd run): {response2}")
        print(f"Identical outputs: {response == response2}")
        print("✓ Reproducibility test completed")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Hint: Install required packages with: pip install vllm")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ Generation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
