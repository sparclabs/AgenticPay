"""Hugging Face LLM Implementation"""

from typing import Optional, Union
import os
from agenticpaygym.models.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    """Hugging Face LLM Implementation
    
    Supports loading models from Hugging Face Hub or local path for inference.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        model_type: str = "text-generation",
        trust_remote_code: bool = False,
        use_pipeline: bool = True,
        **model_kwargs,
    ):
        """Initialize Hugging Face LLM
        
        Args:
            model_id: Hugging Face Hub model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
            model_path: Local model path (e.g., "/path/to/local/model")
            device: Device type ("auto", "cpu", "cuda", "cuda:0", etc.)
            model_type: Model type ("text-generation", "chat", etc.)
            trust_remote_code: Whether to trust remote code
            use_pipeline: Whether to use transformers pipeline (recommended, simpler)
            **model_kwargs: Additional parameters passed to the model (e.g., load_in_8bit, load_in_4bit, etc.)
            
        Note:
            Either model_id or model_path must be provided.
            If both are provided, model_path takes priority.
        """
        if not model_id and not model_path:
            raise ValueError(
                "Either model_id (Hugging Face Hub) or model_path (local) must be provided."
            )
        
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        self.trust_remote_code = trust_remote_code
        self.use_pipeline = use_pipeline
        self.model_kwargs = model_kwargs
        
        # Lazy loading: model and tokenizer are loaded on first call
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._device = None
        
    def _get_device(self):
        """Auto-detect and return device"""
        if self._device is not None:
            return self._device
            
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = self.device
            
        return self._device
    
    def _load_model(self):
        """Load model and tokenizer"""
        if self._pipeline is not None or (self._model is not None and self._tokenizer is not None):
            return  # Already loaded
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise ImportError(
                "transformers package is required. Install it with: pip install transformers"
            )
        
        # Determine model path
        model_name_or_path = self.model_path if self.model_path else self.model_id
        device = self._get_device()
        
        print(f"Loading model from: {model_name_or_path}")
        print(f"Using device: {device}")
        
        if self.use_pipeline:
            # Use pipeline (recommended, simpler)
            try:
                self._pipeline = pipeline(
                    self.model_type,
                    model=model_name_or_path,
                    device_map=device if device != "cpu" else -1,
                    trust_remote_code=self.trust_remote_code,
                    **self.model_kwargs,
                )
                print("✓ Model loaded successfully using pipeline")
            except Exception as e:
                print(f"Pipeline loading failed: {e}")
                print("Falling back to manual loading...")
                self.use_pipeline = False
        
        if not self.use_pipeline or self._pipeline is None:
            # Manually load model and tokenizer
            print("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
            
            # If pad_token is None, use eos_token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            print("Loading model...")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **self.model_kwargs,
            )
            
            # Move model to specified device
            if device != "cpu":
                self._model = self._model.to(device)
            
            print("✓ Model and tokenizer loaded successfully")
    
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
            temperature: Temperature parameter (controls randomness)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters (e.g., top_p, top_k, do_sample, etc.)
            
        Returns:
            Generated text
        """
        # Lazy load model
        self._load_model()
        
        try:
            if self.use_pipeline and self._pipeline is not None:
                # Generate using pipeline
                generation_kwargs = {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens or 512,
                    "do_sample": temperature > 0,
                    **kwargs,
                }
                
                # Remove None values
                generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
                
                result = self._pipeline(prompt, **generation_kwargs)
                
                # Pipeline return format may be string or list of dictionaries
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        return result[0].get("generated_text", str(result[0]))
                    else:
                        return str(result[0])
                elif isinstance(result, str):
                    return result
                else:
                    return str(result)
            else:
                # Manually generate using model and tokenizer
                device = self._get_device()
                
                # Encode input
                inputs = self._tokenizer(prompt, return_tensors="pt")
                if device != "cpu":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generation parameters
                generation_config = {
                    "max_new_tokens": max_tokens or 512,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "pad_token_id": self._tokenizer.pad_token_id,
                    **kwargs,
                }
                
                # Remove None values
                generation_config = {k: v for k, v in generation_config.items() if v is not None}
                
                # Generate
                outputs = self._model.generate(**inputs, **generation_config)
                
                # Decode output
                # Only return newly generated part (remove input part)
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                return generated_text.strip()
                
        except Exception as e:
            raise RuntimeError(f"Hugging Face model generation error: {e}")
    
    def __repr__(self) -> str:
        """Return string representation of LLM"""
        model_source = self.model_path if self.model_path else self.model_id
        return f"HuggingFaceLLM(model={model_source}, device={self.device})"


if __name__ == "__main__":
    """Test Hugging Face LLM"""
    import sys
    
    print("=" * 50)
    print("Testing Hugging Face LLM")
    print("=" * 50)
    
    try:
        # Test loading model from Hugging Face Hub
        print("\n1. Testing loading from Hugging Face Hub...")
        print("Note: This will download the model if not cached.")
        
        # Use a smaller model for testing
        llm = HuggingFaceLLM(
            model_id="gpt2",  # Use small model for testing
            device="cpu",  # Use CPU for testing
            use_pipeline=True,
        )
        print(f"✓ Successfully initialized: {llm}")
        
        # Test generation
        print("\n2. Testing text generation...")
        test_prompt = "Artificial intelligence is"
        print(f"Input prompt: {test_prompt}")
        
        response = llm.generate(
            prompt=test_prompt,
            temperature=0.7,
            max_tokens=50
        )
        
        print(f"\n✓ Generation successful!")
        print(f"Response content: {response}")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Hint: Install transformers with: pip install transformers")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ Generation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

