"""Qwen3 VL (Vision Language Model) Implementation"""

from typing import Optional, Union, List, Any
from pathlib import Path
import os
import io
import torch
from agenticpaygym.models.base_vlm import BaseVLM


class Qwen3VL(BaseVLM):
    """Qwen3 VL Model Implementation
    
    Uses Qwen3VLForConditionalGeneration for vision language model inference.
    Supports loading models from Hugging Face Hub or local path.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        device_map: Optional[Union[str, dict]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
        **model_kwargs,
    ):
        """Initialize Qwen3 VL Model
        
        Args:
            model_id: Hugging Face Hub model ID (e.g., "Qwen/Qwen3-VL-2B-Instruct")
            model_path: Local model path (e.g., "/path/to/local/model")
            device: Device to load model on ("cuda", "cpu", "cuda:0", etc.). 
                   If None, auto-detects based on CUDA availability.
            device_map: Device mapping strategy ("auto", "balanced", etc.) or dict.
                       If provided, overrides device parameter.
            dtype: Data type for model weights ("auto", "float16", "bfloat16", "float32", etc.)
                   or torch.dtype. Default is "auto".
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", etc.)
            trust_remote_code: Whether to trust remote code
            **model_kwargs: Additional parameters passed to model initialization
            
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
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs
        
        # Determine device
        if device_map is not None:
            self.device_map = device_map
            self.device = None
        else:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            self.device_map = None
        
        # Determine dtype
        if dtype is not None:
            if isinstance(dtype, str):
                if dtype == "auto":
                    self.dtype = "auto"
                else:
                    self.dtype = getattr(torch, dtype, None)
                    if self.dtype is None:
                        raise ValueError(f"Invalid dtype: {dtype}")
            else:
                self.dtype = dtype
        else:
            self.dtype = "auto"
        
        self.attn_implementation = attn_implementation
        
        # Lazy loading: model is loaded on first call
        self._model = None
        self._processor = None
    
    def _load_image(self, image_input: Union[str, Path, Any]) -> Any:
        """Load and convert image to PIL Image format
        
        Args:
            image_input: Image input (path, URL, PIL Image, numpy array, bytes, etc.)
            
        Returns:
            PIL Image object
        """
        try:
            from PIL import Image
            import requests
            import numpy as np
        except ImportError:
            raise ImportError(
                "PIL (Pillow) and requests packages are required. "
                "Install them with: pip install pillow requests"
            )
        
        # If already a PIL Image, return as is
        if isinstance(image_input, Image.Image):
            return image_input
        
        # If it's a Path object, convert to string
        if isinstance(image_input, Path):
            image_input = str(image_input)
        
        # If it's a string, check if it's a URL or file path
        if isinstance(image_input, str):
            # Check if it's a URL
            if image_input.startswith(("http://", "https://")):
                response = requests.get(image_input)
                return Image.open(io.BytesIO(response.content))
            else:
                # File path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                return Image.open(image_input)
        
        # If it's a numpy array
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input)
        
        # If it's bytes
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def _load_model(self):
        """Load model and processor"""
        if self._model is not None:
            return
        
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Transformers package is required. Install it with: pip install transformers"
            )
        
        model_name_or_path = self.model_path or self.model_id
        
        # Check if accelerate is available when using device_map
        if self.device_map:
            try:
                import accelerate
            except ImportError:
                print("Warning: device_map requires 'accelerate' package. Falling back to manual device placement.")
                print("Install accelerate with: pip install accelerate")
                self.device_map = None
                if self.device is None:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Qwen3 VL model from: {model_name_or_path}")
        print(f"Device: {self.device_map or self.device}")
        if self.dtype != "auto":
            print(f"Dtype: {self.dtype}")
        if self.attn_implementation:
            print(f"Attention implementation: {self.attn_implementation}")
        
        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **self.model_kwargs
        )
        
        # Prepare model kwargs
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            **self.model_kwargs
        }
        
        # Set dtype
        if self.dtype == "auto":
            model_kwargs["dtype"] = "auto"
        else:
            model_kwargs["dtype"] = self.dtype
        
        # Set device_map
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        else:
            # If no device_map, we'll move model to device after loading
            pass
        
        # Set attention implementation if specified
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        
        # Load model
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        
        # Move to device if device_map is not used
        if not self.device_map:
            self._model = self._model.to(self.device)
        
        self._model.eval()
        print("✓ Qwen3 VL model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, Path, List[Union[str, Path]], Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ) -> str:
        """Generate text from text prompt and optional images using Qwen3 VL
        
        Args:
            prompt: Input text prompt
            images: Image input(s). Can be:
                - str: Path to image file or image URL
                - Path: Path object to image file
                - List[str/Path]: List of image paths/URLs
                - PIL.Image: PIL Image object
                - numpy.ndarray: Image as numpy array
                - bytes: Image as bytes
                - None: No images (text-only generation)
            temperature: Temperature parameter (controls randomness, 0.0-2.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter (-1 means disabled)
            **kwargs: Other parameters (e.g., do_sample, repetition_penalty, etc.)
            
        Returns:
            Generated text response
        """
        self._load_model()
        
        if images is not None and not self._validate_image_input(images):
            raise ValueError("Invalid image input format")
        
        try:
            # Load images if provided
            pil_images = None
            if images is not None:
                images = images if isinstance(images, list) else [images]
                pil_images = [self._load_image(img) for img in images]
            
            # Prepare messages in Qwen3 VL format
            content = []
            if pil_images:
                for img in pil_images:
                    content.append({
                        "type": "image",
                        "image": img,
                    })
            content.append({
                "type": "text",
                "text": prompt,
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            
            # Prepare inputs using apply_chat_template
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move inputs to model device
            target_device = next(self._model.parameters()).device if self.device_map else self.device
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Prepare generation kwargs
            generation_kwargs = {
                "temperature": temperature if temperature > 0 else None,
                "top_p": top_p if top_p < 1.0 else None,
                "top_k": top_k if top_k > 0 else None,
                "do_sample": temperature > 0 or top_p < 1.0 or top_k > 0,
            }
            if max_tokens is not None:
                generation_kwargs["max_new_tokens"] = max_tokens
            generation_kwargs.update(kwargs)
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            
            # Generate
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, **generation_kwargs)
            
            # Decode output
            # Trim input_ids from generated_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Return first output (or concatenate if multiple)
            if isinstance(output_text, list):
                return output_text[0] if len(output_text) == 1 else " ".join(output_text)
            return output_text
                
        except Exception as e:
            raise RuntimeError(f"Qwen3 VL generation error: {e}")
    
    def __repr__(self) -> str:
        """Return string representation of VLM"""
        model_source = self.model_path if self.model_path else self.model_id
        device_info = self.device_map if self.device_map else self.device
        dtype_info = self.dtype if self.dtype != "auto" else "auto"
        return f"Qwen3VL(model={model_source}, device={device_info}, dtype={dtype_info})"


if __name__ == "__main__":
    """Test Qwen3 VL Model"""
    import sys
    
    print("=" * 50)
    print("Testing Qwen3 VL Model")
    print("=" * 50)
    
    try:
        # Test loading model from Hugging Face Hub or local path
        print("\n1. Testing loading model...")
        print("Note: This will download the model if not cached.")
        print("Note: GPU is recommended for better performance, but CPU is also supported.")
        
        # Use Qwen3 VL model
        # Build relative path starting from AgenticPayGym (adaptive to any installation location)
        # Find project root (AgenticPayGym directory) by searching up the directory tree
        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        # Search up the directory tree to find AgenticPayGym directory
        while project_root.name != "AgenticPayGym" and project_root.parent != project_root:
            project_root = project_root.parent
        
        # Build path from AgenticPayGym root (or fallback to relative path if not found)
        if project_root.name == "AgenticPayGym":
            model_path = project_root / "agenticpaygym" / "models" / "download_models" / "Qwen3-VL-2B-Instruct"
        else:
            # Fallback: use relative path from current file if AgenticPayGym not found
            model_path = current_file.parent / "download_models" / "Qwen3-VL-2B-Instruct"
        
        vlm = Qwen3VL(
            # model_id="Qwen/Qwen3-VL-2B-Instruct",  # From Hugging Face Hub
            model_path=str(model_path),  # Local path (relative to AgenticPayGym)
            device=None,  # Auto-detect device
            dtype="auto",  # Auto-detect dtype
            device_map="auto",  # Auto device mapping
            # attn_implementation="flash_attention_2",  # Optional: enable flash attention
        )
        print(f"✓ Successfully initialized: {vlm}")
        
        # Test generation with image URL
        print("\n2. Testing image-to-text generation with URL...")
        response = vlm.generate(
            prompt="Describe this image.",
            images="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            temperature=0.7,
            max_tokens=128
        )
        print(f"Response: {response}")
        print("✓ Image-to-text generation test completed")
        
        # Test text-only generation
        print("\n3. Testing text-only generation...")
        text_response = vlm.generate(
            prompt="你好，请介绍一下你自己。",
            temperature=0.7,
            max_tokens=256
        )
        print(f"Response: {text_response}")
        print("✓ Text-only generation test completed")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Hint: Install required packages with: pip install transformers torch pillow requests")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ Generation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

