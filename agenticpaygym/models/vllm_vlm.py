"""vLLM VLM (Vision Language Model) Implementation

This module provides a VLM implementation using vLLM for efficient inference.
Supports both text-only and vision-language generation.
"""

from typing import Optional, Union, List, Any
from pathlib import Path
import os
import io
from agenticpaygym.models.base_vlm import BaseVLM


class VLLMVLM(BaseVLM):
    """vLLM VLM Implementation
    
    Uses vLLM for efficient vision language model inference.
    Supports loading models from Hugging Face Hub or local path.
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
        """Initialize vLLM VLM Model
        
        Args:
            model_id: Hugging Face Hub model ID (e.g., "Qwen/Qwen3-VL-2B-Instruct")
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
        
        print(f"Loading vLLM model from: {model_name_or_path}")
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
        print("✓ vLLM model loaded successfully")
    
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
    
    def _encode_image_base64(self, image_input: Union[str, Path, Any]) -> str:
        """Encode image to base64 string for vLLM
        
        Args:
            image_input: Image input (path, URL, PIL Image, numpy array, bytes, etc.)
            
        Returns:
            Base64 encoded image string
        """
        try:
            from vllm.multimodal.utils import encode_image_base64
        except ImportError:
            raise ImportError(
                "vLLM multimodal utils are required. "
                "Make sure you have vLLM installed with multimodal support."
            )
        
        # If it's a string path or Path, use directly
        if isinstance(image_input, (str, Path)):
            return encode_image_base64(str(image_input))
        
        # For other types, convert to PIL Image first, then save to temp file
        pil_image = self._load_image(image_input)
        
        # Save PIL Image to temporary file and encode
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pil_image.save(tmp_file.name, format='PNG')
            try:
                encoded = encode_image_base64(tmp_file.name)
            finally:
                os.unlink(tmp_file.name)
        
        return encoded
    
    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, Path, List[Union[str, Path]], Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text from text prompt and optional images using vLLM
        
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
            **kwargs: Other parameters passed to SamplingParams
            
        Returns:
            Generated text response
        """
        from vllm import SamplingParams
        
        self._load_model()
        
        if images is not None and not self._validate_image_input(images):
            raise ValueError("Invalid image input format")
        
        try:
            # Prepare sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens or 256,
                top_p=top_p,
                **kwargs
            )
            
            # Handle text-only generation
            if images is None:
                outputs = self._llm.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text
                return response.strip()
            
            # Handle vision-language generation
            # Convert images to list if single image
            if not isinstance(images, list):
                images = [images]
            
            # Encode images to base64
            encoded_images = [self._encode_image_base64(img) for img in images]
            
            # Prepare multimodal input
            # Format depends on the model's chat template
            # For Qwen models, use the format shown in test file
            content = []
            for img_base64 in encoded_images:
                content.append({
                    "type": "image",
                    "image": img_base64,
                })
            content.append({
                "type": "text",
                "text": prompt,
            })
            
            multimodal_input = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Generate with multimodal input
            outputs = self._llm.generate(multimodal_input, sampling_params)
            response = outputs[0].outputs[0].text
            return response.strip()
                
        except Exception as e:
            raise RuntimeError(f"vLLM generation error: {e}")
    
    def __repr__(self) -> str:
        """Return string representation of VLM"""
        model_source = self.model_path if self.model_path else self.model_id
        return f"VLLMVLM(model={model_source}, gpu_memory_utilization={self.gpu_memory_utilization}, tensor_parallel_size={self.tensor_parallel_size})"


if __name__ == "__main__":
    """Test vLLM VLM Model"""
    import sys
    
    print("=" * 50)
    print("Testing vLLM VLM Model")
    print("=" * 50)
    
    try:
        # Test loading model
        print("\n1. Testing loading model...")
        print("Note: This will load the model from local path.")
        print("Note: GPU is required for vLLM.")
        
        # Use Qwen3 VL model
        # Build relative path starting from AgenticPayGym
        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        # Search up the directory tree to find AgenticPayGym directory
        while project_root.name != "AgenticPayGym" and project_root.parent != project_root:
            project_root = project_root.parent
        
        # Build path from AgenticPayGym root
        if project_root.name == "AgenticPayGym":
            model_path = project_root / "agenticpaygym" / "models" / "download_models" / "Qwen3-VL-2B-Instruct"
        else:
            # Fallback: use relative path from current file
            model_path = current_file.parent / "download_models" / "Qwen3-VL-2B-Instruct"
        
        vlm = VLLMVLM(
            model_path=str(model_path),
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=2,
        )
        print(f"✓ Successfully initialized: {vlm}")
        
        # Test text-only generation
        print("\n2. Testing text-only generation...")
        response = vlm.generate(
            prompt="你好，请介绍一下你自己。你是什么模型？",
            temperature=0.7,
            max_tokens=256
        )
        print(f"Response: {response}")
        print("✓ Text-only generation test completed")
        
        # Test vision-language generation (if image available)
        print("\n3. Testing vision-language generation...")
        print("Note: This requires an image file. Skipping if not available.")
        # Uncomment and provide image path to test:
        # test_image = "path/to/your/image.jpg"
        # response = vlm.generate(
        #     prompt="请详细描述这张图片的内容。",
        #     images=test_image,
        #     temperature=0.7,
        #     max_tokens=256
        # )
        # print(f"Response: {response}")
        # print("✓ Vision-language generation test completed")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Hint: Install required packages with: pip install vllm pillow requests")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ Generation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


