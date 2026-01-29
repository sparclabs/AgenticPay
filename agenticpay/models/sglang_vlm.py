"""SGLang VLM (Vision Language Model) Implementation

This module provides a VLM implementation using SGLang Runtime with OpenAI-compatible API.
Supports both text-only and vision-language generation.
"""

from typing import Optional, Union, List, Any
from pathlib import Path
import os
import io
import base64
from agenticpay.models.base_vlm import BaseVLM


class SGLangVLM(BaseVLM):
    """SGLang VLM Implementation
    
    Uses SGLang Runtime for efficient vision language model inference.
    Supports loading models from local path and uses OpenAI-compatible API.
    """
    
    def __init__(
        self,
        model_path: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **runtime_kwargs,
    ):
        """Initialize SGLang VLM Model
        
        Args:
            model_path: Local model path (e.g., "/path/to/local/model")
            api_key: API key for OpenAI client (default: "None" for SGLang)
            base_url: Custom base URL (optional, will be auto-detected from Runtime)
            **runtime_kwargs: Additional parameters passed to SGLang Runtime initialization
            
        Note:
            SGLang Runtime will start a local server automatically.
            The server endpoint will be auto-detected from the Runtime object.
        """
        try:
            from sglang import Runtime
            import openai
        except ImportError:
            raise ImportError(
                "SGLang and OpenAI packages are required. "
                "Install them with: pip install sglang openai"
            )
        
        # Disable SGLang's CuDNN compatibility check (since vLLM works fine)
        os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
        
        if not model_path:
            raise ValueError("model_path must be provided.")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.api_key = api_key or "None"  # SGLang doesn't require API key
        self.runtime_kwargs = runtime_kwargs
        
        # Lazy loading: Runtime and client are initialized on first call
        self._runtime = None
        self._client = None
        self._base_url = base_url
    
    def _init_runtime(self):
        """Initialize SGLang Runtime and OpenAI client"""
        if self._runtime is not None:
            return
        
        from sglang import Runtime
        import openai
        
        print(f"Loading SGLang model from: {self.model_path}")
        
        # Initialize SGLang Runtime (this will start a local server)
        self._runtime = Runtime(model_path=self.model_path, **self.runtime_kwargs)
        print("✓ SGLang model loaded successfully")
        
        # Get server address (SGLang default port is 30000)
        if self._base_url:
            base_url = self._base_url
        else:
            # Try to get URL from runtime
            if hasattr(self._runtime, 'url'):
                base_url = self._runtime.url
            elif hasattr(self._runtime, 'endpoint'):
                endpoint = self._runtime.endpoint
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
        
        self._base_url = base_url
        print(f"✓ Server endpoint: {base_url}")
        
        # Create OpenAI-compatible client (connects to SGLang server via base_url)
        self._client = openai.OpenAI(
            base_url=f"{base_url}/v1",
            api_key=self.api_key
        )
        print("✓ Client connected successfully")
    
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
        """Encode image to base64 string for SGLang
        
        Args:
            image_input: Image input (path, URL, PIL Image, numpy array, bytes, etc.)
            
        Returns:
            Base64 encoded image string
        """
        # Load image to PIL Image
        pil_image = self._load_image(image_input)
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        # Save as PNG for better quality
        pil_image.save(buffered, format='PNG')
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return image_base64
    
    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, Path, List[Union[str, Path]], Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text from text prompt and optional images using SGLang
        
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
            **kwargs: Other parameters passed to OpenAI API (e.g., top_p, frequency_penalty, etc.)
            
        Returns:
            Generated text response
        """
        self._init_runtime()
        
        if images is not None and not self._validate_image_input(images):
            raise ValueError("Invalid image input format")
        
        try:
            # Handle text-only generation
            if images is None:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens or 256,
                    **kwargs,
                )
                return response.choices[0].message.content.strip()
            
            # Handle vision-language generation
            # Convert images to list if single image
            if not isinstance(images, list):
                images = [images]
            
            # Encode images to base64
            encoded_images = [self._encode_image_base64(img) for img in images]
            
            # Prepare multimodal input
            # Format for OpenAI-compatible API with vision models
            content = []
            for img_base64 in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
            content.append({
                "type": "text",
                "text": prompt,
            })
            
            # Generate with multimodal input
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens or 256,
                **kwargs,
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            raise RuntimeError(f"SGLang generation error: {e}")
    
    def shutdown(self):
        """Shutdown SGLang Runtime and clean up resources"""
        if self._runtime is not None and hasattr(self._runtime, 'shutdown'):
            try:
                self._runtime.shutdown()
                print("✓ SGLang Runtime shut down successfully")
            except Exception as e:
                print(f"Warning: Error during shutdown: {e}")
            finally:
                self._runtime = None
                self._client = None
    
    def __repr__(self) -> str:
        """Return string representation of VLM"""
        return f"SGLangVLM(model_path={self.model_path})"
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass


if __name__ == "__main__":
    """Test SGLang VLM Model"""
    import sys
    
    print("=" * 50)
    print("Testing SGLang VLM Model")
    print("=" * 50)
    
    try:
        # Test loading model
        print("\n1. Testing loading model...")
        print("Note: This will load the model from local path.")
        
        # Use Qwen3 VL model
        # Build relative path starting from AgenticPayGym
        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        # Search up the directory tree to find AgenticPayGym directory
        while project_root.name != "AgenticPayGym" and project_root.parent != project_root:
            project_root = project_root.parent
        
        # Build path from AgenticPayGym root
        if project_root.name == "AgenticPayGym":
            model_path = project_root / "agenticpay" / "models" / "download_models" / "Qwen3-VL-8B-Instruct"
        else:
            # Fallback: use relative path from current file
            model_path = current_file.parent / "download_models" / "Qwen3-VL-8B-Instruct"
        
        vlm = SGLangVLM(model_path=str(model_path))
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
        
        # Cleanup
        print("\n4. Cleaning up...")
        vlm.shutdown()
        
        print("\n" + "=" * 50)
        print("Test completed!")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Hint: Install required packages with: pip install sglang openai pillow requests")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ Generation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

