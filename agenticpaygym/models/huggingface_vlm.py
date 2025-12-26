"""Hugging Face VLM (Vision Language Model) Implementation"""

from typing import Optional, Union, List, Any
from pathlib import Path
import os
import io
from agenticpaygym.models.base_vlm import BaseVLM


class HuggingFaceVLM(BaseVLM):
    """Hugging Face VLM Implementation
    
    Supports loading vision language models from Hugging Face Hub or local path for inference.
    Supports models like BLIP, LLaVA, InstructBLIP, Qwen-VL, etc.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        model_type: str = "image-to-text",
        trust_remote_code: bool = False,
        use_pipeline: bool = True,
        **model_kwargs,
    ):
        """Initialize Hugging Face VLM
        
        Args:
            model_id: Hugging Face Hub model ID (e.g., "Salesforce/blip-image-captioning-base", 
                     "llava-hf/llava-1.5-7b-hf", "Qwen/Qwen-VL")
            model_path: Local model path (e.g., "/path/to/local/model")
            device: Device type ("auto", "cpu", "cuda", "cuda:0", etc.)
            model_type: Model type ("image-to-text", "image-to-text-generation", etc.)
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
        
        # Lazy loading: model, tokenizer, and image processor are loaded on first call
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._processor = None  # Some models use a unified processor
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
                return Image.open(response.raw)
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
        """Load model, tokenizer, and image processor"""
        if self._pipeline is not None or (self._model is not None and self._tokenizer is not None):
            return  # Already loaded
        
        try:
            from transformers import (
                AutoModelForVision2Seq,
                AutoProcessor,
                AutoImageProcessor,
                AutoTokenizer,
                pipeline,
            )
        except ImportError:
            raise ImportError(
                "transformers package is required. Install it with: pip install transformers"
            )
        
        # Determine model path
        model_name_or_path = self.model_path if self.model_path else self.model_id
        device = self._get_device()
        
        print(f"Loading VLM model from: {model_name_or_path}")
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
            # Manually load model, tokenizer, and image processor
            print("Loading processor/tokenizer...")
            try:
                # Try to load unified processor first (used by many VLMs)
                self._processor = AutoProcessor.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                )
                print("✓ Loaded unified processor")
            except Exception:
                # Fall back to separate tokenizer and image processor
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=self.trust_remote_code,
                    )
                    self._image_processor = AutoImageProcessor.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=self.trust_remote_code,
                    )
                    print("✓ Loaded tokenizer and image processor separately")
                except Exception as e:
                    raise RuntimeError(f"Failed to load processor: {e}")
            
            print("Loading model...")
            try:
                # Try AutoModelForVision2Seq first (common for VLMs)
                self._model = AutoModelForVision2Seq.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.model_kwargs,
                )
            except Exception:
                # Fall back to other model types if needed
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.model_kwargs,
                )
            
            # Move model to specified device
            if device != "cpu":
                self._model = self._model.to(device)
            
            print("✓ Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, Path, List[Union[str, Path]], Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text from text prompt and optional images
        
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
            temperature: Temperature parameter (controls randomness)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters (e.g., top_p, top_k, do_sample, etc.)
            
        Returns:
            Generated text response
        """
        # Lazy load model
        self._load_model()
        
        # Validate image input
        if images is not None and not self._validate_image_input(images):
            raise ValueError("Invalid image input format")
        
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
                
                # Prepare inputs for pipeline
                if images is not None:
                    # Convert images to list if single image
                    if not isinstance(images, list):
                        images = [images]
                    
                    # Load images
                    pil_images = [self._load_image(img) for img in images]
                    
                    # Pipeline expects different formats depending on model
                    if len(pil_images) == 1:
                        result = self._pipeline(pil_images[0], prompt, **generation_kwargs)
                    else:
                        result = self._pipeline(pil_images, prompt, **generation_kwargs)
                else:
                    # Text-only generation
                    result = self._pipeline(prompt, **generation_kwargs)
                
                # Pipeline return format may be string or list of dictionaries
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        return result[0].get("generated_text", result[0].get("caption", str(result[0])))
                    else:
                        return str(result[0])
                elif isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    return result.get("generated_text", result.get("caption", str(result)))
                else:
                    return str(result)
            else:
                # Manually generate using model, tokenizer, and image processor
                device = self._get_device()
                
                # Prepare images
                if images is not None:
                    # Convert images to list if single image
                    if not isinstance(images, list):
                        images = [images]
                    
                    # Load images
                    pil_images = [self._load_image(img) for img in images]
                else:
                    pil_images = None
                
                # Prepare inputs
                if self._processor is not None:
                    # Use unified processor
                    if pil_images is not None:
                        inputs = self._processor(
                            images=pil_images,
                            text=prompt,
                            return_tensors="pt",
                            padding=True,
                        )
                    else:
                        inputs = self._processor(
                            text=prompt,
                            return_tensors="pt",
                            padding=True,
                        )
                else:
                    # Use separate tokenizer and image processor
                    if pil_images is not None:
                        image_inputs = self._image_processor(
                            images=pil_images,
                            return_tensors="pt",
                        )
                        text_inputs = self._tokenizer(
                            prompt,
                            return_tensors="pt",
                            padding=True,
                        )
                        inputs = {**image_inputs, **text_inputs}
                    else:
                        inputs = self._tokenizer(
                            prompt,
                            return_tensors="pt",
                            padding=True,
                        )
                
                # Move inputs to device
                if device != "cpu":
                    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Generation parameters
                generation_config = {
                    "max_new_tokens": max_tokens or 512,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    **kwargs,
                }
                
                # Add pad_token_id if tokenizer has it
                if self._tokenizer is not None and self._tokenizer.pad_token_id is not None:
                    generation_config["pad_token_id"] = self._tokenizer.pad_token_id
                elif self._processor is not None and hasattr(self._processor, 'tokenizer'):
                    if self._processor.tokenizer.pad_token_id is not None:
                        generation_config["pad_token_id"] = self._processor.tokenizer.pad_token_id
                
                # Remove None values
                generation_config = {k: v for k, v in generation_config.items() if v is not None}
                
                # Generate
                outputs = self._model.generate(**inputs, **generation_config)
                
                # Decode output
                if self._processor is not None:
                    generated_text = self._processor.decode(outputs[0], skip_special_tokens=True)
                else:
                    generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove input prompt from output if present
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
                
                return generated_text.strip()
                
        except Exception as e:
            raise RuntimeError(f"Hugging Face VLM generation error: {e}")
    
    def __repr__(self) -> str:
        """Return string representation of VLM"""
        model_source = self.model_path if self.model_path else self.model_id
        return f"HuggingFaceVLM(model={model_source}, device={self.device})"


if __name__ == "__main__":
    """Test Hugging Face VLM"""
    import sys
    
    print("=" * 50)
    print("Testing Hugging Face VLM")
    print("=" * 50)
    
    try:
        # Test loading model from Hugging Face Hub
        print("\n1. Testing loading from Hugging Face Hub...")
        print("Note: This will download the model if not cached.")
        
        # Use a smaller model for testing (BLIP base model)
        vlm = HuggingFaceVLM(
            model_id="Salesforce/blip-image-captioning-base",  # Small VLM model for testing
            device="cpu",  # Use CPU for testing
            use_pipeline=True,
        )
        print(f"✓ Successfully initialized: {vlm}")
        
        # Test generation with image (requires an actual image file)
        print("\n2. Testing image-to-text generation...")
        print("Note: This requires an image file. Using a placeholder test.")
        
        # For actual testing, you would provide an image path:
        # test_image = "path/to/your/image.jpg"
        # response = vlm.generate(
        #     prompt="What's in this image?",
        #     images=test_image,
        #     temperature=0.7,
        #     max_tokens=100
        # )
        
        print("✓ VLM model loaded successfully")
        print("To test with actual images, provide an image path to the generate() method.")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Hint: Install required packages with: pip install transformers pillow requests")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ Generation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

