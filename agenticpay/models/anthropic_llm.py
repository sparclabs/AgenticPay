"""Anthropic LLM Implementation

Connects to the Anthropic API (Claude models) using the official anthropic SDK.

Typical usage:
    model = AnthropicLLM(
        model="claude-opus-4-5",
        api_key="sk-ant-...",   # or set ANTHROPIC_API_KEY env var
    )
"""

from typing import Optional
import os
from agenticpay.models.base_llm import BaseLLM


class AnthropicLLM(BaseLLM):
    """Anthropic LLM implementation using the official anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        api_key: Optional[str] = None,
    ):
        """Initialize Anthropic LLM client.

        Args:
            model: Claude model ID, e.g. "claude-opus-4-5", "claude-sonnet-4-5",
                   "claude-haiku-4-5-20251001".
            api_key: Anthropic API key. If not provided, read from the
                     ANTHROPIC_API_KEY environment variable.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install it with: pip install anthropic"
            )

        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set it via api_key parameter or the ANTHROPIC_API_KEY environment variable."
            )

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text via the Anthropic Messages API.

        Args:
            prompt: Input text prompt.
            temperature: Sampling temperature (0.0–1.0).
            max_tokens: Maximum tokens to generate (default: 1024).
            **kwargs: Additional parameters forwarded to messages.create().

        Returns:
            Generated text response.
        """
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def __repr__(self) -> str:
        return f"AnthropicLLM(model={self.model})"
