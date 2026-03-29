"""vLLM HTTP LLM Implementation

This module provides a LLM implementation that connects to a running vLLM server
via its OpenAI-compatible HTTP API (started with `vllm serve`).

Unlike VLLMLLM (which loads the model in-process), this class requires no GPU
locally — it delegates inference to an already-running vLLM endpoint.

Typical usage:
    # Start vLLM server first:
    #   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    model = VLLMHttpLLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000",
    )
"""

from typing import Optional
from agenticpay.models.base_llm import BaseLLM


class VLLMHttpLLM(BaseLLM):
    """vLLM HTTP LLM Implementation

    Connects to a running vLLM server via its OpenAI-compatible HTTP API.
    The server is started externally (e.g., `vllm serve <model> --port 8000`).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        api_key: str = "EMPTY",
        enable_thinking: bool = False,
    ):
        """Initialize vLLM HTTP LLM client

        Args:
            model: The model name being served (must match what the vLLM server was started with,
                   e.g. "Qwen/Qwen2.5-7B-Instruct" or a local path).
            base_url: Base URL of the running vLLM server (default: "http://localhost:8000").
                      The /v1 suffix is appended automatically.
            api_key: API key sent in the Authorization header. vLLM does not validate
                     this by default; "EMPTY" is the conventional placeholder.
            enable_thinking: Enable chain-of-thought thinking for Qwen3 reasoning models
                             (default: False). When False, the model returns a direct answer
                             without spending tokens on reasoning steps.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        self.model = model
        self.api_key = api_key
        self.enable_thinking = enable_thinking

        # Normalise base_url: strip trailing slash, then append /v1
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self.base_url = base_url

        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self._check_server()

    def _check_server(self):
        """Verify the vLLM server is reachable and serving the expected model."""
        import urllib.request
        import urllib.error

        # Hit /v1/models — lightweight endpoint available on all vLLM servers
        health_url = self.base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(health_url, timeout=5):
                pass
        except urllib.error.URLError as e:
            server_root = self.base_url.replace("/v1", "")
            raise ConnectionError(
                f"Cannot reach vLLM server at {server_root}.\n"
                f"  Start it with:  vllm serve {self.model} --port "
                f"{server_root.rsplit(':', 1)[-1] if ':' in server_root else '8000'}\n"
                f"  Then retry.  (original error: {e})"
            ) from e

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text via the vLLM HTTP server

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (0.0–2.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters forwarded to the chat completions API

        Returns:
            Generated text response
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
                **kwargs,
            )
            msg = response.choices[0].message
            # Qwen3 reasoning models set content=None when thinking is enabled;
            # the final answer is in reasoning_content in that case.
            content = msg.content
            if content is None:
                content = getattr(msg, "reasoning_content", None) or ""
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"vLLM HTTP generation error: {e}")

    def __repr__(self) -> str:
        return f"VLLMHttpLLM(model={self.model}, base_url={self.base_url})"
