"""Azure OpenAI provider implementation."""

from typing import Optional

from .base import (
    ModelCapabilities,
    ProviderType,
    RangeTemperatureConstraint,
)
from .openai_compatible import OpenAICompatibleProvider


class AzureOpenAIProvider(OpenAICompatibleProvider):
    """Azure OpenAI model provider."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint_url: str,
        deployment_name: str,
        api_version: str = "2023-07-01-preview",
        streaming: bool = True,
        **kwargs,
    ):
        """Initialize Azure OpenAI provider with endpoint and deployment."""
        super().__init__(api_key, **kwargs)
        self.endpoint_url = endpoint_url
        self.deployment_name = deployment_name
        self.api_version = api_version
        self._supports_streaming = bool(streaming)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except Exception as e:  # pragma: no cover - import error surfaces in tests
                raise ImportError("openai>=1.0.0 is required for Azure support") from e
            self._client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint_url,
                api_version=self.api_version,
            )
        return self._client

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Return capabilities for the configured deployment."""
        if model_name != self.deployment_name:
            raise ValueError(f"Unsupported Azure deployment: {model_name}")

        return ModelCapabilities(
            provider=ProviderType.AZURE,
            model_name=self.deployment_name,
            friendly_name="Azure OpenAI",
            context_window=200_000,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=self._supports_streaming,
            supports_function_calling=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
        )

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Generate content using Azure OpenAI deployment."""
        kwargs.setdefault("stream", self._supports_streaming)
        return super().generate_content(
            prompt=prompt,
            model_name=self.deployment_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def get_provider_type(self) -> ProviderType:
        return ProviderType.AZURE

    def validate_model_name(self, model_name: str) -> bool:
        return model_name == self.deployment_name

    def supports_thinking_mode(self, model_name: str) -> bool:  # pragma: no cover - none support
        return False
