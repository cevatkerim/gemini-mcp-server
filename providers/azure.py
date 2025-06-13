"""Azure OpenAI provider implementation supporting multiple deployments."""

from dataclasses import dataclass
from typing import Dict, Optional

from .base import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    RangeTemperatureConstraint,
)
from .openai_compatible import OpenAICompatibleProvider


@dataclass
class AzureDeployment:
    """Configuration for an Azure OpenAI deployment."""

    deployment_name: str
    api_version: str
    streaming: bool = True
    context_window: int = 200_000
    supports_extended_thinking: bool = False


class AzureOpenAIProvider(OpenAICompatibleProvider):
    """Azure OpenAI provider with support for multiple deployments."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint_url: str,
        deployments: Dict[str, dict],
        default_api_version: str = "2023-07-01-preview",
        **kwargs,
    ) -> None:
        """Initialize provider with endpoint and deployment configuration."""
        super().__init__(api_key, **kwargs)
        self.endpoint_url = endpoint_url
        self.default_api_version = default_api_version
        self.deployments: Dict[str, AzureDeployment] = {}
        for name, cfg in deployments.items():
            self.deployments[name] = AzureDeployment(
                deployment_name=cfg.get("deployment_name", name),
                api_version=cfg.get("api_version", default_api_version),
                streaming=cfg.get("streaming", True),
                context_window=cfg.get("context_window", 200_000),
                supports_extended_thinking=cfg.get("supports_extended_thinking", False),
            )
        self._clients: dict[str, object] = {}

    def _get_client(self, api_version: str):
        """Get or create Azure client for the specified API version."""
        if api_version not in self._clients:
            try:
                from openai import AzureOpenAI
            except Exception as e:  # pragma: no cover - import error surfaces in tests
                raise ImportError("openai>=1.0.0 is required for Azure support") from e
            self._clients[api_version] = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint_url,
                api_version=api_version,
            )
        return self._clients[api_version]

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        config = self.deployments.get(model_name)
        if not config:
            raise ValueError(f"Unsupported Azure deployment: {model_name}")

        return ModelCapabilities(
            provider=ProviderType.AZURE,
            model_name=model_name,
            friendly_name="Azure OpenAI",
            context_window=config.context_window,
            supports_extended_thinking=config.supports_extended_thinking,
            supports_system_prompts=True,
            supports_streaming=config.streaming,
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
        config = self.deployments.get(model_name)
        if not config:
            raise ValueError(f"Unsupported Azure deployment: {model_name}")

        kwargs.setdefault("stream", config.streaming)
        client = self._get_client(config.api_version)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion_params = {
            "model": config.deployment_name,
            "messages": messages,
            "temperature": temperature,
        }

        if max_output_tokens:
            completion_params["max_tokens"] = max_output_tokens

        for key, value in kwargs.items():
            if key in ["top_p", "frequency_penalty", "presence_penalty", "seed", "stop", "stream"]:
                completion_params[key] = value

        try:
            response = client.chat.completions.create(**completion_params)
            content = response.choices[0].message.content
            usage = self._extract_usage(response)

            return ModelResponse(
                content=content,
                usage=usage,
                model_name=model_name,
                friendly_name="Azure OpenAI",
                provider=ProviderType.AZURE,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                    "id": response.id,
                    "created": response.created,
                },
            )
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API error for model {model_name}: {e}") from e

    def get_provider_type(self) -> ProviderType:
        return ProviderType.AZURE

    def validate_model_name(self, model_name: str) -> bool:
        return model_name in self.deployments

    def supports_thinking_mode(self, model_name: str) -> bool:
        config = self.deployments.get(model_name)
        return bool(config and config.supports_extended_thinking)
