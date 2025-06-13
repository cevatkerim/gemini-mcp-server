"""Tests for AzureOpenAIProvider with multiple deployments."""

from unittest.mock import MagicMock

import pytest

from providers.azure import AzureOpenAIProvider
from providers.base import ProviderType


DEPLOYMENTS = {
    "o3": {
        "deployment_name": "o3",
        "streaming": True,
        "api_version": "2024-02-15",
        "fixed_temperature": 1.0,
    },
    "gpt-4.1": {"deployment_name": "gpt-4-1", "streaming": False},
}


class TestAzureOpenAIProvider:
    def test_initialization(self):
        provider = AzureOpenAIProvider(
            api_key="test",
            endpoint_url="https://example.azure.com",
            deployments=DEPLOYMENTS,
        )
        assert provider.endpoint_url == "https://example.azure.com"
        assert provider.validate_model_name("o3")
        assert provider.get_provider_type() == ProviderType.AZURE

    def test_get_capabilities(self):
        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint_url="https://example.azure.com",
            deployments=DEPLOYMENTS,
        )
        caps = provider.get_capabilities("o3")
        assert caps.provider == ProviderType.AZURE
        assert caps.model_name == "o3"
        assert caps.supports_streaming is True
        # Fixed temperature should be enforced
        assert caps.temperature_constraint.get_default() == 1.0

    def test_validate_model_name(self):
        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint_url="https://example.azure.com",
            deployments=DEPLOYMENTS,
        )
        assert provider.validate_model_name("o3")
        assert not provider.validate_model_name("other")

    def test_generate_content_uses_deployment(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="ok"), finish_reason="stop")
        ]
        fake_client.chat.completions.create.return_value.model = "gpt"
        fake_client.chat.completions.create.return_value.id = "1"
        fake_client.chat.completions.create.return_value.created = 0
        fake_client.chat.completions.create.return_value.usage = MagicMock(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )

        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint_url="https://example.azure.com",
            deployments=DEPLOYMENTS,
        )

        monkeypatch.setattr(provider, "_get_client", lambda v: fake_client)

        resp = provider.generate_content(prompt="p", model_name="gpt-4.1")

        fake_client.chat.completions.create.assert_called_once()
        assert resp.friendly_name == "Azure OpenAI"
        assert resp.provider == ProviderType.AZURE

    def test_fixed_temperature_override(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="ok"), finish_reason="stop")
        ]
        fake_client.chat.completions.create.return_value.model = "gpt"
        fake_client.chat.completions.create.return_value.id = "1"
        fake_client.chat.completions.create.return_value.created = 0
        fake_client.chat.completions.create.return_value.usage = MagicMock(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )

        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint_url="https://example.azure.com",
            deployments=DEPLOYMENTS,
        )

        def fake_client_get(api_version):
            return fake_client

        monkeypatch.setattr(provider, "_get_client", fake_client_get)

        provider.generate_content(prompt="p", model_name="o3", temperature=0.2)

        # Temperature should be overridden to 1.0
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 1.0


