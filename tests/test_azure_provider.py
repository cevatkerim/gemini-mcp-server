"""Tests for AzureOpenAIProvider."""

from unittest.mock import MagicMock, patch

import pytest

from providers.azure import AzureOpenAIProvider
from providers.base import ProviderType


class TestAzureOpenAIProvider:
    """Basic tests for Azure provider."""

    def test_initialization(self):
        provider = AzureOpenAIProvider(
            api_key="test",
            endpoint_url="https://example.azure.com",
            deployment_name="test-dep",
            api_version="2024-02-15",
            streaming=False,
        )
        assert provider.endpoint_url == "https://example.azure.com"
        assert provider.deployment_name == "test-dep"
        assert provider.api_version == "2024-02-15"
        assert provider.get_provider_type() == ProviderType.AZURE

    def test_get_capabilities(self):
        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint_url="https://example.azure.com",
            deployment_name="dep",
            streaming=True,
        )
        caps = provider.get_capabilities("dep")
        assert caps.provider == ProviderType.AZURE
        assert caps.model_name == "dep"
        assert caps.supports_streaming is True

    def test_validate_model_name(self):
        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint_url="https://example.azure.com",
            deployment_name="dep",
        )
        assert provider.validate_model_name("dep")
        assert not provider.validate_model_name("other")

    @patch("providers.azure.OpenAICompatibleProvider.generate_content")
    def test_generate_content_uses_deployment(self, mock_gen):
        mock_resp = MagicMock()
        mock_gen.return_value = mock_resp
        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint_url="https://example.azure.com",
            deployment_name="dep",
        )
        result = provider.generate_content(prompt="p", model_name="dep", temperature=0.7)
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs["model_name"] == "dep"
        assert result == mock_resp
