# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------

"""
Blackbox tests for dynamic credential authentication.

These tests verify that fresh tokens are fetched on each HTTP request,
which is essential for supporting expiring JWT tokens from credential ports.
"""

import unittest
import httpx
from models._credential_auth import (
    TokenProvider,
    CredentialPortTokenProvider,
    StaticTokenProvider,
    DynamicAuth,
    create_http_client,
    create_async_http_client,
)


class CountingTokenProvider(TokenProvider):
    """A token provider that returns incrementing tokens for testing.

    Each call to get_token() returns a different value, allowing tests
    to verify that fresh tokens are fetched on each request.
    """

    def __init__(self, auth_schema: str = "Bearer"):
        self._call_count = 0
        self._auth_schema = auth_schema

    def get_token(self) -> str:
        self._call_count += 1
        return f"token_{self._call_count}"

    @property
    def auth_schema(self) -> str:
        return self._auth_schema

    @property
    def call_count(self) -> int:
        return self._call_count


class MockCredentialPortSpec:
    """Mock credential port spec that returns incrementing tokens."""

    def __init__(self, auth_schema: str = "Bearer"):
        self._call_count = 0
        self._auth_schema = auth_schema

    @property
    def auth_parameters(self) -> str:
        self._call_count += 1
        return f"jwt_token_{self._call_count}"

    @property
    def auth_schema(self) -> str:
        return self._auth_schema

    @property
    def call_count(self) -> int:
        return self._call_count


class TestTokenProviders(unittest.TestCase):
    """Tests for TokenProvider implementations."""

    def test_static_token_provider_returns_same_token(self):
        """Static provider should return the same token on every call."""
        provider = StaticTokenProvider("my_api_key")

        self.assertEqual(provider.get_token(), "my_api_key")
        self.assertEqual(provider.get_token(), "my_api_key")
        self.assertEqual(provider.auth_schema, "Bearer")

    def test_static_token_provider_custom_schema(self):
        """Static provider should use custom auth schema."""
        provider = StaticTokenProvider("my_key", auth_schema="Basic")

        self.assertEqual(provider.auth_schema, "Basic")

    def test_credential_port_token_provider_fetches_fresh_tokens(self):
        """Credential port provider should fetch fresh token on each call."""
        mock_spec = MockCredentialPortSpec()
        provider = CredentialPortTokenProvider(mock_spec)

        # Each call should return a different token
        token1 = provider.get_token()
        token2 = provider.get_token()
        token3 = provider.get_token()

        self.assertEqual(token1, "jwt_token_1")
        self.assertEqual(token2, "jwt_token_2")
        self.assertEqual(token3, "jwt_token_3")
        self.assertEqual(mock_spec.call_count, 3)

    def test_credential_port_token_provider_uses_spec_auth_schema(self):
        """Provider should use auth schema from the credential spec."""
        mock_spec = MockCredentialPortSpec(auth_schema="Basic")
        provider = CredentialPortTokenProvider(mock_spec)

        self.assertEqual(provider.auth_schema, "Basic")


class TestDynamicAuth(unittest.TestCase):
    """Tests for DynamicAuth httpx.Auth implementation."""

    def test_sync_auth_flow_fetches_fresh_token_each_request(self):
        """Each request should get a fresh token from the provider."""
        provider = CountingTokenProvider()
        auth = DynamicAuth(provider)

        # Simulate two requests
        request1 = httpx.Request("GET", "https://example.com/api")
        flow1 = auth.sync_auth_flow(request1)
        result1 = next(flow1)

        request2 = httpx.Request("GET", "https://example.com/api")
        flow2 = auth.sync_auth_flow(request2)
        result2 = next(flow2)

        # Each request should have a different token
        self.assertEqual(result1.headers["Authorization"], "Bearer token_1")
        self.assertEqual(result2.headers["Authorization"], "Bearer token_2")
        self.assertEqual(provider.call_count, 2)

    def test_auth_uses_correct_schema(self):
        """Auth should format header with correct schema."""
        provider = CountingTokenProvider(auth_schema="Basic")
        auth = DynamicAuth(provider)

        request = httpx.Request("GET", "https://example.com/api")
        flow = auth.sync_auth_flow(request)
        result = next(flow)

        self.assertEqual(result.headers["Authorization"], "Basic token_1")

    def test_auth_without_schema_for_api_key_header(self):
        """When use_auth_schema=False, token should be used directly."""
        provider = StaticTokenProvider("my_api_key")
        auth = DynamicAuth(provider, header_name="X-API-Key", use_auth_schema=False)

        request = httpx.Request("GET", "https://example.com/api")
        flow = auth.sync_auth_flow(request)
        result = next(flow)

        self.assertEqual(result.headers["X-API-Key"], "my_api_key")
        self.assertNotIn("Authorization", result.headers)

    def test_headers_to_remove_removes_conflicting_headers(self):
        """headers_to_remove should remove specified headers before setting auth."""
        provider = StaticTokenProvider("real_token")
        auth = DynamicAuth(
            provider,
            header_name="Authorization",
            use_auth_schema=True,
            headers_to_remove=["api-key", "X-Old-Auth"],
        )

        # Create request with pre-existing headers (simulating SDK behavior)
        request = httpx.Request(
            "GET",
            "https://example.com/api",
            headers={
                "api-key": "placeholder",
                "X-Old-Auth": "old_value",
                "Content-Type": "application/json",
            },
        )
        flow = auth.sync_auth_flow(request)
        result = next(flow)

        # Conflicting headers should be removed
        self.assertNotIn("api-key", result.headers)
        self.assertNotIn("X-Old-Auth", result.headers)
        # Auth header should be set correctly
        self.assertEqual(result.headers["Authorization"], "Bearer real_token")
        # Unrelated headers should remain
        self.assertEqual(result.headers["Content-Type"], "application/json")

    def test_headers_to_remove_handles_missing_headers(self):
        """headers_to_remove should not fail if headers don't exist."""
        provider = StaticTokenProvider("token")
        auth = DynamicAuth(
            provider,
            header_name="api-key",
            use_auth_schema=False,
            headers_to_remove=["api-key", "Authorization"],
        )

        # Request without the headers to remove
        request = httpx.Request("GET", "https://example.com/api")
        flow = auth.sync_auth_flow(request)
        result = next(flow)

        self.assertEqual(result.headers["api-key"], "token")
        self.assertNotIn("Authorization", result.headers)

    def test_azure_api_key_mode_removes_placeholder_and_sets_api_key(self):
        """Azure API key mode should remove placeholder and set api-key header."""
        provider = StaticTokenProvider("real_api_key")
        auth = DynamicAuth(
            provider,
            header_name="api-key",
            use_auth_schema=False,
            headers_to_remove=["api-key", "Authorization"],
        )

        # Simulate SDK adding placeholder api-key header
        request = httpx.Request(
            "GET",
            "https://myresource.openai.azure.com/openai/deployments/gpt-4",
            headers={"api-key": "placeholder"},
        )
        flow = auth.sync_auth_flow(request)
        result = next(flow)

        self.assertEqual(result.headers["api-key"], "real_api_key")
        self.assertNotIn("Authorization", result.headers)

    def test_azure_entra_mode_removes_placeholder_and_sets_bearer(self):
        """Azure Entra mode should remove placeholder and set Authorization header."""
        provider = CountingTokenProvider(auth_schema="Bearer")
        auth = DynamicAuth(
            provider,
            header_name="Authorization",
            use_auth_schema=True,
            headers_to_remove=["api-key", "Authorization"],
        )

        # Simulate SDK adding placeholder api-key header
        request = httpx.Request(
            "GET",
            "https://myresource.openai.azure.com/openai/deployments/gpt-4",
            headers={"api-key": "placeholder"},
        )
        flow = auth.sync_auth_flow(request)
        result = next(flow)

        self.assertNotIn("api-key", result.headers)
        self.assertEqual(result.headers["Authorization"], "Bearer token_1")


class TestHttpClientCreation(unittest.TestCase):
    """Tests for HTTP client factory functions."""

    def test_create_http_client_with_dynamic_auth(self):
        """Created client should use dynamic auth."""
        provider = CountingTokenProvider()
        client = create_http_client(provider)

        self.assertIsInstance(client, httpx.Client)
        self.assertIsInstance(client.auth, DynamicAuth)

    def test_create_async_http_client_with_dynamic_auth(self):
        """Created async client should use dynamic auth."""
        provider = CountingTokenProvider()
        client = create_async_http_client(provider)

        self.assertIsInstance(client, httpx.AsyncClient)
        self.assertIsInstance(client.auth, DynamicAuth)


class TestAzureHttpClientCreation(unittest.TestCase):
    """Tests for Azure-specific HTTP client creation using headers_to_remove."""

    # Azure uses mutually exclusive auth headers
    AZURE_HEADERS_TO_REMOVE = ["api-key", "Authorization"]

    def test_create_http_client_azure_api_key_mode(self):
        """Azure client in API key mode should use api-key header."""
        provider = StaticTokenProvider("my_azure_key")
        client = create_http_client(
            provider,
            header_name="api-key",
            use_auth_schema=False,
            headers_to_remove=self.AZURE_HEADERS_TO_REMOVE,
        )

        self.assertIsInstance(client, httpx.Client)
        self.assertIsInstance(client.auth, DynamicAuth)

        # Verify the auth is configured correctly
        auth: DynamicAuth = client.auth
        self.assertEqual(auth._header_name, "api-key")
        self.assertFalse(auth._use_auth_schema)
        self.assertIn("api-key", auth._headers_to_remove)
        self.assertIn("Authorization", auth._headers_to_remove)

    def test_create_http_client_azure_entra_mode(self):
        """Azure client in Entra mode should use Authorization header."""
        provider = CountingTokenProvider()
        client = create_http_client(
            provider,
            header_name="Authorization",
            use_auth_schema=True,
            headers_to_remove=self.AZURE_HEADERS_TO_REMOVE,
        )

        self.assertIsInstance(client, httpx.Client)
        self.assertIsInstance(client.auth, DynamicAuth)

        # Verify the auth is configured correctly
        auth: DynamicAuth = client.auth
        self.assertEqual(auth._header_name, "Authorization")
        self.assertTrue(auth._use_auth_schema)
        self.assertIn("api-key", auth._headers_to_remove)
        self.assertIn("Authorization", auth._headers_to_remove)

    def test_create_async_http_client_azure_api_key_mode(self):
        """Azure async client in API key mode should use api-key header."""
        provider = StaticTokenProvider("my_azure_key")
        client = create_async_http_client(
            provider,
            header_name="api-key",
            use_auth_schema=False,
            headers_to_remove=self.AZURE_HEADERS_TO_REMOVE,
        )

        self.assertIsInstance(client, httpx.AsyncClient)
        self.assertIsInstance(client.auth, DynamicAuth)
        self.assertEqual(client.auth._header_name, "api-key")

    def test_create_async_http_client_azure_entra_mode(self):
        """Azure async client in Entra mode should use Authorization header."""
        provider = CountingTokenProvider()
        client = create_async_http_client(
            provider,
            header_name="Authorization",
            use_auth_schema=True,
            headers_to_remove=self.AZURE_HEADERS_TO_REMOVE,
        )

        self.assertIsInstance(client, httpx.AsyncClient)
        self.assertIsInstance(client.auth, DynamicAuth)
        self.assertEqual(client.auth._header_name, "Authorization")

    def test_azure_api_key_mode_request_flow(self):
        """End-to-end test: API key mode should set api-key header correctly."""
        provider = StaticTokenProvider("real_azure_api_key")
        client = create_http_client(
            provider,
            header_name="api-key",
            use_auth_schema=False,
            headers_to_remove=self.AZURE_HEADERS_TO_REMOVE,
        )

        # Simulate a request with placeholder header (as SDK would set)
        request = httpx.Request(
            "POST",
            "https://myresource.openai.azure.com/openai/deployments/gpt-4/chat/completions",
            headers={"api-key": "placeholder"},
        )

        flow = client.auth.sync_auth_flow(request)
        result = next(flow)

        # Should have real key, not placeholder
        self.assertEqual(result.headers["api-key"], "real_azure_api_key")
        self.assertNotIn("Authorization", result.headers)

    def test_azure_entra_mode_request_flow(self):
        """End-to-end test: Entra mode should set Authorization header correctly."""
        mock_spec = MockCredentialPortSpec()
        provider = CredentialPortTokenProvider(mock_spec)
        client = create_http_client(
            provider,
            header_name="Authorization",
            use_auth_schema=True,
            headers_to_remove=self.AZURE_HEADERS_TO_REMOVE,
        )

        # Simulate a request with placeholder header (as SDK would set)
        request = httpx.Request(
            "POST",
            "https://myresource.openai.azure.com/openai/deployments/gpt-4/chat/completions",
            headers={"api-key": "placeholder"},
        )

        flow = client.auth.sync_auth_flow(request)
        result = next(flow)

        # Should have Bearer token, placeholder api-key should be removed
        self.assertNotIn("api-key", result.headers)
        self.assertEqual(result.headers["Authorization"], "Bearer jwt_token_1")


class TestIntegrationWithMockCredentialSpec(unittest.TestCase):
    """Integration tests simulating real credential port usage."""

    def test_multiple_requests_get_fresh_tokens(self):
        """Simulates what happens when a model makes multiple API calls."""
        mock_spec = MockCredentialPortSpec()
        provider = CredentialPortTokenProvider(mock_spec)
        auth = DynamicAuth(provider)

        # Simulate 5 API calls (e.g., during an agent's reasoning loop)
        tokens_used = []
        for _ in range(5):
            request = httpx.Request("POST", "https://api.example.com/chat")
            flow = auth.sync_auth_flow(request)
            result = next(flow)
            tokens_used.append(result.headers["Authorization"])

        # Each request should have gotten a fresh token
        expected = [
            "Bearer jwt_token_1",
            "Bearer jwt_token_2",
            "Bearer jwt_token_3",
            "Bearer jwt_token_4",
            "Bearer jwt_token_5",
        ]
        self.assertEqual(tokens_used, expected)
        self.assertEqual(mock_spec.call_count, 5)


if __name__ == "__main__":
    unittest.main()
