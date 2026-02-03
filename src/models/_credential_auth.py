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
Unified authentication module for dynamic credential refresh.

This module provides a unified way to handle authentication for HTTP clients,
supporting both static API keys and dynamic credential ports that may provide
expiring JWT tokens.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator, List, Optional
import httpx


class TokenProvider(ABC):
    """Abstract interface for getting authentication tokens/credentials.

    Implementations of this interface are called on each HTTP request to get
    fresh credentials, enabling support for expiring JWT tokens.
    """

    @abstractmethod
    def get_token(self) -> str:
        """Get the current token/API key.

        This method is called on each HTTP request to ensure fresh tokens
        are used even if the original token has expired.
        """
        pass

    @property
    def auth_schema(self) -> str:
        """Get the auth schema (e.g., 'Bearer', 'Basic'). Default is 'Bearer'."""
        return "Bearer"


class CredentialPortTokenProvider(TokenProvider):
    """Token provider that wraps a CredentialPortObjectSpec.

    This provider fetches tokens dynamically from the credential port spec,
    which handles token refresh automatically for expiring credentials.
    """

    def __init__(self, credential_spec):
        """Initialize with a credential port object spec.

        Args:
            credential_spec: A CredentialPortObjectSpec or HubAuthenticationPortObjectSpec
                that provides auth_parameters and optionally auth_schema.
        """
        self._credential_spec = credential_spec

    def get_token(self) -> str:
        """Get the current token from the credential spec.

        The credential spec handles token refresh internally, so each call
        may return a fresh token if the previous one has expired.
        """
        return self._credential_spec.auth_parameters

    @property
    def auth_schema(self) -> str:
        """Get the auth schema from the credential spec."""
        return getattr(self._credential_spec, "auth_schema", "Bearer")


class StaticTokenProvider(TokenProvider):
    """Token provider for static API keys that don't expire."""

    def __init__(self, token: str, auth_schema: str = "Bearer"):
        """Initialize with a static token.

        Args:
            token: The static API key or token.
            auth_schema: The authentication schema (default: 'Bearer').
        """
        self._token = token
        self._auth_schema = auth_schema

    def get_token(self) -> str:
        """Return the static token."""
        return self._token

    @property
    def auth_schema(self) -> str:
        """Return the configured auth schema."""
        return self._auth_schema


class DynamicAuth(httpx.Auth):
    """Unified httpx.Auth implementation supporting dynamic credential refresh.

    This auth class fetches fresh tokens from a TokenProvider on each request,
    enabling support for expiring JWT tokens from credential ports.

    Supports both sync and async HTTP clients.
    """

    def __init__(
        self,
        token_provider: TokenProvider,
        header_name: str = "Authorization",
        use_auth_schema: bool = True,
        headers_to_remove: Optional[List[str]] = None,
    ):
        """Initialize the dynamic auth handler.

        Args:
            token_provider: Provider that supplies fresh tokens on each request.
            header_name: HTTP header name for the auth token (default: 'Authorization').
            use_auth_schema: If True and header_name is 'Authorization', format as
                '{schema} {token}'. If False, use token directly (default: True).
            headers_to_remove: List of header names to remove before setting auth header.
                This is useful for Azure OpenAI where we need to remove any placeholder
                'api-key' headers set by the SDK before adding our auth header.
        """
        self._token_provider = token_provider
        self._header_name = header_name
        self._use_auth_schema = use_auth_schema
        self._headers_to_remove = headers_to_remove or []

    def _get_header_value(self) -> str:
        """Get the formatted header value with a fresh token."""
        token = self._token_provider.get_token()
        if self._use_auth_schema and self._header_name == "Authorization":
            return f"{self._token_provider.auth_schema} {token}"
        return token

    def _apply_auth(self, request: httpx.Request) -> None:
        """Apply authentication to the request, removing conflicting headers first."""
        # Remove any headers that might conflict (e.g., placeholder api-key from SDK)
        for header in self._headers_to_remove:
            if header in request.headers:
                del request.headers[header]

        # Set the auth header
        header_value = self._get_header_value()
        request.headers[self._header_name] = header_value

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Sync auth flow that adds fresh credentials to each request."""
        self._apply_auth(request)
        yield request

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Async auth flow that adds fresh credentials to each request."""
        self._apply_auth(request)
        yield request


def create_http_client(
    token_provider: TokenProvider,
    header_name: str = "Authorization",
    use_auth_schema: bool = True,
    headers_to_remove: Optional[List[str]] = None,
    **kwargs,
) -> httpx.Client:
    """Create a sync httpx client with dynamic authentication.

    Args:
        token_provider: Provider that supplies fresh tokens on each request.
        header_name: HTTP header name for the auth token (default: 'Authorization').
        use_auth_schema: If True, format Authorization header as '{schema} {token}'.
        headers_to_remove: List of header names to remove before setting auth header.
            Useful for Azure OpenAI where placeholder headers need to be removed.
        **kwargs: Additional arguments passed to httpx.Client.

    Returns:
        An httpx.Client configured with dynamic authentication.
    """
    auth = DynamicAuth(token_provider, header_name, use_auth_schema, headers_to_remove)
    return httpx.Client(auth=auth, **kwargs)


def create_async_http_client(
    token_provider: TokenProvider,
    header_name: str = "Authorization",
    use_auth_schema: bool = True,
    headers_to_remove: Optional[List[str]] = None,
    **kwargs,
) -> httpx.AsyncClient:
    """Create an async httpx client with dynamic authentication.

    Args:
        token_provider: Provider that supplies fresh tokens on each request.
        header_name: HTTP header name for the auth token (default: 'Authorization').
        use_auth_schema: If True, format Authorization header as '{schema} {token}'.
        headers_to_remove: List of header names to remove before setting auth header.
            Useful for Azure OpenAI where placeholder headers need to be removed.
        **kwargs: Additional arguments passed to httpx.AsyncClient.

    Returns:
        An httpx.AsyncClient configured with dynamic authentication.
    """
    auth = DynamicAuth(token_provider, header_name, use_auth_schema, headers_to_remove)
    return httpx.AsyncClient(auth=auth, **kwargs)
