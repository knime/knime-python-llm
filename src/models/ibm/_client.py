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
Lightweight HTTP client for IBM watsonx.ai REST API.

This module provides direct REST API access to watsonx.ai without requiring
the heavy ibm-watsonx-ai SDK or langchain-ibm packages.
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .._credential_auth import TokenProvider, create_http_client


# IBM IAM token endpoint
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# API version for watsonx.ai
API_VERSION = "2024-05-31"

# Mapping of ML (Machine Learning) API base URLs to their corresponding WDP (Watson Data Platform) API base URLs.
# - ML API: Used for model inference, embeddings, and listing foundation models
#   Reference: https://cloud.ibm.com/apidocs/watsonx-ai
# - WDP API: Used for listing projects and deployment spaces
#   Reference: https://cloud.ibm.com/apidocs/watson-data-api
_ML_TO_WDP_URL_MAP = {
    "https://us-south.ml.cloud.ibm.com": "https://api.dataplatform.cloud.ibm.com",
    "https://eu-de.ml.cloud.ibm.com": "https://api.eu-de.dataplatform.cloud.ibm.com",
    "https://eu-gb.ml.cloud.ibm.com": "https://api.eu-gb.dataplatform.cloud.ibm.com",
    "https://jp-tok.ml.cloud.ibm.com": "https://api.jp-tok.dataplatform.cloud.ibm.com",
    "https://au-syd.ml.cloud.ibm.com": "https://api.au-syd.dai.cloud.ibm.com",
    "https://ca-tor.ml.cloud.ibm.com": "https://api.ca-tor.dai.cloud.ibm.com",
    "https://ap-south-1.aws.wxai.ibm.com": "https://api.ap-south-1.aws.data.ibm.com",
}


def _get_wdp_url(ml_base_url: str) -> str:
    """Get the Watson Data Platform API URL corresponding to an ML API URL."""
    # Normalize URL (remove trailing slash)
    ml_base_url = ml_base_url.rstrip("/")

    if ml_base_url in _ML_TO_WDP_URL_MAP:
        return _ML_TO_WDP_URL_MAP[ml_base_url]

    # Fallback: try to construct WDP URL from ML URL
    # e.g., https://eu-de.ml.cloud.ibm.com -> https://api.eu-de.dataplatform.cloud.ibm.com
    if ".ml.cloud.ibm.com" in ml_base_url:
        region = ml_base_url.replace("https://", "").replace(".ml.cloud.ibm.com", "")
        return f"https://api.{region}.dataplatform.cloud.ibm.com"

    raise ValueError(
        f"Unknown ML API base URL: {ml_base_url}. Cannot determine WDP API URL."
    )


@dataclass
class _IAMToken:
    """Cached IAM token with expiration tracking."""

    access_token: str
    expires_at: float  # Unix timestamp

    def is_expired(self, buffer_seconds: int = 60) -> bool:
        """Check if token is expired or about to expire."""
        return time.time() >= (self.expires_at - buffer_seconds)


class IBMIAMTokenProvider(TokenProvider):
    """
    Token provider for IBM Cloud IAM authentication.

    Exchanges an IBM Cloud API key for a short-lived IAM bearer token,
    caching and automatically refreshing it when expired.
    """

    def __init__(self, api_key: str, timeout: float = 30.0):
        """
        Initialize the IAM token provider.

        Args:
            api_key: IBM Cloud API key
            timeout: Timeout for token refresh requests
        """
        self._api_key = api_key
        self._timeout = timeout
        self._token: _IAMToken | None = None

    def get_token(self) -> str:
        """
        Get a valid IAM token, refreshing if necessary.

        This method is called on each HTTP request by the DynamicAuth handler.
        """
        if self._token is None or self._token.is_expired():
            self._refresh_token()
        return self._token.access_token

    def _refresh_token(self) -> None:
        """Obtain a new IAM token from the IBM Cloud IAM endpoint."""
        # Use a simple httpx request for token refresh (not the pooled client)
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                IAM_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": self._api_key,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to obtain IAM token: {response.status_code} - {response.text}"
                )

            data = response.json()
            expires_in = data.get("expires_in", 3600)
            self._token = _IAMToken(
                access_token=data["access_token"],
                expires_at=time.time() + expires_in,
            )


class WatsonxClient:
    """
    Lightweight HTTP client for IBM watsonx.ai REST API.

    This client provides direct access to watsonx.ai APIs without requiring
    the heavy ibm-watsonx-ai SDK. It handles:
    - IAM token authentication with automatic refresh (via IBMIAMTokenProvider)
    - Listing foundation models (chat and embedding)
    - Listing projects and spaces
    - Text generation (chat)
    - Embeddings generation

    Uses connection pooling via httpx.Client for efficient request handling.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 120.0,
    ):
        """
        Initialize the watsonx.ai client.

        Args:
            api_key: IBM Cloud API key
            base_url: Base URL for the ML API (e.g., https://eu-de.ml.cloud.ibm.com)
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._wdp_url = _get_wdp_url(self._base_url)
        self._timeout = timeout

        # Create token provider and pooled HTTP client
        token_provider = IBMIAMTokenProvider(api_key, timeout=min(timeout, 30.0))
        self._http_client = create_http_client(
            token_provider,
            timeout=httpx.Timeout(timeout),
        )

    def _get_headers(self) -> dict[str, str]:
        """Get common headers for API requests (excluding auth, handled by client)."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _ml_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a request to the ML API."""
        url = f"{self._base_url}{path}"
        params = params or {}
        params["version"] = API_VERSION

        response = self._http_client.request(
            method=method,
            url=url,
            headers=self._get_headers(),
            params=params,
            json=json_data,
        )

        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"watsonx.ai API request failed: {response.status_code} - {response.text}"
            )

        return response.json()

    def _wdp_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a request to the Watson Data Platform API."""
        url = f"{self._wdp_url}{path}"

        response = self._http_client.request(
            method=method,
            url=url,
            headers=self._get_headers(),
            params=params,
        )

        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Watson Data Platform API request failed: {response.status_code} - {response.text}"
            )

        return response.json()

    # =========================================================================
    # Foundation Models API
    # =========================================================================

    def list_foundation_models(
        self,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        List all available foundation models.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of model specifications
        """
        params = {"limit": limit}
        data = self._ml_request("GET", "/ml/v1/foundation_model_specs", params=params)
        return data.get("resources", [])

    def _model_has_function(self, model: dict[str, Any], function_id: str) -> bool:
        """Check if a model has a specific function capability."""
        functions = model.get("functions", [])
        return any(f.get("id") == function_id for f in functions)

    def list_chat_models(self) -> list[str]:
        """List available chat model IDs (models with 'text_chat' function)."""
        models = self.list_foundation_models()
        return [
            m.get("model_id")
            for m in models
            if m.get("model_id") and self._model_has_function(m, "text_chat")
        ]

    def list_embedding_models(self) -> list[str]:
        """List available embedding model IDs (models with 'embedding' function)."""
        models = self.list_foundation_models()
        return [
            m.get("model_id")
            for m in models
            if m.get("model_id") and self._model_has_function(m, "embedding")
        ]

    def list_function_calling_models(self) -> list[str]:
        """
        List models that support function/tool calling.

        Note: This checks for models that have the 'text_chat' function,
        as tool calling is typically associated with chat models.
        The actual tool calling capability may vary by model.
        """
        # Currently, watsonx.ai doesn't expose a specific function ID for tool calling
        # Chat models that support tools are typically the instruction-tuned models
        # For now, we return chat models - actual tool support is model-specific
        return self.list_chat_models()

    # =========================================================================
    # Projects and Spaces API
    # =========================================================================

    def list_projects(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        List available projects.

        Returns:
            List of project resources with 'name' and 'id' keys
        """
        data = self._wdp_request("GET", "/v2/projects", params={"limit": limit})
        resources = data.get("resources", [])

        # Normalize the response format
        result = []
        for proj in resources:
            entity = proj.get("entity", {})
            metadata = proj.get("metadata", {})
            result.append(
                {
                    "name": entity.get("name", proj.get("name", "")),
                    "id": metadata.get("guid", proj.get("id", "")),
                }
            )

        return result

    def list_spaces(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        List available deployment spaces.

        Returns:
            List of space resources with 'name' and 'id' keys
        """
        import html

        data = self._wdp_request("GET", "/v2/spaces", params={"limit": limit})
        resources = data.get("resources", [])

        # Normalize the response format
        result = []
        for space in resources:
            entity = space.get("entity", {})
            metadata = space.get("metadata", {})
            # Space names may be HTML-encoded in the API response
            name = html.unescape(entity.get("name", space.get("name", "")))
            result.append(
                {
                    "name": name,
                    "id": metadata.get("id", space.get("id", "")),
                }
            )

        return result

    def get_project_id_by_name(self, name: str) -> str | None:
        """Get project ID by name."""
        projects = self.list_projects()
        for proj in projects:
            if proj.get("name") == name:
                return proj.get("id")
        return None

    def get_space_id_by_name(self, name: str) -> str | None:
        """Get space ID by name."""
        spaces = self.list_spaces()
        for space in spaces:
            if space.get("name") == name:
                return space.get("id")
        return None

    # =========================================================================
    # Text Generation (Chat) API
    # =========================================================================

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        project_id: str | None = None,
        space_id: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate a chat completion.

        Args:
            model_id: The model to use
            messages: List of message dicts with 'role' and 'content' keys
            project_id: Project ID (required if space_id not provided)
            space_id: Space ID (required if project_id not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            tools: Optional list of tool definitions for function calling
            **kwargs: Additional parameters to pass to the API

        Returns:
            API response with generated content
        """
        if not project_id and not space_id:
            raise ValueError("Either project_id or space_id must be provided")

        payload: dict[str, Any] = {
            "model_id": model_id,
            "messages": messages,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                **kwargs,
            },
        }

        if project_id:
            payload["project_id"] = project_id
        if space_id:
            payload["space_id"] = space_id
        if tools:
            payload["tools"] = tools

        return self._ml_request("POST", "/ml/v1/text/chat", json_data=payload)

    # =========================================================================
    # Embeddings API
    # =========================================================================

    def embeddings(
        self,
        model_id: str,
        inputs: list[str],
        project_id: str | None = None,
        space_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate embeddings for input texts.

        Args:
            model_id: The embedding model to use
            inputs: List of texts to embed
            project_id: Project ID (required if space_id not provided)
            space_id: Space ID (required if project_id not provided)

        Returns:
            API response with embeddings
        """
        if not project_id and not space_id:
            raise ValueError("Either project_id or space_id must be provided")

        payload = {
            "model_id": model_id,
            "inputs": inputs,
        }

        if project_id:
            payload["project_id"] = project_id
        if space_id:
            payload["space_id"] = space_id

        return self._ml_request("POST", "/ml/v1/text/embeddings", json_data=payload)

    def embed_documents(
        self,
        model_id: str,
        texts: list[str],
        project_id: str | None = None,
        space_id: str | None = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        This is a convenience method that returns just the embedding vectors.

        Args:
            model_id: The embedding model to use
            texts: List of texts to embed
            project_id: Project ID (required if space_id not provided)
            space_id: Space ID (required if project_id not provided)

        Returns:
            List of embedding vectors
        """
        response = self.embeddings(
            model_id=model_id,
            inputs=texts,
            project_id=project_id,
            space_id=space_id,
        )

        results = response.get("results", [])
        return [r.get("embedding", []) for r in results]

    def embed_query(
        self,
        model_id: str,
        text: str,
        project_id: str | None = None,
        space_id: str | None = None,
    ) -> list[float]:
        """
        Generate embedding for a single query.

        This is a convenience method that returns just the embedding vector.

        Args:
            model_id: The embedding model to use
            text: The text to embed
            project_id: Project ID (required if space_id not provided)
            space_id: Space ID (required if project_id not provided)

        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents(
            model_id=model_id,
            texts=[text],
            project_id=project_id,
            space_id=space_id,
        )
        return embeddings[0] if embeddings else []
