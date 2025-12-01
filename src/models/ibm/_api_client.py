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
REST API client for IBM watsonx.ai.

This module provides direct REST API access to IBM watsonx.ai services,
replacing the ibm_watsonx_ai SDK to avoid complex dependency management.
"""

import time
from dataclasses import dataclass
from typing import Optional
import requests


# IBM Cloud IAM endpoint for token exchange
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# API version to use for watsonx.ai endpoints
API_VERSION = "2024-05-31"


@dataclass
class IAMToken:
    """Represents an IBM Cloud IAM access token with expiration."""

    access_token: str
    expires_at: float  # Unix timestamp when the token expires

    def is_expired(self) -> bool:
        """Check if the token is expired (with 60s buffer)."""
        return time.time() >= (self.expires_at - 60)


class WatsonxAPIClient:
    """
    REST API client for IBM watsonx.ai.

    Handles authentication via IAM tokens and provides methods to interact
    with the watsonx.ai Foundation Models API.
    """

    def __init__(self, api_key: str, base_url: str):
        """
        Initialize the API client.

        Args:
            api_key: IBM Cloud API key for authentication
            base_url: Base URL for the watsonx.ai API (e.g., https://eu-de.ml.cloud.ibm.com)
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._token: Optional[IAMToken] = None

    def _get_access_token(self) -> str:
        """
        Get a valid IAM access token, refreshing if necessary.

        Returns:
            Valid access token string
        """
        if self._token is None or self._token.is_expired():
            self._token = self._request_new_token()
        return self._token.access_token

    def _request_new_token(self) -> IAMToken:
        """
        Request a new IAM access token using the API key.

        Returns:
            New IAMToken with access token and expiration
        """
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self._api_key,
        }

        response = requests.post(IAM_TOKEN_URL, headers=headers, data=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        # IBM returns expires_in in seconds, convert to absolute timestamp
        expires_at = time.time() + result.get("expires_in", 3600)

        return IAMToken(
            access_token=result["access_token"],
            expires_at=expires_at,
        )

    def _get_headers(self) -> dict:
        """Get headers with current authentication token."""
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        timeout: int = 120,
    ) -> dict:
        """
        Make an authenticated request to the watsonx.ai API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (without base URL)
            params: Query parameters
            json_data: JSON body for POST requests
            timeout: Request timeout in seconds

        Returns:
            JSON response as dictionary
        """
        url = f"{self._base_url}{endpoint}"

        # Always include API version in params
        if params is None:
            params = {}
        params["version"] = API_VERSION

        response = requests.request(
            method=method,
            url=url,
            headers=self._get_headers(),
            params=params,
            json=json_data,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Foundation Model Specs
    # =========================================================================

    def get_foundation_model_specs(
        self, filters: Optional[str] = None
    ) -> dict:
        """
        Get available foundation model specifications.

        Args:
            filters: Optional filter string (e.g., "function_text_chat")

        Returns:
            Dictionary containing list of model specs in "resources" key
        """
        params = {}
        if filters:
            params["filters"] = filters

        return self._make_request("GET", "/ml/v1/foundation_model_specs", params=params)

    def get_chat_model_specs(self) -> dict:
        """Get specifications for chat-capable models."""
        return self.get_foundation_model_specs(filters="function_text_chat")

    def get_embeddings_model_specs(self) -> dict:
        """Get specifications for embedding models."""
        return self.get_foundation_model_specs(filters="function_embedding")

    def get_chat_function_calling_model_specs(self) -> dict:
        """Get specifications for models that support function/tool calling."""
        # Models with function calling support for chat
        return self.get_foundation_model_specs(
            filters="function_text_chat,function_tools"
        )

    # =========================================================================
    # Projects and Spaces
    # =========================================================================

    def list_projects(self) -> dict:
        """
        List available Watson Studio projects.

        Note: Projects API uses a different base URL (dataplatform.cloud.ibm.com)
        """
        # Projects are managed through the Data Platform API, not the ML API
        # We need to use a different endpoint
        headers = self._get_headers()

        # The projects API is at a fixed URL
        url = "https://api.dataplatform.cloud.ibm.com/v2/projects"
        params = {"version": API_VERSION}

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Convert to the format expected by the existing code (NAME and ID lists)
        projects = result.get("resources", [])
        return {
            "NAME": [p.get("entity", {}).get("name", "") for p in projects],
            "ID": [p.get("metadata", {}).get("guid", "") for p in projects],
        }

    def list_spaces(self) -> dict:
        """
        List available Watson Studio deployment spaces.
        """
        # Spaces can be listed through the ML API
        result = self._make_request("GET", "/v2/spaces")

        # Convert to the format expected by the existing code (NAME and ID lists)
        spaces = result.get("resources", [])
        return {
            "NAME": [s.get("entity", {}).get("name", "") for s in spaces],
            "ID": [s.get("metadata", {}).get("id", "") for s in spaces],
        }

    # =========================================================================
    # Text Generation (Chat)
    # =========================================================================

    def text_chat(
        self,
        model_id: str,
        messages: list[dict],
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """
        Send a chat completion request.

        Args:
            model_id: The model to use for chat
            messages: List of message dicts with "role" and "content"
            project_id: Watson Studio project ID (mutually exclusive with space_id)
            space_id: Deployment space ID (mutually exclusive with project_id)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            tools: Optional list of tool definitions for function calling

        Returns:
            Chat completion response
        """
        payload = {
            "model_id": model_id,
            "messages": messages,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
        }

        if project_id:
            payload["project_id"] = project_id
        elif space_id:
            payload["space_id"] = space_id

        if tools:
            payload["tools"] = tools

        return self._make_request("POST", "/ml/v1/text/chat", json_data=payload)

    def text_chat_stream(
        self,
        model_id: str,
        messages: list[dict],
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        tools: Optional[list[dict]] = None,
    ):
        """
        Send a streaming chat completion request.

        Yields:
            Chunks of the chat completion response
        """
        payload = {
            "model_id": model_id,
            "messages": messages,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
        }

        if project_id:
            payload["project_id"] = project_id
        elif space_id:
            payload["space_id"] = space_id

        if tools:
            payload["tools"] = tools

        url = f"{self._base_url}/ml/v1/text/chat_stream"
        params = {"version": API_VERSION}

        with requests.post(
            url,
            headers=self._get_headers(),
            params=params,
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    # SSE format: "data: {...}"
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data:"):
                        import json

                        yield json.loads(line_str[5:].strip())

    # =========================================================================
    # Text Embeddings
    # =========================================================================

    def text_embeddings(
        self,
        model_id: str,
        inputs: list[str],
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
    ) -> dict:
        """
        Generate text embeddings.

        Args:
            model_id: The embedding model to use
            inputs: List of text strings to embed
            project_id: Watson Studio project ID (mutually exclusive with space_id)
            space_id: Deployment space ID (mutually exclusive with project_id)

        Returns:
            Embeddings response with results
        """
        payload = {
            "model_id": model_id,
            "inputs": inputs,
        }

        if project_id:
            payload["project_id"] = project_id
        elif space_id:
            payload["space_id"] = space_id

        return self._make_request("POST", "/ml/v1/text/embeddings", json_data=payload)

