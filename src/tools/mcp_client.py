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
MCP (Model Context Protocol) client library for connecting to MCP servers via SSE.

This module provides a client implementation for the Model Context Protocol,
supporting Server-Sent Events (SSE) transport for tool discovery and execution.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """Information about a tool available on an MCP server."""

    name: str
    description: str
    input_schema: dict


class MCPClientError(Exception):
    """Base exception for MCP client errors."""

    pass


class MCPConnectionError(MCPClientError):
    """Error connecting to MCP server."""

    pass


class MCPToolExecutionError(MCPClientError):
    """Error executing an MCP tool."""

    pass


class MCPClient:
    """
    Client for connecting to MCP servers via SSE (Server-Sent Events).

    Supports listing available tools and executing them with parameters.
    """

    def __init__(self, server_uri: str, auth: Optional[Any] = None):
        """
        Initialize MCP client.

        Parameters
        ----------
        server_uri : str
            The URI of the MCP server (e.g., "http://localhost:8080/mcp")
        auth : optional
            An ``httpx.Auth`` instance (e.g. ``httpx.BasicAuth``) or any auth
            object accepted by ``httpx.Client``.  ``None`` for unauthenticated
            requests.
        """
        self.server_uri = server_uri
        self._request_id = 0
        self._auth = auth

    def _get_next_request_id(self) -> str:
        """Generate next request ID for JSON-RPC."""
        self._request_id += 1
        return str(self._request_id)

    def _make_jsonrpc_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a JSON-RPC 2.0 request to the MCP server.

        Parameters
        ----------
        method : str
            The JSON-RPC method name
        params : dict, optional
            Parameters for the method

        Returns
        -------
        dict
            The JSON-RPC response result

        Raises
        ------
        MCPConnectionError
            If connection to server fails
        MCPClientError
            If the server returns an error
        """
        try:
            import httpx
        except ImportError:
            raise MCPClientError(
                "httpx library is required for MCP client. Install it with: pip install httpx"
            )

        request_data: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": method,
        }
        if params is not None:
            request_data["params"] = params

        try:
            with httpx.Client(timeout=30.0, auth=self._auth) as client:
                response = client.post(
                    self.server_uri,
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

            response_data = response.json()

            if "error" in response_data:
                error = response_data["error"]
                raise MCPClientError(
                    f"MCP server error: {error.get('message', 'Unknown error')}"
                )

            return response_data.get("result", {})

        except httpx.HTTPError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}")
        except json.JSONDecodeError as e:
            raise MCPClientError(f"Invalid JSON response from MCP server: {e}")

    def list_tools(self) -> List[MCPToolInfo]:
        """
        List all tools available on the MCP server.

        Returns
        -------
        list of MCPToolInfo
            Information about available tools

        Raises
        ------
        MCPConnectionError
            If connection to server fails
        MCPClientError
            If the server returns an error
        """
        _logger.info(f"Listing tools from MCP server: {self.server_uri}")

        try:
            result = self._make_jsonrpc_request("tools/list")

            tools = []
            for tool_data in result.get("tools", []):
                tools.append(
                    MCPToolInfo(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                    )
                )

            _logger.info(f"Found {len(tools)} tools on MCP server")
            return tools

        except Exception as e:
            _logger.error(f"Failed to list tools: {e}")
            raise

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on the MCP server.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute
        arguments : dict
            Arguments to pass to the tool

        Returns
        -------
        Any
            The result from the tool execution

        Raises
        ------
        MCPToolExecutionError
            If tool execution fails
        MCPConnectionError
            If connection to server fails
        """
        _logger.info(f"Calling tool '{tool_name}' with arguments: {arguments}")

        try:
            result = self._make_jsonrpc_request(
                "tools/call",
                params={"name": tool_name, "arguments": arguments},
            )

            _logger.debug(f"Tool '{tool_name}' returned: {result}")
            return result

        except MCPClientError as e:
            raise MCPToolExecutionError(f"Failed to execute tool '{tool_name}': {e}")
        except Exception as e:
            _logger.error(f"Unexpected error calling tool '{tool_name}': {e}")
            raise MCPToolExecutionError(f"Unexpected error: {e}")
