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
MCP Tool Provider node for fetching tools from MCP servers.
"""

import logging
import knime.extension as knext
import pandas as pd

_logger = logging.getLogger(__name__)


def _filter_simple_types(schema: dict) -> dict:
    """
    Filter parameter schema to only include simple types (string, integer, number).

    Parameters
    ----------
    schema : dict
        JSON schema for tool parameters

    Returns
    -------
    dict
        Filtered schema with only simple types
    """
    if not schema or "properties" not in schema:
        return schema

    filtered_props = {}
    for prop_name, prop_def in schema.get("properties", {}).items():
        prop_type = prop_def.get("type", "")
        if prop_type in ["string", "integer", "number"]:
            filtered_props[prop_name] = prop_def
        else:
            _logger.debug(
                f"Skipping parameter '{prop_name}' with unsupported type: {prop_type}"
            )

    required = schema.get("required", [])
    filtered_required = [r for r in required if r in filtered_props]

    return {
        "type": "object",
        "properties": filtered_props,
        "required": filtered_required,
    }


@knext.node(
    name="MCP Tool Provider",
    node_type=knext.NodeType.SOURCE,
    icon_path="icons/generic/brain.png",
    category="/community/llm",
    keywords=["MCP", "Model Context Protocol", "Tool", "Agent", "GenAI"],
)
@knext.output_table(
    name="Tools",
    description="Table containing MCP tools from the server",
)
class MCPToolProvider:
    """
    Fetches tools from an MCP (Model Context Protocol) server.

    This node connects to an MCP server via Server-Sent Events (SSE) and retrieves
    all available tools. Each tool is converted to a row in the output table.

    **Note:** This is a prototype implementation that only supports simple parameter
    types (string, integer, number) and does not handle complex data structures.

    **MCP Server Configuration:**
    Provide the URL of your MCP server endpoint (e.g., "http://localhost:8080/mcp").
    The server must support the Model Context Protocol with tools/list capability.
    """

    server_url = knext.StringParameter(
        label="Server URL",
        description="The URL of the MCP server endpoint (e.g., 'http://localhost:8080/mcp')",
        default_value="",
    )

    def configure(self, ctx: knext.ConfigurationContext) -> knext.Schema:
        """
        Configure the output schema.

        Returns a schema with a single column of MCPTool type.
        """
        if not self.server_url:
            raise knext.InvalidParametersError("Server URL must not be empty")

        # Import MCPTool type
        import knime.types.tool as ktt

        return knext.Schema.from_columns(
            [knext.Column(knext.logical(ktt.Tool), "MCP Tools")]
        )

    def execute(self, ctx: knext.ExecutionContext) -> knext.Table:
        """
        Execute the node by connecting to the MCP server and fetching tools.

        Returns
        -------
        knext.Table
            Table with one row per MCP tool
        """
        from .mcp_client import MCPClient, MCPConnectionError, MCPClientError
        import knime.types.tool as ktt

        _logger.info(f"Connecting to MCP server: {self.server_url}")

        try:
            client = MCPClient(self.server_url)
            tool_infos = client.list_tools()

            _logger.info(f"Found {len(tool_infos)} tools from MCP server")

            # Convert each MCP tool info to Tool object
            mcp_tools = []
            for tool_info in tool_infos:
                # Filter to only include simple types
                filtered_schema = _filter_simple_types(tool_info.input_schema)

                mcp_tool = ktt.Tool.create_mcp_tool(
                    name=tool_info.name,
                    description=tool_info.description,
                    parameter_schema=filtered_schema,
                    server_uri=self.server_url,
                    tool_name=tool_info.name,
                )
                mcp_tools.append(mcp_tool)

            # Create output table with MCP tools
            if not mcp_tools:
                _logger.warning("No tools found on MCP server")
                # Return empty table with correct schema
                df = pd.DataFrame(
                    {
                        "MCP Tools": pd.Series(
                            [], dtype=knext.logical(ktt.Tool).to_pandas()
                        )
                    }
                )
            else:
                df = pd.DataFrame({"MCP Tools": mcp_tools})

            return knext.Table.from_pandas(df)

        except MCPConnectionError as e:
            raise knext.InvalidParametersError(
                f"Failed to connect to MCP server: {e}"
            )
        except MCPClientError as e:
            raise RuntimeError(f"MCP client error: {e}")
        except Exception as e:
            _logger.error(f"Unexpected error: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error fetching MCP tools: {e}")
