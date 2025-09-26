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

from dataclasses import dataclass
import re
from ._data import Port, DataRegistry, port_to_dict
from ._common import render_structured
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain.tools import StructuredTool
import knime.extension as knext
import logging

from enum import Enum, auto

_logger = logging.getLogger(__name__)


@dataclass
class WorkflowTool:
    """Mirrors the tool class defined in knime-python so we can use type hints here."""

    name: str
    description: str
    parameter_schema: dict
    tool_bytes: bytes
    input_ports: list[Port]
    output_ports: list[Port]


class ExecutionMode(Enum):
    DEBUG = auto()
    DEFAULT = auto()
    DETACHED = auto()


class LangchainToolConverter:
    def __init__(
        self,
        data_registry: DataRegistry,
        ctx,
        execution_mode: ExecutionMode,
        with_view_nodes: bool = False,
    ):
        self._data_registry = data_registry
        self._ctx = ctx
        self._execution_hints = {
            "execution-mode": execution_mode.name,
            "with-view-nodes": str(with_view_nodes),
        }
        self.sanitized_to_original = {}
        self._has_data_tools = False
        self._with_view_nodes = with_view_nodes
        # lazily built reverse mapping (original -> sanitized)
        self._original_to_sanitized_cache = None

    @property
    def has_data_tools(self) -> bool:
        """Returns True if there are tools that require data inputs or outputs."""
        return self._has_data_tools

    def _sanitize_tool_name(self, name: str) -> str:
        """Replaces characters that are not alphanumeric, underscores, or hyphens because
        OpenAI rejects the tools otherwise. Also handles duplicates by appending a suffix."""
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # Handle duplicates by appending a suffix if needed
        base = sanitized
        i = 1
        while (
            sanitized in self.sanitized_to_original
            and self.sanitized_to_original[sanitized] != name
        ):
            sanitized = f"{base}_{i}"
            i += 1
        self.sanitized_to_original[sanitized] = name
        return sanitized

    def desanitize_tool_name(self, sanitized_name: str) -> str:
        return self.sanitized_to_original.get(sanitized_name, sanitized_name)

    def desanitize_tool_names(self, msg: BaseMessage) -> BaseMessage:
        """Desanitizes tool calls in a message by reverting the name back to the original user-provided name."""
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Tool calls can be either a dict or a ToolCall object, so we handle both cases
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_call["name"] = self.desanitize_tool_name(tool_call["name"])
                else:
                    tool_call.name = self.desanitize_tool_name(tool_call.name)
        elif isinstance(msg, ToolMessage) and msg.name:
            msg.name = self.desanitize_tool_name(msg.name)
        return msg

    # --- Sanitization for historical messages ---
    def _original_to_sanitized(self, original_name: str) -> str:
        """Return the sanitized name for an original tool name.

        Supports historical conversations that may reference tools unknown to the
        currently provided tool list (e.g., conversation handover between agents).
        For unknown tools we still sanitize the name (to avoid model/API rejections
        due to invalid characters) and register a reversible mapping so that later
        desanitization yields the original name in user-facing output.
        """
        # Build cache if missing
        if self._original_to_sanitized_cache is None:
            self._original_to_sanitized_cache = {
                original: sanitized
                for sanitized, original in self.sanitized_to_original.items()
            }

        if original_name in self._original_to_sanitized_cache:
            return self._original_to_sanitized_cache[original_name]

        # Unknown original name: create sanitized mapping now.
        sanitized = self._sanitize_tool_name(original_name)
        # Invalidate & rebuild reverse cache entry for this name only (cheap update)
        if self._original_to_sanitized_cache is not None:
            self._original_to_sanitized_cache[original_name] = sanitized
        return sanitized

    def sanitize_tool_names(self, msg: BaseMessage) -> BaseMessage:
        """Sanitize tool names in an incoming (historical) message.

        When conversation history is reused, it contains original tool names (we store
        desanitized names for user readability). Before passing the history to the
        language model, we need to replace those original names with the sanitized
        variants that the current tool set (and thus tool calls) use internally.
        """
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_call["name"] = self._original_to_sanitized(tool_call["name"])
                else:
                    tool_call.name = self._original_to_sanitized(tool_call.name)
        elif isinstance(msg, ToolMessage) and msg.name:
            msg.name = self._original_to_sanitized(msg.name)
        return msg

    def to_langchain_tool(
        self,
        tool: WorkflowTool,
    ) -> StructuredTool:
        sanitized_name = self._sanitize_tool_name(tool.name)
        properties = {}
        if tool.parameter_schema:
            properties["configuration"] = _create_configuration_schema(tool)
        description_parts = {"description": tool.description}

        if tool.input_ports:
            self._has_data_tools = True
            properties["data_inputs"] = _create_input_data_schema(tool)
            params_parser = self._create_data_input_parser(tool)
            description_parts["input_ports"] = list(map(port_to_dict, tool.input_ports))
        else:
            params_parser = self._create_no_data_input_parser(tool)

        if tool.output_ports:
            self._has_data_tools = True
            outputs_processor = self._create_data_output_processor(tool)
            description_parts["output_ports"] = list(
                map(port_to_dict, tool.output_ports)
            )
        else:
            outputs_processor = self._create_no_data_output_processor()

        args_schema = _create_object_schema(
            description="Parameters for the tool.",
            properties=properties,
        )

        # Create the tool function
        def tool_function(**params: dict) -> str:
            try:
                configuration, inputs = params_parser(params)
                message, outputs, view_node_ids = self._ctx._execute_tool(
                    tool, configuration, inputs, self._execution_hints
                )
                return outputs_processor(message, outputs, view_node_ids)
            except Exception as e:
                _logger.exception(e)
                raise

        return StructuredTool.from_function(
            func=tool_function,
            name=sanitized_name,
            description=render_structured(**description_parts),
            args_schema=args_schema,
        )

    def _create_data_output_processor(self, tool: WorkflowTool):
        def _process_outputs_with_data(
            message: str, outputs: list[knext.Table], view_node_ids: list[str]
        ):
            output_references = {}
            for port, output in zip(tool.output_ports, outputs):
                output_reference = self._data_registry.add_table(output, port)
                output_references.update(output_reference)
            return self._append_view_node_ids(
                message
                + "\n\n## Data repository update\n"
                + render_structured(**output_references),
                view_node_ids,
            )

        return _process_outputs_with_data

    def _create_no_data_output_processor(self):
        def _process_outputs_no_data(
            message: str, outputs: list[knext.Table], view_node_ids: list[str]
        ):
            return self._append_view_node_ids(message, view_node_ids)

        return _process_outputs_no_data

    def _create_no_data_input_parser(self, tool: WorkflowTool):
        def _parse_params_no_data(params: dict):
            configuration = params.get("configuration", {})
            _validate_required_fields(
                tool.parameter_schema.keys(), configuration, "configuration parameters"
            )
            return configuration, []

        return _parse_params_no_data

    def _create_data_input_parser(self, tool: WorkflowTool):
        def _parse_params_with_data(params: dict):
            configuration = params.get("configuration", {})
            data_inputs = params.get("data_inputs", {})

            _validate_required_fields(
                tool.parameter_schema.keys(), configuration, "configuration parameters"
            )
            _validate_required_fields(
                [port.name for port in tool.input_ports],
                data_inputs,
                "data inputs for ports",
            )
            inputs = [self._data_registry.get_data(i) for i in data_inputs.values()]
            return configuration, inputs

        return _parse_params_with_data

    def _append_view_node_ids(self, message: str, view_node_ids: list[str]) -> str:
        if view_node_ids:
            return message + "\n\n## View node IDs\n" + ",".join(view_node_ids)
        return message


def _validate_required_fields(required_fields, provided_dict, error_prefix):
    missing = [field for field in required_fields if field not in provided_dict]
    if missing:
        raise knext.InvalidParametersError(
            f"Missing {error_prefix}: {', '.join(missing)}"
        )


def _create_object_schema(description: str, properties: dict):
    return {
        "type": "object",
        "description": description,
        "properties": properties,
        "required": list(properties.keys()),
    }


def _create_configuration_schema(tool: WorkflowTool) -> dict:
    """Creates a parameter schema for the tool based on its input ports."""
    return _create_object_schema(
        description="Configures the tool to perform the task at hand.",
        properties=tool.parameter_schema,
    )


def _create_input_data_schema(tool: WorkflowTool) -> knext.Schema:
    return _create_object_schema(
        description="Data inputs for the tool.",
        properties=_create_input_port_properties(tool.input_ports),
    )


def _create_input_port_properties(ports: list[Port]):
    return {
        port.name: {
            "type": "integer",
            "description": "ID of the data to feed to the port for " + port.description,
        }
        for port in ports
    }
