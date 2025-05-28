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
from typing import Callable, Optional, Sequence
import knime.extension as knext

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool
import pandas as pd


import json
from langchain.tools import StructuredTool
from langchain_core.messages import AIMessage, BaseMessage

import logging
import re

_logger = logging.getLogger(__name__)


@dataclass
class DataItem:
    llm_representation: str
    data: knext.Table


@dataclass
class Port:
    name: str
    description: str
    type: str
    spec: Optional[str]


def _empty_table():
    """Returns an empty knext.Table."""
    # Assuming knext.Table() creates an empty table, adjust as necessary
    return knext.Table.from_pandas(pd.DataFrame())


@dataclass
class WorkflowTool:
    """Mirrors the tool class defined in knime-python so we can use type hints here."""

    name: str
    description: str
    parameter_schema: dict
    tool_bytes: bytes
    input_ports: list[Port]
    output_ports: list[Port]


class DataRegistry:
    def __init__(self, initial_tables: Sequence[knext.Table] = None):
        self._data: list[DataItem] = []
        if initial_tables:
            for table in initial_tables:
                self.add_table(table)

    def add_table(self, table: knext.Table) -> dict:
        table_representation = self._table_representation(table)
        self._data.append(DataItem(table_representation, table))
        return {len(self._data) - 1: table_representation}

    def get_data(self, index: int) -> knext.Table:
        if index < 0 or index >= len(self._data):
            raise IndexError("Index out of range")
        return self._data[index].data

    def get_last_tables(self, num_tables: int) -> list[knext.Table]:
        """Returns the last `num_tables` tables added to the registry."""
        if num_tables <= 0:
            return []
        tables = [data_item.data for data_item in self._data[-num_tables:]]
        if len(tables) < num_tables:
            empty_table = _empty_table()
            tables = tables + [empty_table] * (num_tables - len(tables))
        return tables

    def create_port_description(self, port: Port) -> dict:
        return {
            "name": port.name,
            "description": port.description,
            "type": port.type,
            "spec": port.spec,
        }

    def llm_representation(self) -> dict:
        return {id: data.llm_representation for id, data in enumerate(self._data)}

    def _column_representation(self, column: knext.Column) -> str:
        return f"({column.name}, {str(column.ktype)})"

    def _table_representation(self, table: knext.Table) -> str:
        return f"[{', '.join(map(self._column_representation, table.schema))}]"


class LangchainToolConverter:
    def __init__(
        self, data_registry: DataRegistry, ctx, message_renderer: Callable, debug: bool
    ):
        self._data_registry = data_registry
        self._ctx = ctx
        self._message_renderer = message_renderer
        self._debug = debug
        self.sanitized_to_original = {}

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

    def desanitize_tool_calls(self, msg: BaseMessage) -> BaseMessage:
        """Desanitizes tool calls in a message by reverting the name back to the original user-provided name."""
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Tool calls can be either a dict or a ToolCall object, so we handle both cases
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_call["name"] = self.desanitize_tool_name(tool_call["name"])
                else:
                    tool_call.name = self.desanitize_tool_name(tool_call.name)
        return msg

    def to_langchain_tool(
        self,
        tool: WorkflowTool,
    ) -> StructuredTool:
        sanitized_name = self._sanitize_tool_name(tool.name)
        if tool.input_ports or tool.output_ports:
            return self._to_langchain_tool_with_data(tool, sanitized_name)
        else:
            return self._to_langchain_tool_without_data(tool, sanitized_name)

    def _validate_required_fields(self, required_fields, provided_dict, error_prefix):
        missing = [field for field in required_fields if field not in provided_dict]
        if missing:
            raise knext.InvalidParametersError(
                f"Missing {error_prefix}: {', '.join(missing)}"
            )

    def _to_langchain_tool_without_data(
        self, tool: WorkflowTool, sanitized_name: str
    ) -> StructuredTool:
        args_schema = {
            "type": "object",
            "properties": tool.parameter_schema,
            "required": list(tool.parameter_schema.keys()),
        }

        def func(**params):
            try:
                self._validate_required_fields(
                    tool.parameter_schema.keys(), params, "configuration parameters"
                )
                return self._ctx._execute_tool(tool, params, [], self._debug)[0]
            except Exception as e:
                _logger.exception(e)
                raise

        return StructuredTool.from_function(
            func=func,
            name=sanitized_name,
            description=tool.description,
            args_schema=args_schema,
        )

    def _to_langchain_tool_with_data(
        self,
        tool: WorkflowTool,
        sanitized_name: str,
    ) -> StructuredTool:
        args_schema = {
            "type": "object",
            "properties": {
                "configuration": {
                    "type": "object",
                    "properties": tool.parameter_schema,
                    "description": "Configures the tool to perform the task at hand.",
                    "required": list(tool.parameter_schema.keys()),
                },
                "data_inputs": self._create_input_data_schema(tool),
            },
            "required": ["parameters", "data_inputs"],
        }

        input_ports = tool.input_ports if tool.input_ports else []
        output_ports = tool.output_ports if tool.output_ports else []

        description_with_data_info = self._message_renderer(
            description=tool.description,
            input_ports=list(
                map(self._data_registry.create_port_description, input_ports)
            ),
            output_ports=list(
                map(self._data_registry.create_port_description, output_ports)
            ),
        )

        def func(configuration: dict = None, data_inputs: dict = None) -> str:
            configuration = configuration or {}
            data_inputs = data_inputs or {}

            # Unified configuration validation
            self._validate_required_fields(
                tool.parameter_schema.keys(), configuration, "configuration parameters"
            )
            # Unified data input validation
            self._validate_required_fields(
                [port.name for port in tool.input_ports],
                data_inputs,
                "data inputs for ports",
            )

            try:
                inputs = [self._data_registry.get_data(i) for i in data_inputs.values()]
                message, outputs = self._ctx._execute_tool(
                    tool, configuration, inputs, self._debug
                )
                output_references = {}
                for output in outputs:
                    output_reference = self._data_registry.add_table(output)
                    output_references.update(output_reference)

                return self._message_renderer(
                    message=message, outputs=output_references
                )
            except Exception as e:
                _logger.exception(e)
                raise

        return StructuredTool.from_function(
            func=func,
            name=sanitized_name,
            description=description_with_data_info,
            args_schema=args_schema,
        )

    def _create_input_data_schema(self, tool: WorkflowTool) -> knext.Schema:
        return {
            "type": "object",
            "description": "The input data the tool requires for the task at hand.",
            "properties": self._create_input_port_properties(tool.input_ports),
            "required": [port.name for port in tool.input_ports],
        }

    def _create_input_port_properties(self, ports: list[Port]):
        return {
            port.name: {
                "type": "integer",
                "description": "ID of the data to feed to the port for "
                + port.description,
            }
            for port in ports
        }


def render_message_as_json(**kwargs) -> str:
    return json.dumps(kwargs)


class ChatAgentPrompterDataService:
    def __init__(self, agent_graph, data_registry: DataRegistry, show_tool_messages: bool):
        self._agent_graph = agent_graph
        self._data_registry = data_registry
        self._messages = [
            {
                "role": "user",
                "content": render_message_as_json(
                    data=data_registry.llm_representation()
                ),
            }
        ]
        self._show_tool_messages = show_tool_messages

    def init(self):
        pass

    def post_user_message(self, user_message: str):
        self._messages.append({"role": "user", "content": user_message})
        final_state = self._agent_graph.invoke({"messages": self._messages})
        self.messages = final_state["messages"]
        if(self._show_tool_messages):
            last_human_index = next(
                (i for i in reversed(range(len(self.messages))) if self.messages[i].type == "human"),
                -1
            ) 
            return [
                {
                    "role": msg.type,
                    "content": msg.content,
                    "tool_calls": str(msg.tool_calls) if hasattr(msg, "tool_calls") else None,
                } 
                for msg in self.messages[last_human_index + 1 :]
            ]
        else:
            # only the last message, 1 item list
            return [
                {
                    "role": self.messages[-1].type,
                    "content": self.messages[-1].content,
                    "tool_calls": str(self.messages[-1].tool_calls)
                    if hasattr(self.messages[-1], "tool_calls")
                    else None,
                }
            ]

class State(TypedDict):
    messages: Annotated[list, add_messages]
    output_table_ids: list[str]


def build_agent_graph(
    chat_model: BaseChatModel, tools: list[BaseTool], num_data_outputs: int
):
    output_selection_tool = _output_selection_tool(num_data_outputs)
    chat_model = chat_model.bind_tools(tools + [output_selection_tool])
    agent_node = _build_agent_node(chat_model)
    graph = _compile_graph(agent_node, tools)
    return graph


def _output_selection_tool(num_data_outputs: int) -> StructuredTool:
    def func(content: str, output_table_ids: list[str]):
        return {"content": content, "selected_output_ids": output_table_ids}

    return StructuredTool.from_function(
        func=func,
        name="Output_Table_Selection",
        description="Use this tool to reply to the user. "
        "You should select a number of table IDs from the list of output table "
        "IDs based on their relevance to the user prompt and how many output tables are expected.",
        args_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content of the message to be sent to the user.",
                },
                "output_table_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"The IDs of the {num_data_outputs} output tables that are returned to the user.",
                },
            },
            "required": ["output_table_ids"],
        },
    )


def _build_agent_node(chat_model: BaseChatModel):
    def agent_node(state: State):
        messages = state["messages"]
        response = chat_model.invoke(messages)
        messages.append(response)
        return {"messages": messages}

    return agent_node


def _router(state: State):
    messages = state["messages"]
    last_msg = messages[-1]

    if getattr(last_msg, "tool_calls", None):
        return "tools"

    return "END"


def _compile_graph(agent_node, tools):
    builder = StateGraph(State)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        _router,
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("select_output", END)

    return builder.compile()
