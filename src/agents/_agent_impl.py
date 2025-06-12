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
from typing import Optional, Sequence
import knime.extension as knext

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool
import pandas as pd
import yaml


from langchain.tools import StructuredTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import logging
import re

_logger = logging.getLogger(__name__)




@dataclass
class Port:
    name: str
    description: str
    type: str
    spec: Optional[dict] = None

@dataclass
class DataItem:
    """Represents a data item in the registry."""
    meta_data: Port
    data: knext.Table
    

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
    def __init__(self):
        self._data: list[DataItem] = []

    @classmethod
    def create_with_input_tables(self, input_tables: Sequence[knext.Table]) -> "DataRegistry":
        """Creates a DataRegistry with the given input tables."""
        registry = DataRegistry()
        for i, table in enumerate(input_tables):
            spec = _spec_representation(table)
            port = Port(name=f"input_table_{i+1}", description=f"Input table {i+1}", type="Table", spec=spec)
            registry._data.append(DataItem(meta_data=port, data=table))
        return registry

    def add_table(self, table: knext.Table, port: Port) -> dict:
        spec = _spec_representation(table)
        meta_data = Port(
            name=port.name, description=port.description, type=port.type, spec=spec)
        self._data.append(DataItem(meta_data=meta_data, data=table))
        return {len(self._data) - 1: _port_to_dict(meta_data)}

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
    
    @property
    def has_data(self) -> bool:
        """Returns True if there is data in the registry, False otherwise."""
        return len(self._data) > 0

    def create_summary_message(self) -> Optional[HumanMessage]:
        if not self._data:
            return None
        content = _render_structured(**{str(id): _port_to_dict(data.meta_data) for id, data in enumerate(self._data)})
        msg = """# Data Summary
This message summarizes the data available for tool calls.
Each entry corresponds to a data item with its metadata.
Use the keys to refer to the respective data item for use in tool calls.
Tools that produce data will also include similar information in their output.
Data:
"""
        
        return HumanMessage(msg + content)

def _spec_representation(table: knext.Table) -> dict:
    return {column.name: str(column.ktype) for column in table.schema}

def _port_to_dict(port: Port) -> dict:
        return {
            "name": port.name,
            "description": port.description,
            "type": port.type,
            "spec": port.spec,
        }

class LangchainToolConverter:
    def __init__(
        self, data_registry: DataRegistry, ctx, debug: bool
    ):
        self._data_registry = data_registry
        self._ctx = ctx
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

        description_with_data_info = _render_structured(
            description=tool.description,
            input_ports=list(
                map(_port_to_dict, input_ports)
            ),
            output_ports=list(
                map(_port_to_dict, output_ports)
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
                for port, output in zip(output_ports, outputs):
                    output_reference = self._data_registry.add_table(output, port)
                    output_references.update(output_reference)

                return _render_structured(
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


def _render_structured(**kwargs) -> str:
    return yaml.dump(kwargs, sort_keys=False)


class AgentChatViewDataService:
    def __init__(
        self,
        agent_graph,
        data_registry: DataRegistry,
        initial_message: str,
        recursion_limit: int,
        show_tool_calls_and_results: bool,
    ):
        self._agent_graph = agent_graph
        self._data_registry = data_registry
        self._messages = [data_registry.create_summary_message()] if data_registry.has_data else []
        self._initial_message = initial_message
        self._recursion_limit = recursion_limit
        self._show_tool_calls_and_results = show_tool_calls_and_results

    def get_initial_message(self):
        if self._initial_message:
            return {
                "type": "ai",
                "content": self._initial_message,
            }
        else:
            pass

    def post_user_message(self, user_message: str):
        import threading

        if not hasattr(self, "_thread") or not self._thread.is_alive():
            self._last_messages = []
            self._thread = threading.Thread(
                target=self._post_user_message, args=(user_message, self._last_messages)
            )
            self._thread.start()

    def get_last_messages(self):
        if not hasattr(self, "_thread"):
            return []
        elif self._thread.is_alive():
            # wait with timeout to enable long-polling
            self._thread.join(timeout=5)

        return self._last_messages

    def _post_user_message(self, user_message: str, last_messages: list):
        self._messages.append({"role": "user", "content": user_message})
        config = {"recursion_limit": self._recursion_limit}
        try:
            final_state = self._agent_graph.invoke({"messages": self._messages}, config)
            self._messages = final_state["messages"]
        except Exception as e:
            if "Recursion limit" in str(e):
                last_messages.append(
                    {
                        "type": "error",
                        "content": f"""Recursion limit of {self._recursion_limit} reached. 
                        You can increase the limit by setting the `recursion_limit` parameter.""",
                    }
                )
                return
            else:
                last_messages.append(
                    {
                        "type": "error",
                        "content": f"An error occurred while executing the agent: {e}",
                    }
                )
                return
        if self._show_tool_calls_and_results:
            last_human_index = next(
                (
                    i
                    for i in reversed(range(len(self._messages)))
                    if self._messages[i].type == "human"
                ),
                -1,
            )
            for msg in self._messages[last_human_index + 1 :]:
                last_messages.append(self._to_frontend_message(msg))
        else:
            last_messages.append(self._to_frontend_message(self._messages[-1]))

    def _to_frontend_message(self, message):
        import json

        fe_message = {
            "id": message.id if hasattr(message, "id") else None,
            "type": message.type,
            "content": message.content if hasattr(message, "content") else None,
            "name": message.name if hasattr(message, "name") else None,
        }

        if message.type == "ai" and hasattr(message, "tool_calls"):
            fe_message["toolCalls"] = [
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "args": json.dumps(tool_call["args"], indent=2) if "args" in tool_call else None,
                }
                for tool_call in message.tool_calls
            ]
        elif message.type == "tool":
            fe_message["toolCallId"] = message.tool_call_id
        return fe_message


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
