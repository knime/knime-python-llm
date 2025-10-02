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


from typing import Optional
import knime.extension as knext
import util

from ._data import DataRegistry
from ._tool import LangchainToolConverter
from ._agent import check_for_invalid_tool_calls
import yaml
import queue
import threading

from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage


from models.base import (
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    OutputFormatOptions,
    chat_model_port_type,
)


import logging

_logger = logging.getLogger(__name__)


agent_icon = "icons/generic/agent.png"
chat_agent_icon = "icons/agentic/chat_agent.png"
agent_category = knext.category(
    path=util.main_category,
    level_id="agents",
    name="Agents",
    description="",
    icon=agent_icon,
)
agent_prompter_icon = "icons/agentic/agent.png"


def _system_message_parameter():
    return knext.MultilineStringParameter(
        "System message",
        "Instructions provided by the workflow builder that guide the agent's behavior. "
        "It typically defines the agent’s role, its tone, boundaries, and behavioral rules."
        "This message is prioritized over the user message and should not contain any information "
        ""
        "that the user can inject in order to prevent prompt injection attacks.",
        default_value="""## PERSISTENCE
You are an agent - please keep going until the user's query is completely 
resolved, before ending your turn and yielding back to the user. Only 
terminate your turn when you are sure that the problem is solved.

## TOOL CALLING
If you are not sure about file content or codebase structure pertaining to 
the user's request, use your tools to read files and gather the relevant 
information: do NOT guess or make up an answer.

## PLANNING
You MUST plan extensively before each function call, and reflect 
extensively on the outcomes of the previous function calls. DO NOT do this 
entire process by making function calls only, as this can impair your 
ability to solve the problem and think insightfully.""",
    )


def _tool_type() -> knext.LogicalType:
    import knime.types.tool as ktt

    return knext.logical(ktt.WorkflowTool)


def _tool_column_parameter():
    return knext.ColumnParameter(
        "Tool column",
        "The column of the tools table holding the tools the agent can use.",
        port_index=1,
        column_filter=util.create_type_filter(_tool_type()),
    )


def _has_tools_table(ctx: knext.DialogCreationContext):
    # Port index 1 is the tools table
    return ctx.get_input_specs()[1] is not None


def _recursion_limit_parameter():
    return knext.IntParameter(
        "Recursion limit",
        """
        The maximum number of times the agent can repeat its steps to
        avoid getting stuck in an endless loop.
        """,
        default_value=25,
        min_value=1,
        is_advanced=True,
    )


def _debug_mode_parameter():
    return knext.BoolParameter(
        "Debug mode",
        "In debug mode, tool executions are displayed as meta nodes in the agent workflow and the meta node is kept in case of an error in the tool.",
        default_value=False,
        is_advanced=True,
    )


def _data_message_prefix_parameter():
    return knext.MultilineStringParameter(
        "Data message prefix",
        "Prefix for the data message shown to the agent. You can use this to customize the instructions about the data repository.",
        default_value="""## Data Tools Interface
You have access to tools that can consume and produce data.
The interaction with these tools is mediated via a data repository that keeps track of all available data items.
The repository is represented as a map from IDs to data items.

Each data item is represented by:
- The name of the data
- The description of the data
- The type of data
- The spec of the data giving a high-level overview of the data (e.g. the columns in a table)

Note: You do not have access to the actual data content, only the metadata and IDs.

## Using Tools with Data
### Consuming Data:
To pass data to a tool, provide the ID of the relevant data item.
Once invoked, the tool will receive the data associated with that ID.

### Producing Data:
- Tools that produce data will include an update to the data repository in their tool message.
- This update follows the same format as the initial data repository: A map of IDs to data items.

You must incorporate these updates into your working view of the data repository.
## Data repository:
""",
        is_advanced=True,
    )


def _last_tool_column(schema: knext.Schema):
    tool_type = _tool_type()
    tool_columns = [col for col in schema if col.ktype == tool_type]
    if not tool_columns:
        raise knext.InvalidParametersError(
            "No tool column found in the tools table. Please provide a valid tool column."
        )
    return tool_columns[-1].name


def _extract_tools_from_table(tools_table: knext.Table, tool_column: str):
    import pyarrow.compute as pc

    tools = tools_table[tool_column].to_pyarrow().column(tool_column)
    filtered_tools = pc.filter(tools, pc.is_valid(tools))
    return filtered_tools.to_pylist()


# region Agent Chat View
@knext.node(
    "Agent Chat View",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=chat_agent_icon,
    category=agent_category,
    is_deprecated=False,
)
@knext.input_port(
    "Chat model", "The chat model to use.", port_type=chat_model_port_type
)
@knext.input_table("Tools", "The tools the agent can use.", optional=True)
@knext.input_table_group(
    "Data inputs",
    "The data inputs for the agent.",
)
@knext.output_view(
    "Chat",
    "Shows the chat interface for interacting with the agent.",
    static_resources="src/agents/chat_app_deprecated/dist",
    index_html_path="index.html",
)
class AgentChatView:
    """Enables interactive, multi-turn conversations with an AI agent that uses tools and data to fulfill user prompts.

    This node enables interactive, multi-turn conversations with an AI agent, combining a chat model with a set of tools and optional input data.

    The agent is assembled from the provided chat model and tools, each defined as a KNIME workflow. Tools can include configurable parameters (e.g., string inputs, numeric settings, column selectors) and may optionally consume input data in the form of KNIME tables. While the agent does not access raw data directly, it is informed about the structure of available tables (i.e., column names and types). This allows the model to select and route data to tools during conversation.

    Unlike the standard Agent Prompter node, which executes a single user prompt, this node supports multi-turn, interactive dialogue. The user can iteratively send prompts and receive responses, with the agent invoking tools as needed in each conversational turn. Tool outputs from earlier turns can be reused in later interactions, enabling rich, context-aware workflows.

    This node is designed for real-time, interactive usage and does not produce a data output port. Instead, the conversation takes place directly within the KNIME view, where the agent’s responses and reasoning are shown incrementally as the dialogue progresses.

    To ensure effective agent behavior, provide meaningful tool names and clear descriptions — including example use cases if applicable.
    """

    developer_message = _system_message_parameter()

    tool_column = _tool_column_parameter().rule(
        knext.DialogContextCondition(_has_tools_table), knext.Effect.SHOW
    )

    initial_message = knext.MultilineStringParameter(
        "Initial AI message",
        "An optional 'AI' initial message to be shown to the user.",
    )

    show_tool_calls_and_results = knext.BoolParameter(
        "Show tool calls and results",
        "If checked, the tool calls and the tool call results will also be shown in the chat.",
        default_value=False,
    )

    show_views = knext.BoolParameter(
        "Show views",
        "If checked, the views of nodes in the workflow-based tool  will be shown in the chat.",
        default_value=lambda v: v
        >= knext.Version(
            5, 6, 0
        ),  # False for versions < 5.6.0 for backwards compatibility
        since_version="5.6.0",
    )

    recursion_limit = _recursion_limit_parameter()

    debug = _debug_mode_parameter()

    data_message_prefix = _data_message_prefix_parameter()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chat_model_spec: ChatModelPortObjectSpec,
        tools_schema: Optional[knext.Schema],
        data_schemas: list[knext.Schema],
    ) -> knext.Schema:
        chat_model_spec.validate_context(ctx)
        if tools_schema is not None:
            if self.tool_column is None:
                self.tool_column = _last_tool_column(tools_schema)
            elif self.tool_column not in tools_schema.column_names:
                raise knext.InvalidParametersError(
                    f"Column {self.tool_column} not found in the tools table."
                )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        tools_table: Optional[knext.Table],
        input_tables: list[knext.Table],
    ):
        pass

    def get_data_service(
        self,
        ctx,
        chat_model: ChatModelPortObject,
        tools_table: Optional[knext.Table],
        input_tables: list[knext.Table],
    ):
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        from ._data_service import (
            DataRegistry,
            LangchainToolConverter,
        )
        from ._tool import ExecutionMode

        chat_model = chat_model.create_model(
            ctx, output_format=OutputFormatOptions.Text
        )
        data_registry = DataRegistry.create_with_input_tables(
            input_tables, data_message_prefix=self.data_message_prefix
        )
        tool_converter = LangchainToolConverter(
            data_registry,
            ctx,
            ExecutionMode.DEBUG if self.debug else ExecutionMode.DETACHED,
            self.show_views,
        )
        if tools_table is not None:
            tool_cells = _extract_tools_from_table(tools_table, self.tool_column)
            tools = [tool_converter.to_langchain_tool(tool) for tool in tool_cells]
        else:
            tools = []

        memory = MemorySaver()
        agent = create_react_agent(
            chat_model, tools=tools, prompt=self.developer_message, checkpointer=memory
        )

        return AgentChatViewDataService(
            agent,
            data_registry,
            self.initial_message,
            self.recursion_limit,
            self.show_tool_calls_and_results,
            tool_converter,
        )


class AgentChatViewDataService:
    def __init__(
        self,
        agent_graph,
        data_registry: DataRegistry,
        initial_message: str,
        recursion_limit: int,
        show_tool_calls_and_results: bool,
        tool_converter: LangchainToolConverter,
    ):
        self._agent_graph = agent_graph
        self._data_registry = data_registry
        self._tool_converter = tool_converter
        self._messages = (
            [data_registry.create_data_message()]
            if data_registry.has_data or tool_converter.has_data_tools
            else []
        )
        self._initial_message = initial_message
        self._recursion_limit = recursion_limit
        self._show_tool_calls_and_results = show_tool_calls_and_results

        self._message_queue = queue.Queue()
        self._thread = None

    def get_initial_message(self):
        if self._initial_message:
            return {
                "type": "ai",
                "content": self._initial_message,
            }

    def post_user_message(self, user_message: str):
        if not self._thread or not self._thread.is_alive():
            while not self._message_queue.empty():
                try:
                    self._message_queue.get_nowait()
                except queue.Empty:
                    continue

            self._thread = threading.Thread(
                target=self._post_user_message, args=(user_message,)
            )
            self._thread.start()

    def get_last_messages(self):
        messages = []

        try:
            msg = self._message_queue.get(timeout=2)
            messages.append(msg)

            while not self._message_queue.empty():
                messages.append(self._message_queue.get_nowait())
        except queue.Empty:
            pass

        return messages

    def is_processing(self):
        is_alive = self._thread and self._thread.is_alive()

        return {"is_processing": is_alive or not self._message_queue.empty()}

    def get_configuration(self):
        return {
            "show_tool_calls_and_results": self._show_tool_calls_and_results,
        }

    def _post_user_message(self, user_message: str):
        self._messages.append(HumanMessage(content=user_message))
        config = {
            "recursion_limit": self._recursion_limit,
            "configurable": {"thread_id": "1"},
        }

        try:
            num_messages_at_last_step = len(self._messages)
            state_stream = self._agent_graph.stream(
                {"messages": self._messages},
                config,
                stream_mode="values",  # streams state by state
            )

            final_state = None
            for state in state_stream:
                new_messages = state["messages"][num_messages_at_last_step:]
                num_messages_at_last_step = len(state["messages"])
                final_state = state

                for new_message in new_messages:
                    # already added
                    if isinstance(new_message, HumanMessage):
                        continue

                    fe_messages = self._to_frontend_messages(new_message)
                    is_ai = isinstance(new_message, AIMessage)

                    for fe_msg in fe_messages:
                        should_send = False

                        if self._show_tool_calls_and_results:
                            # always send everything if showing tool calls
                            should_send = True
                        else:
                            # otherwise only send Views and AI messages
                            if fe_msg.get("type") == "view" or is_ai:
                                should_send = True

                        if should_send:
                            self._message_queue.put(fe_msg)

            if final_state:
                self._messages = final_state["messages"]
                if self._messages:
                    check_for_invalid_tool_calls(self._messages[-1])

        except Exception as e:
            error_message = {"type": "error", "content": f"An error occurred: {e}"}
            if "Recursion limit" in str(e):
                error_message["content"] = (
                    f"Recursion limit of {self._recursion_limit} reached."
                )
            self._message_queue.put(error_message)

    def _to_frontend_messages(self, message):
        # split the node-view-ids out into a separate message
        content = None
        viewNodeIds = []
        if hasattr(message, "content"):
            split = message.content.split("View node IDs")
            content = split[0]
            viewNodeIds = split[1].strip().split(",") if len(split) > 1 else []

        fe_message = {
            "id": message.id if hasattr(message, "id") else None,
            "type": message.type,
            "content": content,
            "name": message.name if hasattr(message, "name") else None,
        }

        if message.type == "ai" and hasattr(message, "tool_calls"):
            fe_message["toolCalls"] = [
                self._render_tool_call(tool_call) for tool_call in message.tool_calls
            ]
        elif message.type == "tool":
            fe_message["toolCallId"] = message.tool_call_id
            fe_message["name"] = self._tool_converter.desanitize_tool_name(message.name)

        if len(viewNodeIds) > 0:
            view_msgs = []
            base_id = getattr(message, "id", "msg")
            view_name = (
                fe_message.get("name") if message.type == "tool" else "Node View"
            )
            for idx, viewNodeId in enumerate(viewNodeIds):
                view_msgs.append(
                    {
                        "id": f"{base_id}-view-{idx}",
                        "type": "view",
                        "content": viewNodeId,
                        "name": view_name,
                    }
                )
            return [fe_message] + view_msgs

        return [fe_message]

    def _render_tool_call(self, tool_call):
        args = tool_call.get("args")
        return {
            "id": tool_call["id"],
            "name": self._tool_converter.desanitize_tool_name(tool_call["name"]),
            "args": yaml.dump(args, indent=2) if args else None,
        }


# endregion
