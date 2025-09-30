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

from models.base import (
    LLMPortObjectSpec,
    LLMPortObject,
    ChatConversationSettings,
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    OutputFormatOptions,
    chat_model_port_type,
)
from tools.base import (
    tool_list_port_type,
    ToolListPortObject,
    ToolListPortObjectSpec,
)

from knime.extension.nodes import (
    get_port_type_for_id,
    get_port_type_for_spec_type,
    load_port_object,
    save_port_object,
    FilestorePortObject,
)
from base import AIPortObjectSpec

import os
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


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing Google Search API key.",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


@knext.parameter_group(label="Prompt Settings")
class ChatMessageSettings:
    message = knext.MultilineStringParameter(
        "New message", "Specify the new message to be sent to the agent."
    )


# TODO: Add agent type in the future?
class AgentPortObjectSpec(AIPortObjectSpec):
    def __init__(
        self,
        llm_spec: LLMPortObjectSpec,
    ) -> None:
        self._llm_spec = llm_spec
        self._llm_type = get_port_type_for_spec_type(type(llm_spec))

    @property
    def llm_spec(self) -> LLMPortObjectSpec:
        return self._llm_spec

    @property
    def llm_type(self) -> knext.PortType:
        return self._llm_type

    def validate_context(self, ctx: knext.ConfigurationContext):
        self._llm_spec.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "llm": {"type": self.llm_type.id, "spec": self.llm_spec.serialize()},
        }

    @classmethod
    def deserialize_llm_spec(cls, data: dict):
        llm_data = data["llm"]
        spec_type = get_port_type_for_id(llm_data["type"])

        return spec_type.spec_class.deserialize(llm_data["spec"])

    @classmethod
    def deserialize(cls, data: dict) -> "AgentPortObjectSpec":
        return cls(cls.deserialize_llm_spec(data))


class AgentPortObject(FilestorePortObject):
    def __init__(self, spec: AgentPortObjectSpec, llm: LLMPortObject) -> None:
        super().__init__(spec)
        self._llm = llm

    @property
    def llm(self) -> LLMPortObject:
        return self._llm

    def create_agent(self, ctx, tools):
        raise NotImplementedError()

    def write_to(self, file_path):
        os.makedirs(file_path)
        llm_path = os.path.join(file_path, "llm")
        save_port_object(self.llm, llm_path)

    @classmethod
    def read_from(cls, spec: AgentPortObjectSpec, file_path: str) -> "AgentPortObject":
        llm_path = os.path.join(file_path, "llm")
        llm = load_port_object(spec.llm_type.object_class, spec.llm_spec, llm_path)
        return cls(spec, llm)


agent_port_type = knext.port_type("Agent", AgentPortObject, AgentPortObjectSpec)


# region Agent Prompter 1.0
@knext.node(
    "Agent Prompter",
    knext.NodeType.PREDICTOR,
    agent_icon,
    category=agent_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Large Language Model",
        "OpenAI",
        "RAG",
        "Retrieval Augmented Generation",
    ],
    is_deprecated=True,
)
@knext.input_port("Agent", "The agent to prompt.", agent_port_type)
@knext.input_port("Tools", "The tools the agent can use.", tool_list_port_type)
@knext.input_table(
    "Conversation",
    """Table containing the conversation history. 
                    Has to contain at least two string columns, which can be empty if this is the beginning of the conversation.
                    """,
)
@knext.output_table(
    "Continued conversation",
    """The conversation table extended by two rows.
                    One row for the user's prompt and one row for the agent's response.
                    Can be used as input in subsequent Agent Prompter executions to continue the conversation further.
                    """,
)
class AgentPrompter:
    """
    Combines an agent with a set of tools and prompts it with a user-provided prompt.

    This node supplies an LLM agent with a set of tools and the conversation history, and
    prompts it with the user-provided query.

    The conversation table is expected to have at least two string columns that define previous conversation.
    If this is the start of the conversation, the conversation table can be empty.

    The agent always receives the full conversation table as context, which can lead to
    slower execution times for longer conversations, or execution failures if the context
    becomes too large for the LLM. If you experience such issues, you can truncate the conversation
    table by only keeping the last few messages, or use an LLM to summarize the conversation held so far.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the LLM,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable was not saved and will therefore not be available to downstream nodes.
    """

    conversation_settings = ChatConversationSettings(port_index=2)
    message_settings = ChatMessageSettings()
    enable_debug_output = knext.BoolParameter(
        "Enable debug output",
        "If checked, prints the output of LangChain's debug mode to the console.",
        False,
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        agent_spec: AgentPortObjectSpec,
        tools_spec: ToolListPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        agent_spec.validate_context(ctx)
        tools_spec.validate_context(ctx)
        self.conversation_settings.configure(input_table_spec)
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), self.conversation_settings.role_column),
                knext.Column(knext.string(), self.conversation_settings.content_column),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        agent_obj: AgentPortObject,
        tools_obj: ToolListPortObject,
        input_table: knext.Table,
    ):
        chat_history_df = input_table[
            [
                self.conversation_settings.role_column,
                self.conversation_settings.content_column,
            ]
        ].to_pandas()

        # TODO return messages might depend on the type of model (i.e. chat needs it llm doesn't)?

        import langchain
        import langchain.agents

        langchain.debug = self.enable_debug_output

        messages = self.conversation_settings.create_messages(chat_history_df)

        tools = tools_obj.create_tools(ctx)
        agent = agent_obj.create_agent(ctx, tools)

        agent_exec = langchain.agents.AgentExecutor(
            agent=agent,
            tools=tools,
        )

        response = agent_exec.invoke(
            {
                "input": self.message_settings.message,
                "chat_history": messages,
            }
        )

        user_input_row = ["Human", self.message_settings.message]
        agent_output_row = ["AI", response["output"]]

        chat_history_df.loc[f"Row{len(chat_history_df)}"] = user_input_row
        chat_history_df.loc[f"Row{len(chat_history_df)}"] = agent_output_row

        return knext.Table.from_pandas(chat_history_df)


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


def _last_tool_column(schema: knext.Schema):
    tool_type = _tool_type()
    tool_columns = [col for col in schema if col.ktype == tool_type]
    if not tool_columns:
        raise knext.InvalidParametersError(
            "No tool column found in the tools table. Please provide a valid tool column."
        )
    return tool_columns[-1].name


def _last_history_column(schema: knext.Schema):
    message_type = _message_type()
    history_columns = [col for col in schema if col.ktype == message_type]
    if not history_columns:
        raise knext.InvalidParametersError(
            "No history column found in the conversation history table. Please provide a valid history column."
        )
    return history_columns[-1].name


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


def _message_type() -> knext.LogicalType:
    from knime.types.message import MessageValue

    return knext.logical(MessageValue)


def has_conversation_table(ctx: knext.DialogCreationContext):
    return ctx.get_input_specs()[2] is not None


def _conversation_column_parameter() -> knext.ColumnParameter:
    return knext.ColumnParameter(
        "Conversation column",
        "The column containing the conversation history if a conversation history table is connected.",
        port_index=2,
        column_filter=util.create_type_filter(_message_type()),
    ).rule(knext.DialogContextCondition(has_conversation_table), knext.Effect.SHOW)


def _has_tools_table(ctx: knext.DialogCreationContext):
    # Port index 1 is the tools table
    return ctx.get_input_specs()[1] is not None


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


# endregion


# region Agent Prompter 2.0
@knext.node(
    "Agent Prompter",
    node_type=knext.NodeType.PREDICTOR,
    icon_path=agent_prompter_icon,
    category=agent_category,
)
@knext.input_port(
    "Chat model", "The chat model to use.", port_type=chat_model_port_type
)
@knext.input_table("Tools", "The tools the agent can use.")
@knext.input_table(
    "Conversation History",
    "The table containing the conversation held so far.",
    optional=True,
)
@knext.input_table_group(
    "Data inputs",
    "The data inputs for the agent.",
)
@knext.output_table(
    "Conversation",
    "The conversation between the LLM and the tools reflecting the agent execution.",
)
@knext.output_table_group(
    "Data outputs",
    "The data outputs of the agent.",
)
class AgentPrompter2:
    """
    Combines a chat model with a set of tools to form an agent and prompts it with a user-provided prompt.

    This node builds and executes an agent that responds to a single user prompt using a language model and a set of user-defined tools.

    Each tool is represented as a KNIME workflow with configurations (e.g., strings, numbers, column selection) and can optionally accept input data tables.
     While the agent does not have access to raw data, it is informed about available tables through metadata — including column names and types — enabling it to select suitable data for each tool as needed.

    When the node is executed, the agent reasons step-by-step to fulfill the user’s prompt.
    It may call one or more tools in sequence, using the output of one tool as input for another.
    The model autonomously selects tools and input data based on the prompt and the tools’ names and descriptions.
    Including clear tool descriptions and example use cases significantly improves the agent’s decision-making.

    The entire internal reasoning process — including tool calls and decisions — is captured and available via the conversation output table.
    This node is designed for non-interactive, one-shot execution. For interactive, multi-turn conversations, use the Chat Agent Prompter node.
    """

    developer_message = _system_message_parameter()

    user_message = knext.MultilineStringParameter(
        "User message",
        "Message from the end user, prioritized behind the developer message.",
    )

    tool_column = _tool_column_parameter()

    conversation_column = _conversation_column_parameter()

    # New parameter for column name if no history table is connected
    conversation_column_name = knext.StringParameter(
        "Conversation column name",
        "Name of the conversation column if no conversation history table is connected.",
        default_value="Conversation",
    ).rule(knext.DialogContextCondition(has_conversation_table), knext.Effect.HIDE)

    recursion_limit = _recursion_limit_parameter()

    debug = _debug_mode_parameter()

    data_message_prefix = _data_message_prefix_parameter()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chat_model_spec: ChatModelPortObjectSpec,
        tools_schema: knext.Schema,
        history_schema: Optional[knext.Schema],
        data_schemas: list[knext.Schema],
    ) -> knext.Schema:
        chat_model_spec.validate_context(ctx)
        self._configure_tool_tables(tools_schema)

        return self._create_conversation_schema(history_schema), [
            None
        ] * ctx.get_connected_output_port_numbers()[1]

    def _configure_tool_tables(self, tools_schema: knext.Schema):
        if self.tool_column is None:
            self.tool_column = _last_tool_column(tools_schema)
        elif self.tool_column not in tools_schema.column_names:
            raise knext.InvalidParametersError(
                f"Column {self.tool_column} not found in the tools table."
            )

    def _create_conversation_schema(
        self, history_schema: Optional[knext.Schema]
    ) -> knext.Schema:
        if history_schema is not None:
            if self.conversation_column is None:
                self.conversation_column = _last_history_column(history_schema)
            if self.conversation_column not in history_schema.column_names:
                raise knext.InvalidParametersError(
                    f"Column {self.conversation_column} not found in the conversation history table."
                )
            return knext.Schema.from_columns(
                [
                    knext.Column(_message_type(), self.conversation_column),
                ]
            )
        # Use user-provided column name if no history table is given
        return knext.Schema.from_columns(
            [knext.Column(_message_type(), self.conversation_column_name)]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        tools_table: knext.Table,
        history_table: Optional[knext.Table],
        input_tables: list[knext.Table],
    ):
        from langchain.chat_models.base import BaseChatModel
        import pandas as pd
        from ._data_service import (
            DataRegistry,
            LangchainToolConverter,
        )
        from ._tool import ExecutionMode
        from ._agent import check_for_invalid_tool_calls
        from langgraph.prebuilt import create_react_agent
        from knime.types.message import to_langchain_message, from_langchain_message

        data_registry = DataRegistry.create_with_input_tables(
            input_tables, data_message_prefix=self.data_message_prefix
        )
        tool_converter = LangchainToolConverter(
            data_registry,
            ctx,
            ExecutionMode.DEBUG if self.debug else ExecutionMode.DEFAULT,
        )

        chat_model: BaseChatModel = chat_model.create_model(
            ctx, output_format=OutputFormatOptions.Text
        )

        tool_cells = _extract_tools_from_table(tools_table, self.tool_column)

        tools = [tool_converter.to_langchain_tool(tool) for tool in tool_cells]

        messages = []

        if history_table is not None:
            if self.conversation_column not in history_table.column_names:
                raise knext.InvalidParametersError(
                    f"Column {self.conversation_column} not found in the conversation history table."
                )
            history_df = history_table[self.conversation_column].to_pandas()
            messages = []
            for msg in history_df[self.conversation_column]:
                lc_msg = to_langchain_message(msg)
                # Sanitize tool names so they match the current sanitized mapping
                lc_msg = tool_converter.sanitize_tool_names(lc_msg)
                messages.append(lc_msg)

        if data_registry.has_data or tool_converter.has_data_tools:
            messages.append(data_registry.create_data_message())

        if self.user_message:
            messages.append({"role": "user", "content": self.user_message})

        num_data_outputs = ctx.get_connected_output_port_numbers()[1]

        graph = create_react_agent(
            chat_model,
            tools=tools,
            prompt=self.developer_message,
        )

        inputs = {"messages": messages}
        config = {"recursion_limit": self.recursion_limit}

        try:
            final_state = graph.invoke(inputs, config=config)
        except Exception as e:
            if "Recursion limit" in str(e):
                raise knext.InvalidParametersError(
                    f"""Recursion limit of {self.recursion_limit} reached. 
                    You can increase the limit by setting the `recursion_limit` parameter to a higher value."""
                )
            else:
                raise knext.InvalidParametersError(
                    f"An error occurred while executing the agent: {e}"
                )

        messages = final_state["messages"]

        check_for_invalid_tool_calls(messages[-1])

        desanitized_messages = [
            tool_converter.desanitize_tool_names(msg) for msg in messages
        ]

        output_column_name = (
            self.conversation_column_name
            if history_table is None
            else self.conversation_column
        )

        result_df = pd.DataFrame(
            {
                output_column_name: [
                    from_langchain_message(msg) for msg in desanitized_messages
                ]
            }
        )

        conversation_table = knext.Table.from_pandas(result_df)

        if num_data_outputs == 0:
            return conversation_table
        else:
            # allow the model to pick which output tables to return
            return conversation_table, data_registry.get_last_tables(num_data_outputs)


def _extract_tools_from_table(tools_table: knext.Table, tool_column: str):
    import pyarrow.compute as pc

    tools = tools_table[tool_column].to_pyarrow().column(tool_column)
    filtered_tools = pc.filter(tools, pc.is_valid(tools))
    return filtered_tools.to_pylist()


# endregion


# region Agent Chat Widget


@knext.node(
    "Agent Chat Widget",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=chat_agent_icon,
    category=agent_category,
)
@knext.input_port(
    "Chat model", "The chat model to use.", port_type=chat_model_port_type
)
@knext.input_table("Tools", "The tools the agent can use.", optional=True)
@knext.input_table_group(
    "Data inputs",
    "The data inputs for the agent.",
)
@knext.output_table(
    "Conversation",
    "The conversation between the LLM and the tools reflecting the agent execution.",
)
@knext.output_table_group(
    "Data outputs",
    "The data outputs of the agent.",
)
@knext.output_view(
    "Chat",
    "Shows the chat interface for interacting with the agent.",
    static_resources="src/agents/chat_app/dist/",
    index_html_path="index.html",
)
class AgentChatWidget:
    """

    Enables interactive, multi-turn conversations with an AI agent that uses tools and data to fulfill user prompts.

    This node enables interactive, multi-turn conversations with an AI agent, combining a chat model with a set of tools and optional input data.

    The agent is assembled from the provided chat model and tools, each defined as a KNIME workflow. Tools can include configurable parameters (e.g., string inputs, numeric settings, column selectors) and may optionally consume input data in the form of KNIME tables. While the agent does not access raw data directly, it is informed about the structure of available tables (i.e., column names and types). This allows the model to select and route data to tools during conversation.

    Unlike the standard Agent Prompter node, which executes a single user prompt, this node supports multi-turn, interactive dialogue. The user can iteratively send prompts and receive responses, with the agent invoking tools as needed in each conversational turn. Tool outputs from earlier turns can be reused in later interactions, enabling rich, context-aware workflows.

    This node is designed for real-time, interactive usage where the conversation takes place directly within the KNIME view, where the agent’s responses and reasoning are shown incrementally as the dialogue progresses. Additionally, it can also optionally output the conversation history as table.

    To ensure effective agent behavior, provide meaningful tool names and clear descriptions — including example use cases if applicable.
    """

    developer_message = _system_message_parameter()

    tool_column = _tool_column_parameter().rule(
        knext.DialogContextCondition(_has_tools_table), knext.Effect.SHOW
    )

    conversation_column_name = knext.StringParameter(
        "Conversation column name",
        "Name of the conversation column in the output table.",
        default_value="Conversation",
    )

    class ReexecutionTrigger(knext.EnumParameterOptions):
        NONE = (
            "Never",
            "Never re-execute the node to update the conversation and data outputs explicitly from within the chat.",
        )
        INTERACTION = (
            "After every interaction",
            "Update the conversation and data outputs after each completed chat interaction.",
        )

    reexecution_trigger = knext.EnumParameter(
        "Re-execute node to update conversation output",
        "The user action that triggers a re-execution of the node in order to update the conversation and data output tables.",
        ReexecutionTrigger.NONE.name,
        ReexecutionTrigger,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
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
        "If checked, the views of nodes in the workflow-based tool will be shown in the chat. The view nodes must be top-level nodes, i.e., not inside a component or metanode.",
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

        return knext.Schema.from_columns(
            [knext.Column(_message_type(), self.conversation_column_name)]
        ), [None] * ctx.get_connected_output_port_numbers()[1]

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        tools_table: Optional[knext.Table],
        input_tables: list[knext.Table],
    ):
        import pandas as pd
        from knime.types.message import from_langchain_message
        from langchain_core.messages.utils import messages_from_dict
        from ._data_service import DataRegistry

        view_data = ctx._get_view_data()
        num_data_outputs = ctx.get_connected_output_port_numbers()[1]

        if view_data:
            messages = messages_from_dict(view_data["data"]["conversation"])
            message_values = [from_langchain_message(msg) for msg in messages]
            result_df = pd.DataFrame({self.conversation_column_name: message_values})
            conversation_table = knext.Table.from_pandas(result_df)
            data_registry = DataRegistry.create_from_view_data(view_data)
            return conversation_table, data_registry.get_last_tables(num_data_outputs)
        else:
            conversation_table = knext.Table.from_pandas(
                pd.DataFrame(
                    {
                        self.conversation_column_name: [],
                    }
                )
            )
            return conversation_table, [
                knext.Table.from_pandas(pd.DataFrame())  # empty table
            ] * num_data_outputs

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
            AgentChatWidgetDataService,
        )
        from ._tool import ExecutionMode
        from langchain_core.messages.utils import messages_from_dict

        view_data = ctx._get_view_data()

        chat_model = chat_model.create_model(
            ctx, output_format=OutputFormatOptions.Text
        )

        if view_data is None:
            data_registry = DataRegistry.create_with_input_tables(
                input_tables, data_message_prefix=self.data_message_prefix
            )
        else:
            data_registry = DataRegistry.create_from_view_data(view_data)

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

        if view_data is None:
            previous_messages = []
        else:
            previous_messages = messages_from_dict(view_data["data"]["conversation"])

        return AgentChatWidgetDataService(
            agent,
            data_registry,
            self.initial_message,
            previous_messages,
            self.recursion_limit,
            self.show_tool_calls_and_results,
            self.reexecution_trigger,
            tool_converter,
        )


# endregion
