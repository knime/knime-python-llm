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
from ._parameters import (
    recursion_limit_mode_param_for_view,
    RecursionLimitMode,
    ErrorHandlingMode,
)
from ._agent import CancelError
from ._error_handler import AgentPrompterErrorHandler

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


@knext.parameter_group(
    label="Error Handling Settings", is_advanced=True, since_version="5.10.0"
)
class AgentPrompterErrorSettings:
    error_handling = knext.EnumParameter(
        "Error handling",
        "Specify the behavior of the agent when an error occurs.",
        ErrorHandlingMode.FAIL.name,
        ErrorHandlingMode,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    use_existing_error_column = knext.BoolParameter(
        "Continue existing error column",
        "If selected, the output table will continue the error column selected from the input conversation table.",
        default_value=False,
    ).rule(
        knext.And(
            knext.DialogContextCondition(has_conversation_table),
            knext.OneOf(
                error_handling,
                [ErrorHandlingMode.COLUMN.name],
            ),
        ),
        knext.Effect.SHOW,
    )

    error_column_name = knext.StringParameter(
        "Error column name",
        "Name of the newly generated error column.",
        default_value="Errors",
    ).rule(
        knext.Or(
            knext.OneOf(
                error_handling,
                [ErrorHandlingMode.FAIL.name],
            ),
            knext.And(
                knext.OneOf(
                    use_existing_error_column,
                    [True],
                ),
                knext.DialogContextCondition(has_conversation_table),
            ),
        ),
        knext.Effect.HIDE,
    )

    error_column = knext.ColumnParameter(
        "Error column",
        "The column containing the errors if a conversation history table is connected.",
        port_index=2,
        column_filter=util.create_type_filter(knext.string()),
    ).rule(
        knext.And(
            knext.DialogContextCondition(has_conversation_table),
            knext.OneOf(
                error_handling,
                [ErrorHandlingMode.COLUMN.name],
            ),
            knext.OneOf(
                use_existing_error_column,
                [True],
            ),
        ),
        knext.Effect.SHOW,
    )


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

    recursion_limit_handling = knext.EnumParameter(
        "When recursion limit is reached",
        "Specify the behavior of the agent when the recursion limit is reached.",
        RecursionLimitMode.FAIL.name,
        RecursionLimitMode,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        is_advanced=True,
        since_version="5.10.0",
    )

    recursion_limit_prompt = knext.MultilineStringParameter(
        "Recursion limit prompt",
        "This message guides the chat model when generating the final response based on all previously "
        "generated messages. It can be used to specify how the LLM should behave if information is missing "
        "because the agent stopped before executing tools.",
        default_value="""Please finalize the conversation using all previous messages.
Tools are not available in this step. 
If essential information is missing because tool calls in the previous message were deleted,
state that the tool could not be executed due to reaching the recursion limit.""",
        is_advanced=True,
        since_version="5.10.0",
    ).rule(
        knext.OneOf(
            recursion_limit_handling,
            [RecursionLimitMode.FINAL_RESPONSE.name],
        ),
        knext.Effect.SHOW,
    )

    debug = _debug_mode_parameter()

    data_message_prefix = _data_message_prefix_parameter()

    errors = AgentPrompterErrorSettings()

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
        self._check_column_names(history_schema)

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

    def _check_column_names(self, history_schema: Optional[knext.Schema]):
        if history_schema is not None:
            if self.conversation_column is None:
                self.conversation_column = _last_history_column(history_schema)
            if self.conversation_column not in history_schema.column_names:
                raise knext.InvalidParametersError(
                    f"Column {self.conversation_column} not found in the conversation history table."
                )

        if self.errors.error_handling == ErrorHandlingMode.COLUMN.name:
            if history_schema is not None and self.errors.use_existing_error_column:
                if self.errors.error_column is None:
                    self.errors.error_column = util.pick_default_column(
                        history_schema, knext.string()
                    )
                if self.errors.error_column not in history_schema.column_names:
                    raise knext.InvalidParametersError(
                        f"Column {self.errors.error_column} not found in the conversation history table."
                    )
                if self.conversation_column == self.errors.error_column:
                    raise knext.InvalidParametersError(
                        "The selected conversation and error columns must not be equal."
                    )
            else:
                if self.conversation_column_name == self.errors.error_column_name:
                    raise knext.InvalidParametersError(
                        "The specified conversation and error column names must not be equal."
                    )

    def _create_conversation_schema(
        self, history_schema: Optional[knext.Schema]
    ) -> knext.Schema:
        if history_schema is not None:
            convo_column_name = self.conversation_column
        else:
            convo_column_name = self.conversation_column_name
        columns = [knext.Column(_message_type(), convo_column_name)]

        if self.errors.error_handling == ErrorHandlingMode.COLUMN.name:
            if history_schema is not None and self.errors.use_existing_error_column:
                error_column_name = self.errors.error_column
            else:
                error_column_name = self.errors.error_column_name
            columns.append(knext.Column(knext.string(), error_column_name))

        return knext.Schema.from_columns(columns)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        tools_table: knext.Table,
        history_table: Optional[knext.Table],
        input_tables: list[knext.Table],
    ):
        from langchain.chat_models.base import BaseChatModel
        from ._data_service import (
            DataRegistry,
            LangchainToolConverter,
        )
        from ._tool import ExecutionMode
        from ._agent import validate_ai_message, Agent, AgentConfig, IterationLimitError
        import langchain_core.messages as lcm

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
        toolset = AgentPrompterToolset(tools)

        conversation = self._get_conversation(ctx, history_table, tool_converter, data_registry)

        num_data_outputs = ctx.get_connected_output_port_numbers()[1]

        config = AgentConfig(self.recursion_limit)
        agent = Agent(conversation, chat_model, toolset, config)

        error_handler = AgentPrompterErrorHandler(
            conversation=conversation,
            recursion_limit=self.recursion_limit,
            recursion_limit_handling=RecursionLimitMode[self.recursion_limit_handling],
            error_handling=ErrorHandlingMode[self.errors.error_handling],
            chat_model=chat_model,
            recursion_limit_prompt=self.recursion_limit_prompt,
        )

        try:
            agent.run()
        except CancelError as e:
            raise e
        except IterationLimitError:
            error_handler.handle_iteration_limit_error()
        except Exception as e:
            error_handler.handle_general_error(e)

        conversation.validate_final_message(validate_ai_message, ctx)

        output_column_name = (
            self.conversation_column_name
            if history_table is None
            else self.conversation_column
        )
        error_column_name = None
        if self.errors.error_handling == ErrorHandlingMode.COLUMN.name:
            error_column_name = (
                self.errors.error_column
                if self.errors.use_existing_error_column and history_table is not None
                else self.errors.error_column_name
            )

        conversation_table = conversation.create_output_table(
            tool_converter, output_column_name, error_column_name
        )

        if num_data_outputs == 0:
            return conversation_table
        else:
            # allow the model to pick which output tables to return
            return conversation_table, data_registry.get_last_tables(num_data_outputs)

    def _check_for_columns(self, history_table):
        if self.conversation_column not in history_table.column_names:
            raise knext.InvalidParametersError(
                f"Column {self.conversation_column} not found in the conversation history table."
            )
        if (
            self.errors.error_handling == ErrorHandlingMode.COLUMN.name
            and self.errors.use_existing_error_column
        ):
            if self.errors.error_column not in history_table.column_names:
                raise knext.InvalidParametersError(
                    f"Column {self.errors.error_column} not found in the conversation history table."
                )

    def _get_conversation(
        self,
        ctx: knext.ExecutionContext,
        history_table: Optional[knext.Table],
        tool_converter,
        data_registry,
    ) -> "AgentPrompterConversation":
        from knime.types.message import to_langchain_message
        from langchain_core.messages import SystemMessage, HumanMessage
        import pandas as pd

        def append_msg(conversation, msg, tool_converter):
            lc_msg = to_langchain_message(msg)
            # Sanitize tool names so they match the current sanitized mapping
            lc_msg = tool_converter.sanitize_tool_names(lc_msg)
            conversation.append_messages(lc_msg)

        conversation = AgentPrompterConversation(self.errors.error_handling, ctx)

        if self.developer_message:
            conversation.append_messages(SystemMessage(self.developer_message))

        if history_table is not None:
            self._load_history_into_conversation(
                conversation, history_table, tool_converter
            )

        if data_registry.has_data or tool_converter.has_data_tools:
            conversation.append_messages(data_registry.create_data_message())
        if self.user_message:
            conversation.append_messages(HumanMessage(self.user_message))

        return conversation

    def _load_history_into_conversation(
        self,
        conversation: "AgentPrompterConversation",
        history_table: knext.Table,
        tool_converter,
    ):
        """Extract history table processing into dedicated method."""
        from knime.types.message import to_langchain_message
        import pandas as pd

        def append_msg(conversation, msg, tool_converter):
            lc_msg = to_langchain_message(msg)
            # Sanitize tool names so they match the current sanitized mapping
            lc_msg = tool_converter.sanitize_tool_names(lc_msg)
            conversation.append_messages(lc_msg)

        if (
            self.errors.error_handling == ErrorHandlingMode.COLUMN.name
            and self.errors.use_existing_error_column
        ):
            history_df = history_table[
                [self.conversation_column, self.errors.error_column]
            ].to_pandas()

            for idx, (msg, err) in enumerate(
                history_df[
                    [self.conversation_column, self.errors.error_column]
                ].itertuples(index=False, name=None)
            ):
                has_msg = pd.notna(msg)
                has_err = pd.notna(err)

                if has_msg and has_err:
                    row_id = history_df.index[idx]
                    raise RuntimeError(
                        f"Conversation table contains row with both message and error. Row ID: {row_id}"
                    )
                if has_msg:
                    append_msg(conversation, msg, tool_converter)
                elif has_err:
                    conversation.append_error(Exception(err))
                else:
                    row_id = history_df.index[idx]
                    raise RuntimeError(
                        f"Conversation table contains empty row. Row ID: {row_id}"
                    )
        else:
            history_df = history_table[self.conversation_column].to_pandas()
            for msg in history_df[self.conversation_column]:
                if pd.notna(msg):
                    append_msg(conversation, msg, tool_converter)


def _extract_tools_from_table(tools_table: knext.Table, tool_column: str):
    import pyarrow.compute as pc

    tools = tools_table[tool_column].to_pyarrow().column(tool_column)
    filtered_tools = pc.filter(tools, pc.is_valid(tools))
    return filtered_tools.to_pylist()


class AgentPrompterConversation:
    def __init__(self, error_handling, ctx: knext.ExecutionContext = None):
        self._error_handling = error_handling
        self._message_and_errors = []
        self._is_message = []
        self._ctx = ctx

    def append_messages(self, messages):
        """Raises a CancelError if the context was canceled."""
        from langchain_core.messages import BaseMessage

        if isinstance(messages, BaseMessage):
            messages = [messages]

        if self._ctx and self._ctx.is_canceled():
            raise CancelError("Execution canceled.")

        for msg in messages:
            self._append(msg)

    def append_error(self, error):
        if not isinstance(error, Exception):
            raise error

        if self._error_handling == ErrorHandlingMode.FAIL.name:
            raise error
        else:
            self._append(error)

    def get_messages(self):
        messages = [
            moe
            for is_msg, moe in zip(self._is_message, self._message_and_errors)
            if is_msg
        ]
        return messages

    def validate_final_message(self, validate_ai_message, ctx):
        """Validate the final AI message and handle validation errors appropriately."""
        from langchain_core.messages import AIMessage
        
        messages = self.get_messages()
        if messages and isinstance(messages[-1], AIMessage):
            try:
                validate_ai_message(messages[-1])
            except Exception as e:
                if self._error_handling == ErrorHandlingMode.FAIL.name:
                    ctx.set_warning(str(e))
                else:
                    self.append_error(e)

    def _append(self, message_or_error):
        from langchain_core.messages import BaseMessage
        
        self._message_and_errors.append(message_or_error)
        self._is_message.append(isinstance(message_or_error, BaseMessage))

    def _construct_output(self):
        return [
            {
                "message": moe if is_msg else None,
                "error": moe if not is_msg else None,
            }
            for is_msg, moe in zip(self._is_message, self._message_and_errors)
        ]

    def create_output_table(
        self,
        tool_converter,
        output_column_name: str,
        error_column_name: str = None,
    ) -> knext.Table:
        import pandas as pd
        from knime.types.message import from_langchain_message
        from langchain_core.messages import SystemMessage

        if error_column_name is None:
            messages = self.get_messages()
            if messages and isinstance(messages[0], SystemMessage):
                messages = messages[1:]
            desanitized_messages = [
                tool_converter.desanitize_tool_names(msg) for msg in messages
            ]
            result_df = pd.DataFrame(
                {
                    output_column_name: [
                        from_langchain_message(msg) for msg in desanitized_messages
                    ]
                }
            )
        else:
            messages_and_errors = [
                moe
                for moe in self._construct_output()
                if not isinstance(moe["message"], SystemMessage)
            ]
            desanitized_messages_and_errors = [
                {
                    "message": tool_converter.desanitize_tool_names(moe["message"])
                    if moe["message"] is not None
                    else None,
                    "error": moe["error"],
                }
                for moe in messages_and_errors
            ]
            messages = [
                from_langchain_message(moe["message"])
                if moe["message"] is not None
                else None
                for moe in desanitized_messages_and_errors
            ]
            errors = [
                str(moe["error"]) if moe["error"] is not None else None
                for moe in desanitized_messages_and_errors
            ]
            result_df = pd.DataFrame(
                {output_column_name: messages, error_column_name: errors}
            )

            if not any(messages):
                result_df[output_column_name] = result_df[output_column_name].astype(
                    _message_type().to_pandas()
                )
            if not any(errors):
                result_df[error_column_name] = result_df[error_column_name].astype(
                    "string"
                )

        return knext.Table.from_pandas(result_df)


class AgentPrompterToolset:
    def __init__(self, tools):
        self._by_name: dict = {t.name: t for t in tools}

    @property
    def tools(self):
        return list(self._by_name.values())

    def execute(self, tool_calls):
        from langchain_core.messages import ToolMessage

        results = []
        for tool_call in tool_calls:
            if tool_call["name"] not in self._by_name:
                msg = f"Error: Tool '{tool_call['name']}' not found among available tools: {list(self._by_name.keys())}"
                results.append(ToolMessage(msg, tool_call_id=tool_call["id"]))
                continue

            tool = self._by_name[tool_call["name"]]

            try:
                result = tool.invoke(tool_call["args"])
            except Exception as e:
                result = "Error: " + str(e)

            results.append(
                ToolMessage(
                    result, tool_call_id=tool_call["id"], name=tool_call["name"]
                )
            )
        return results


# endregion


# region Agent Chat Widget


@knext.node(
    "Agent Chat Widget (experimental)",
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
@knext.output_port(
    "Combined tools workflow",
    "The combined workflow of all tools used by the agent.",
    knext.PortType.WORKFLOW,
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
    static_resources="src/agents/chat_app/dist",
    index_html_path="index.html",
)
class AgentChatWidget:
    """
    Enables interactive, multi-turn conversations with an AI agent that uses tools and data to fulfill user prompts.

    **This node is experimental and intended to gather early feedback. It is providing the base functionality and will change in future releases, potentially in a non-backwards compatible way (e.g. requires to be re-executed).**

    This node enables interactive, multi-turn conversations with an AI agent, combining a chat model with a set of tools and optional input data.

    The agent is assembled from the provided chat model and tools, each defined as a KNIME workflow. Tools can include configurable parameters (e.g., string inputs, numeric settings, column selectors) and may optionally consume input data in the form of KNIME tables. While the agent does not access raw data directly, it is informed about the structure of available tables (i.e., column names and types). This allows the model to select and route data to tools during conversation.

    Unlike the standard Agent Prompter node, which executes a single user prompt, this node supports multi-turn, interactive dialogue. The user can iteratively send prompts and receive responses, with the agent invoking tools as needed in each conversational turn. Tool outputs from earlier turns can be reused in later interactions, enabling rich, context-aware workflows.

    This node is designed for real-time, interactive usage where the conversation takes place directly within the KNIME view, where the agent’s responses and reasoning are shown incrementally as the dialogue progresses. Additionally, it can also optionally output the conversation history as table.

    To ensure effective agent behavior, provide meaningful tool names and clear descriptions — including example use cases if applicable.

    This node is very similar to the **Agent Chat View** but additionally outputs the conversation, tool output data and the combined tools workflow. The outputs are updated on re-execution, either implicitly when being used within a dataapp, or explicitly after each completed response.
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
            "None",
            "Node re-execution is never triggered explicitly from within the chat. But the node will still be re-executed (and the outputs updated) when being used within a dataapp.",
        )
        INTERACTION = (
            "Response completed",
            "Re-execute the node once the last response has been received.",
        )

    reexecution_trigger = knext.EnumParameter(
        "Re-execution trigger",
        "The user action that triggers a re-execution of the node in order to update the conversation, tool output data and combined tools workflow. Re-execution also helps to retain the conversation of the chat view which is otherwise cleared as soon as the chat view is closed.",
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

    has_error_column = knext.BoolParameter(
        "Output errors",
        "If checked, the output table will contain an additional column that contains error messages. "
        "Each row that contains an error message will have a missing value in the conversation column.",
        default_value=False,
        since_version="5.10.0",
        is_advanced=True,
    )

    error_column_name = knext.StringParameter(
        "Error column name",
        "Name of the error column in the output table.",
        default_value="Errors",
        since_version="5.10.0",
        is_advanced=True,
    ).rule(knext.OneOf(has_error_column, [True]), knext.Effect.SHOW)

    recursion_limit = _recursion_limit_parameter()

    recursion_limit_handling = recursion_limit_mode_param_for_view()

    keep_failed_tools = knext.BoolParameter(
        "Keep failed tools",
        "If checked, failed tool calls will be kept in the combined tools workflow being output by the node.",
        default_value=True,
        is_advanced=True,
        since_version="5.10.0",
    )

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

        columns = [knext.Column(_message_type(), self.conversation_column_name)]
        if self.has_error_column:
            if self.conversation_column_name == self.error_column_name:
                raise knext.InvalidParametersError(
                    "The conversation and error column names must not be equal."
                )
            columns.append(knext.Column(knext.string(), self.error_column_name))

        return (
            None,  # combined tools workflow
            knext.Schema.from_columns(columns),
            [None] * ctx.get_connected_output_port_numbers()[2],
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        tools_table: Optional[knext.Table],
        input_tables: list[knext.Table],
    ):
        import pandas as pd
        from ._data_service import DataRegistry
        import pyarrow as pa

        view_data = ctx._get_view_data()
        num_data_outputs = ctx.get_connected_output_port_numbers()[2]
        combined_tools_workflow = ctx._get_combined_tools_workflow()

        if view_data:
            conversation_table = view_data["ports"][0]
            data_registry = DataRegistry.load(
                view_data["data"]["data_registry"], view_data["ports_for_ids"]
            )
            return (
                combined_tools_workflow,
                conversation_table,
                data_registry.get_last_tables(num_data_outputs),
            )
        else:
            message_type = _message_type()
            columns = [
                util.OutputColumn(
                    self.conversation_column_name,
                    message_type,
                    message_type.to_pyarrow(),
                )
            ]
            if self.has_error_column:
                columns.append(
                    util.OutputColumn(
                        self.error_column_name,
                        knext.string(),
                        pa.string(),
                    )
                )
            conversation_table = util.create_empty_table(None, columns)
            return (
                combined_tools_workflow,
                conversation_table,
                [
                    knext.Table.from_pandas(pd.DataFrame())  # empty table
                ]
                * num_data_outputs,
            )

    def get_data_service(
        self,
        ctx,
        chat_model: ChatModelPortObject,
        tools_table: Optional[knext.Table],
        input_tables: list[knext.Table],
    ):
        from ._agent import AgentConfig
        from ._data_service import (
            DataRegistry,
            LangchainToolConverter,
            AgentChatWidgetDataService,
        )
        from ._tool import ExecutionMode

        execution_mode = ExecutionMode.DEBUG if self.debug else ExecutionMode.DETACHED

        view_data = ctx._get_view_data()

        chat_model = chat_model.create_model(
            ctx, output_format=OutputFormatOptions.Text
        )

        if view_data is None:
            project_id, workflow_id, input_ids = ctx._init_combined_tools_workflow(
                input_tables, execution_mode.name, not self.keep_failed_tools
            )
            data_registry = DataRegistry.create_with_input_tables(
                input_tables,
                input_ids,
                data_message_prefix=self.data_message_prefix,
            )
        else:
            project_id, workflow_id, input_ids = ctx._init_combined_tools_workflow(
                [], execution_mode.name, not self.keep_failed_tools
            )
            data_registry = DataRegistry.load(
                view_data["data"]["data_registry"], view_data["ports_for_ids"]
            )

        tool_converter = LangchainToolConverter(
            data_registry, ctx, execution_mode, self.show_views, True
        )
        if tools_table is not None:
            tool_cells = _extract_tools_from_table(tools_table, self.tool_column)
            tools = [tool_converter.to_langchain_tool(tool) for tool in tool_cells]
        else:
            tools = []

        conversation = self._create_conversation_history(
            view_data, data_registry, tool_converter
        )

        toolset = AgentPrompterToolset(tools)
        config = AgentConfig(self.recursion_limit)

        return AgentChatWidgetDataService(
            ctx,
            chat_model,
            conversation,
            toolset,
            config,
            data_registry,
            self.initial_message,
            self.conversation_column_name,
            self.recursion_limit_handling,
            self.show_tool_calls_and_results,
            self.reexecution_trigger,
            tool_converter,
            {
                "project_id": project_id,
                "workflow_id": workflow_id,
            },
            self.has_error_column,
            self.error_column_name,
        )

    def _create_conversation_history(self, view_data, data_registry, tool_converter):
        from knime.types.message import to_langchain_message
        from langchain_core.messages import SystemMessage
        import pandas as pd

        # error_handling=None so that appending errors will not raise
        conversation = AgentPrompterConversation(None)
        previous_messages = False

        if self.developer_message:
            conversation.append_messages(SystemMessage(self.developer_message))

        if view_data is not None:
            conversation_table = view_data["ports"][0]
            if conversation_table is not None:
                columns = [self.conversation_column_name]

                if self.has_error_column:
                    columns.append(self.error_column_name)

                conversation_df = conversation_table[columns].to_pandas()

                for i, msg in enumerate(conversation_df[self.conversation_column_name]):
                    if pd.notna(msg):
                        lc_msg = to_langchain_message(msg)
                        lc_msg = tool_converter.sanitize_tool_names(lc_msg)
                        conversation.append_messages(lc_msg)
                        previous_messages = True
                    elif self.has_error_column:
                        error_msg = conversation_df[self.error_column_name].iloc[i]
                        conversation.append_error(Exception(error_msg))

        if not previous_messages and (
            data_registry.has_data or tool_converter.has_data_tools
        ):
            conversation.append_messages(data_registry.create_data_message())

        return conversation


# endregion
