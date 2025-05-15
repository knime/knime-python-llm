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
agent_category = knext.category(
    path=util.main_category,
    level_id="agents",
    name="Agents",
    description="",
    icon=agent_icon,
)


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


@knext.node(
    "Agent Prompter 2.0",
    node_type=knext.NodeType.PREDICTOR,
    icon_path=agent_icon,
    category=agent_category,
)
@knext.input_port(
    "Chat model", "The chat model to use.", port_type=chat_model_port_type
)
@knext.input_table("Tools", "The tools the agent can use.")
@knext.output_table("Result", "The result of the agent execution.")
@knext.input_table_group(
    "Data inputs",
    "The data inputs for the agent.",
)
@knext.output_table_group(
    "Data outputs",
    "The data outputs of the agent.",
)
class AgentPrompter2:
    # TODO better name + description
    developer_message = knext.MultilineStringParameter(
        "Developer message",
        "Message provided to the agent that instructs it how to act.",
    )

    # TODO better name + description
    # TODO or does it come from the input table?
    user_message = knext.MultilineStringParameter(
        "User message",
        "Message provided to the agent that instructs it how to act.",
    )

    # TODO type filter for tool columns. How to declare the type?
    tool_column = knext.ColumnParameter(
        "Tool column", "The column holding the tools the agent can use.", port_index=1
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chat_model_spec: ChatModelPortObjectSpec,
        tools_schema: knext.Schema,
        data_schemas: list[knext.Schema],
    ) -> knext.Schema:
        chat_model_spec.validate_context(ctx)
        self._configure_tool_tables(tools_schema)

        return self._create_conversation_schema(), [
            None
        ] * ctx.get_connected_output_port_numbers()[1]

    def _configure_tool_tables(self, tools_schema):
        if self.tool_column is None:
            # TODO instead pick the last tool column from the table. Similar to the type filter, we need the type for this
            raise knext.InvalidParametersError("Select the column holding the tools.")
        elif self.tool_column not in tools_schema.column_names:
            raise knext.InvalidParametersError(
                f"Column {self.tool_column} not found in the tools table."
            )

    def _create_conversation_schema(self) -> knext.Schema: ...

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        tools_table: knext.Table,
        input_tables: list[knext.Table],
    ):
        from langchain.chat_models.base import BaseChatModel
        from langgraph.prebuilt import create_react_agent
        import pandas as pd
        from ._agent_impl import (
            DataRegistry,
            LangchainToolConverter,
            _render_message_as_json,
        )

        data_registry = DataRegistry(input_tables)
        tool_converter = LangchainToolConverter(
            data_registry, ctx, _render_message_as_json
        )

        # TODO check if JSON output format is compatible with tool calling
        chat_model: BaseChatModel = chat_model.create_model(
            ctx, output_format=OutputFormatOptions.Text
        )

        tool_cells = _extract_tools_from_table(tools_table, self.tool_column)

        tools = [tool_converter.to_langchain_tool(tool) for tool in tool_cells]
        graph = create_react_agent(
            chat_model, tools=tools, prompt=self.developer_message
        )

        initial_message = _render_message_as_json(
            initial_data=data_registry.llm_representation(),
            user_message=self.user_message,
        )

        inputs = {"messages": [{"role": "user", "content": initial_message}]}
        final_state = graph.invoke(inputs)
        messages = final_state["messages"]
        result_df = pd.DataFrame(
            {
                "Role": [msg.type for msg in messages],
                "Content": [msg.content for msg in messages],
                "Tool Calls": [
                    str(msg.tool_calls) if hasattr(msg, "tool_calls") else None
                    for msg in messages
                ],
            }
        )
        conversation_table = knext.Table.from_pandas(result_df)
        num_data_outputs = ctx.get_connected_output_port_numbers()[1]

        if num_data_outputs == 0:
            return conversation_table
        # TODO allow the model to pick the outputs
        # return the last n tables from the data registry
        return conversation_table, [
            item.data for item in data_registry._data[-num_data_outputs:]
        ]


def _extract_tools_from_table(tools_table: knext.Table, tool_column: str):
    tools_df = tools_table[tool_column].to_pandas()
    tool_list = tools_df[tool_column].tolist()
    return tool_list


@knext.node(
    "Chat Agent Prompter",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=agent_icon,
    category=agent_category,
)
@knext.input_port(
    "Chat model", "The chat model to use.", port_type=chat_model_port_type
)
@knext.input_table("Tools", "The tools the agent can use.")
@knext.input_table_group(
    "Data inputs",
    "The data inputs for the agent.",
)
@knext.output_view("KNIME Icon", "Shows the KNIME icon", static_resources="assets")
class ChatAgentPrompter:
    # TODO better name + description
    developer_message = knext.MultilineStringParameter(
        "Developer message",
        "Message provided to the agent that instructs it how to act.",
    )

    # TODO type filter for tool columns. How to declare the type?
    tool_column = knext.ColumnParameter(
        "Tool column", "The column holding the tools the agent can use.", port_index=1
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chat_model_spec: ChatModelPortObjectSpec,
        tools_schema: knext.Schema,
        data_schemas: list[knext.Schema],
    ) -> knext.Schema:
        if self.tool_column is None:
            # TODO instead pick the last tool column from the table (requires type)
            self.tool_column = tools_schema.column_names[-1]

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        tools_table: knext.Table,
        input_tables: list[knext.Table],
    ):
        # Load the index.html file as a string
        # Assume the repository root is two levels up from this file
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        html_file_path = os.path.join(repo_root, "index.html")

        with open(html_file_path, "r", encoding="utf-8") as html_file:
            html_content = html_file.read()

        return knext.view_html(html_content)

    def get_data_service(
        self,
        ctx,
        chat_model: ChatModelPortObject,
        tools_table: knext.Table,
        input_tables: list[knext.Table],
    ):
        from langgraph.prebuilt import create_react_agent
        from ._agent_impl import (
            DataRegistry,
            LangchainToolConverter,
            ChatAgentPrompterDataService,
            _render_message_as_json,
        )

        chat_model = chat_model.create_model(
            ctx, output_format=OutputFormatOptions.Text
        )
        data_registry = DataRegistry(input_tables)
        tool_converter = LangchainToolConverter(
            data_registry, ctx, _render_message_as_json
        )
        tool_cells = _extract_tools_from_table(tools_table, self.tool_column)
        tools = [tool_converter.to_langchain_tool(tool) for tool in tool_cells]
        agent = create_react_agent(
            chat_model, tools=tools, prompt=self.developer_message
        )

        return ChatAgentPrompterDataService(agent, data_registry)
