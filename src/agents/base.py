import knime.extension as knext
import pandas as pd
import util


from models.base import (
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
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

import os

from langchain import PromptTemplate


import langchain.agents

from langchain.prompts import MessagesPlaceholder
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
)

agent_icon = "icons/agent.png"
agent_category = knext.category(
    path=util.main_category,
    level_id="agents",
    name="Agents",
    description="",
    icon=agent_icon,
)


# agents = {
#     "CONVERSATIONAL_REACT_DESCRIPTION": AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     "REACT": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     "REACT_DOCUMENTSTORE": AgentType.REACT_DOCSTORE,
# }


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing Google Search API key.",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


@knext.parameter_group(label="Chat History Settings")
class ChatHistorySettings:
    type_column = knext.ColumnParameter(
        "Message Type", "Specifies the sender of the messages.", port_index=1
    )

    message_column = knext.ColumnParameter(
        "Messages",
        "Represents the chat messages exchanged between the agent and the user.",
        port_index=1,
    )

    history_length = knext.IntParameter(
        "History length",
        """Specifies the number of chat messages to be considered as history in the next prompt.
        It is recommended to keep this value relatively low to manage token usage efficiently.
        A higher value will consume more memory and tokens.""",
        default_value=5,
        is_advanced=True,
    )


@knext.parameter_group(label="Chat Message Settings")
class ChatMessageSettings:
    chat_message = knext.StringParameter(
        "Chat message", "Message to send to the agent."
    )

    # TODO this should be in the agent creator
    system_prefix = knext.StringParameter(
        "Agent prompt prefix",
        """The prefix that will be added to the agent prompt to guide its behavior. 
        For example, you can set it to 'You are a friendly KNIME assistant' to shape 
        the agent's response accordingly.""",
    )


# TODO: Currently not used
@knext.parameter_group(label="LMM Agent Type Selection")
class LLMAgentTypeSettings:
    class AgentOptions(knext.EnumParameterOptions):
        CONVERSATIONAL_REACT_DESCRIPTION = (
            "Conversational react description",
            """An agent optimized for conversation. 
            Other agents are often optimized for using tools to figure 
            out the best response, which is not ideal in a conversational 
            setting where you may want the agent to be able to chat with the user as well.
            
            For more information see 
            [LangChain documentation](https://python.langchain.com/docs/modules/agents/agent_types/chat_conversation_agent)
            """,
        )
        REACT = (
            "ReAct",
            """Implements the [ReAct paper](https://react-lm.github.io/). 'ReAct prompting is intuitive and flexible to design, 
            and achieves state-of-the-art few-shot performances across a variety of tasks, from question answering to online shopping'

            For more information see 
            [LangChain documentation](https://python.langchain.com/docs/modules/agents/agent_types/react)
            """,
        )
        REACT_DOCUMENTSTORE = (
            "ReAct document store",
            """Implement the [ReAct logic](https://react-lm.github.io/) for working with document 
            store specifically.""",
        )

    agent = knext.EnumParameter(
        "Agent Type",
        "Choose between agents that are optimized for different tasks",
        AgentOptions.CONVERSATIONAL_REACT_DESCRIPTION.name,
        AgentOptions,
    )


@knext.parameter_group(label="Memory Type Selection")
class MemoryTypeSettings:
    class MemoryOptions(knext.EnumParameterOptions):
        CONVERSATION_BUFFER_MEMORY = (
            "Conversation Buffer Memory",
            """Simple yet effective method of storing and recalling past conversations
            between a human and an AI. It retains the entire history of the conversation,
            allowing the AI to access and reference previous exchanges with each new
            message. This ensures that the context of the ongoing conversation is
            preserved and readily available. However, it's important to note that
            using this method may lead to a rapid increase in token usage as the
            conversation progresses.""",
        )
        CONVERSATION_BUFFER_WINDOW_MEMORY = (
            "Conversation Buffer Window Memory",
            """A more optimized approach to storing and accessing past conversations
            between a human and an AI. While it retains the input of the entire conversation,
            it only sends a specified number of recent messages (the "history length") with each
            new message. This controlled window of conversation history helps manage token
            usage more efficiently, ensuring that the growth of tokens remains predictable
            and within a predetermined limit. By limiting the number of messages sent, this
            method strikes a balance between preserving context and controlling resource consumption.""",
        )

    memory = knext.EnumParameter(
        "Memory Type",
        "Choose between different memory implementations.",
        MemoryOptions.CONVERSATION_BUFFER_WINDOW_MEMORY.name,
        MemoryOptions,
    )


# TODO: Add agent type in the future?
class AgentPortObjectSpec(knext.PortObjectSpec):
    def __init__(
        self,
        memory_type,
        llm_spec: LLMPortObjectSpec,
        tools_spec: ToolListPortObjectSpec,
    ) -> None:
        self._memory_type = memory_type
        self._llm_spec = llm_spec
        self._llm_type = get_port_type_for_spec_type(type(llm_spec))
        self._tools_spec = tools_spec
        self._tools_type = get_port_type_for_spec_type(type(tools_spec))

    @property
    def llm_spec(self) -> LLMPortObjectSpec:
        return self._llm_spec

    @property
    def llm_type(self) -> knext.PortType:
        return self._llm_type

    @property
    def tools_spec(self) -> ToolListPortObjectSpec:
        return self._tools_spec

    @property
    def tools_type(self) -> knext.PortType:
        return self._tools_type

    @property
    def memory_type(self):
        return self._memory_type

    def serialize(self) -> dict:
        return {
            "memory_type": self._memory_type,
            "llm": {"type": self.llm_type.id, "spec": self.llm_spec.serialize()},
            "tools": {"type": self.tools_type.id, "spec": self.tools_spec.serialize()},
        }

    @classmethod
    def deserialize(cls, data: dict) -> "AgentPortObjectSpec":
        llm_data = data["llm"]
        llm_spec = get_port_type_for_id(llm_data["type"]).spec_class.deserialize(
            llm_data["spec"]
        )
        tools_data = data["tools"]
        tools_spec = get_port_type_for_id(tools_data["type"]).spec_class.deserialize(
            tools_data["spec"]
        )
        return cls(data["memory_type"], llm_spec, tools_spec)


class AgentPortObject(FilestorePortObject):
    def __init__(
        self,
        spec: AgentPortObjectSpec,
        llm: LLMPortObject,
        tools: ToolListPortObject,
    ) -> None:
        super().__init__(spec)
        self._llm = llm
        self._tools = tools

    @property
    def llm(self) -> LLMPortObject:
        return self._llm

    @property
    def tools(self) -> ToolListPortObject:
        return self._tools

    def write_to(self, file_path):
        os.makedirs(file_path)
        llm_path = os.path.join(file_path, "llm")
        save_port_object(self.llm, llm_path)
        tools_path = os.path.join(file_path, "tools")
        save_port_object(self.tools, tools_path)

    @classmethod
    def read_from(cls, spec: AgentPortObjectSpec, file_path: str) -> "AgentPortObject":
        llm_path = os.path.join(file_path, "llm")
        llm = load_port_object(spec.llm_type.object_class, spec.llm_spec, llm_path)
        tools_path = os.path.join(file_path, "tools")
        tools = load_port_object(
            spec.tools_type.object_class, spec.tools_spec, tools_path
        )
        return cls(spec, llm, tools)


agent_port_type = knext.port_type("Agent", AgentPortObject, AgentPortObjectSpec)


@knext.node(
    "LLM Agent Creator",
    knext.NodeType.SOURCE,
    agent_icon,
    category=agent_category,
)
@knext.input_port("LLM", "A large language model.", llm_port_type)
@knext.input_port(
    "Tool List", "A list of tools for the agent to use.", tool_list_port_type
)
@knext.output_port("Agent", "A configured agent.", agent_port_type)
class LLMAgentCreator:
    """
    Creates an LLM-based agent equipped with the provided tools.

    Currently, only the **Conversational** agent type is available.

    [Conversational agents](https://python.langchain.com/docs/modules/agents/agent_types/chat_conversation_agent)
    are optimized for conversation. They expect to be used with a memory component.

    """

    memory_type = MemoryTypeSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm_spec: LLMPortObjectSpec,
        tools_spec: ToolListPortObjectSpec,
    ):
        return AgentPortObjectSpec(self.memory_type.memory, llm_spec, tools_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm: LLMPortObject,
        tools: ToolListPortObject,
    ):
        return AgentPortObject(
            AgentPortObjectSpec(self.memory_type.memory, llm.spec, tools.spec),
            llm,
            tools,
        )


@knext.node(
    "Agent Executor",
    knext.NodeType.PREDICTOR,
    agent_icon,
    category=agent_category,
)
@knext.input_port("Agent", "Configured agent.", agent_port_type)
@knext.input_table("Chat History", "Table containing the chat history for the agent.")
@knext.output_table("Chat History", "Table containing the chat history for the agent.")
class AgentExecutor:
    """
    Executes a chat agent equipped with tools and memory.

    The memory table is expected to have at least two string columns and be
    either empty or filled by a previous agent execution.

    """

    history_settings = ChatHistorySettings()
    message_settings = ChatMessageSettings()

    def load_messages_from_input_table(
        self, memory: ConversationBufferMemory, chat_history_df: pd.DataFrame
    ):
        for index in range(0, len(chat_history_df), 2):
            memory.save_context(
                {"input": chat_history_df.loc[f"Row{index}"].at["Message"]},
                {"output": chat_history_df.loc[f"Row{index+1}"].at["Message"]},
            )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        agent_spec: AgentPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        if len(input_table_spec.column_names) < 2:
            raise knext.InvalidParametersError(
                "Please provide at least two String columns."
            )

        if self.history_settings.type_column == self.history_settings.message_column:
            raise knext.InvalidParametersError(
                "Type and Message columns cannot be the same."
            )

        for c in input_table_spec:
            if (
                c.name == self.history_settings.type_column
                or c.name == self.history_settings.message_column
            ):
                if not util.is_nominal(c):
                    raise knext.InvalidParametersError(
                        f"{c.name} has to be a String column."
                    )

        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), self.history_settings.type_column),
                knext.Column(knext.string(), self.history_settings.message_column),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        agent: AgentPortObject,
        input_table: knext.Table,
    ):
        chat_history_df = input_table.to_pandas()

        template = """
        {chat_history}
        Human: {input}
        Agent:
        {agent_scratchpad}"""

        prefix = self.message_settings.system_prefix

        prompt = PromptTemplate(
            input_variables=["chat_history", "input", "agent_scratchpad"],
            template=template,
            prefix=prefix,
        )

        if agent.spec.memory_type == "CONVERSATION_BUFFER_MEMORY":
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="input",
                return_messages=True,
            )
        else:
            memory = ConversationBufferWindowMemory(
                k=self.history_settings.history_length,
                memory_key="chat_history",
                input_key="input",
                return_messages=True,
            )

        if (
            not chat_history_df[
                [
                    self.history_settings.type_column,
                    self.history_settings.message_column,
                ]
            ]
            .isna()
            .any()
            .any()
        ):
            self.load_messages_from_input_table(memory, chat_history_df)

        tool_list = agent.tools.tools
        tools = [tool.create(ctx) for tool in tool_list]

        llm = agent.llm.create_model(ctx)

        chat_history = MessagesPlaceholder(variable_name="chat_history")

        agent = langchain.agents.ConversationalAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            prefix=prefix,
            input_variables=["chat_history", "input", "agent_scratchpad"],
        )

        agent_exec = langchain.agents.AgentExecutor(
            memory=memory, agent=agent, tools=tools, verbose=True
        )

        response = agent_exec.run(
            input=self.message_settings.chat_message,
            chat_history=chat_history,
            prompt=prompt,
        )

        new_df = chat_history_df[
            [self.history_settings.type_column, self.history_settings.message_column]
        ].copy()

        user_input_row = ["Human", self.message_settings.chat_message]
        agent_output_row = ["AI", response]

        new_df.loc[f"Row{len(new_df)}"] = user_input_row
        new_df.loc[f"Row{len(new_df)}"] = agent_output_row

        return knext.Table.from_pandas(new_df)
