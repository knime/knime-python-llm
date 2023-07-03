# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods

import knime.extension as knext
import pandas as pd


from models.base import (
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    chat_model_port_type,
)
from indexes.base import (
    tool_list_port_type,
    ToolListPortObject,
    ToolListPortObjectSpec,
)
import util

from langchain.prompts import MessagesPlaceholder
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
)
from langchain.agents import initialize_agent, AgentType


import pickle


agent_icon = "icons/agent.png"
agent_category = knext.category(
    path=util.main_cat,
    level_id="agents",
    name="Agents",
    description="",
    icon=agent_icon,
)

import logging

LOGGER = logging.getLogger(__name__)


# TODO: Add more agents that behave differently e.g. no tools, with specific tools, or have agent types in the config option
@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing Google Search API key",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


@knext.parameter_group(label="Chat History Settings")
class ChatHistorySettings:
    type_column = knext.ColumnParameter(
        "Message Type", "Specifies the sender of the messages", port_index=2
    )

    message_column = knext.ColumnParameter(
        "Messages", "Message sent to and from the agent", port_index=2
    )

    history_length = knext.IntParameter(
        "History length",
        """How many chat messages will be considered as history in the next prompt.
        It is recommended to keep this fairly low, as more memory will increase the token
        usage.
        """,
        default_value=5,
        is_advanced=True,
    )


@knext.parameter_group(label="Chat Message Settings")
class ChatMessageSettings:
    chat_message = knext.StringParameter(
        "Chat message", "Message to send to the Chat Bot"
    )

    system_prefix = knext.StringParameter(
        "Agent prompt prefix",
        "The prefix will be used for better control what its doing. Example: 'You are a friendly assisstant'",
    )


# TODO: Better descriptions
# TODO: Change name of node
@knext.node(
    "ChatBot Agent Executor",
    knext.NodeType.PREDICTOR,
    agent_icon,
    category=agent_category,
)
@knext.input_port(
    "Chat", "The large language model to chat with.", chat_model_port_type
)
@knext.input_port("Tool List", "Vectorstore input.", tool_list_port_type)
@knext.input_table("Chat History", "Table containing the chat history for the agent.")
@knext.output_table("Chat History", "Table containing the chat history for the agent.")
class ChatBotAgentExecutor:
    """

    Executes a chat agent equipped with tools and memory

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
        chatmodel: ChatModelPortObjectSpec,
        tool_list_spec: ToolListPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        if len(input_table_spec.column_names) < 2:
            raise ValueError("Please provide at least two string columns")

        if self.history_settings.type_column == self.history_settings.message_column:
            raise ValueError("Message Type and Messages columns can not be the same")

        for c in input_table_spec:
            if (
                c.name == self.history_settings.type_column
                or c.name == self.history_settings.message_column
            ):
                if not util.is_nominal(c):
                    raise ValueError(f"{c.name} has to be a string column.")

        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), self.history_settings.type_column),
                knext.Column(knext.string(), self.history_settings.message_column),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chatmodel_port: ChatModelPortObject,
        tool_list_port: ToolListPortObject,
        input_table: knext.Table,
    ):
        chat_history_df = input_table.to_pandas()

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=self.history_settings.history_length,
        )

        memory.save_context(
            {"system": self.message_settings.system_prefix}, {"outputs": ""}
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

        tool_list = tool_list_port.tool_list
        tools = [tool.create(ctx) for tool in tool_list]

        chatmodel = chatmodel_port.create_model(ctx)
        chat_history = MessagesPlaceholder(variable_name="chat_history")

        agent_chain = initialize_agent(
            tools,
            chatmodel,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            agent_kwargs={
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"],
            },
            handle_parsing_errors="Check your output and make sure it conforms!",
        )

        answer = agent_chain.run(input=self.message_settings.chat_message)

        new_df = chat_history_df[
            [self.history_settings.type_column, self.history_settings.message_column]
        ].copy()

        user_input_row = ["input", self.message_settings.chat_message]
        agent_output_row = ["output", answer]

        new_df.loc[f"Row{len(new_df)}"] = user_input_row
        new_df.loc[f"Row{len(new_df)}"] = agent_output_row

        return knext.Table.from_pandas(new_df)


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
            """TODO""",
        )
        CONVERSATION_BUFFER_WINDOW_MEMORY = (
            "Conversatio Buffer Window Memory",
            """TODO""",
        )

    memory = knext.EnumParameter(
        "Memory Type",
        "Choose between different memory implementations",
        MemoryOptions.CONVERSATION_BUFFER_WINDOW_MEMORY.name,
        MemoryOptions,
    )


class AgentPortObjectSpec(knext.PortObjectSpec):
    def __init__(self, agent_type, memory_type) -> None:
        super().__init__()
        self._agent_type = agent_type
        self._memory_type = memory_type

    @property
    def agent_type(self):
        return self._agent_type

    @property
    def memory_type(self):
        return self._memory_type

    def serialize(self) -> dict:
        return {"agent_type": self._agent_type, "memory_type": self._memory_type}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["agent_type"], data["memory_type"])


class AgentPortObject(knext.PortObject):
    def __init__(
        self,
        spec: AgentPortObjectSpec,
        model: LLMPortObject,
        tool_list: ToolListPortObject,
    ) -> None:
        super().__init__(spec)
        self._model = model
        self._tool_list = tool_list

    @property
    def model(self):
        return self._model

    @property
    def tool_list(self):
        return self._tool_list.tool_list

    def serialize(self):
        config = {"model": self._model, "tool_list": self._tool_list}
        return pickle.dumps(config)

    @classmethod
    def deserialize(cls, spec: AgentPortObjectSpec, data):
        config = pickle.loads(data)
        return cls(spec, config["model"], config["tool_list"])


agent_port_type = knext.port_type("Agent", AgentPortObject, AgentPortObjectSpec)


# TODO: Better descriptions
@knext.node(
    "LMM Agent Creator",
    knext.NodeType.SOURCE,
    agent_icon,
    category=agent_category,
)
@knext.input_port("LLM", "A large language model.", llm_port_type)
@knext.input_port(
    "Tool List", "A list of tools for the agent to use.", tool_list_port_type
)
@knext.output_binary("Agent", "A configured agent", agent_port_type)
class LLMAgentCreator:
    """
    Creates a llm based agent equipped with tools.
    """

    agent_type = LLMAgentTypeSettings()
    memory_type = MemoryTypeSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm_spec: LLMPortObjectSpec,
        tool_list_spec: ToolListPortObjectSpec,
    ):

        spec = AgentPortObjectSpec(
            self.agent_type.AgentOptions[self.agent_type.agent].label,
            self.memory_type.MemoryOptions[self.memory_type.memory].label,
        )
        LOGGER.info(vars(spec))
        LOGGER.info(dir(spec))

        return AgentPortObjectSpec(
            self.agent_type.AgentOptions[self.agent_type.agent].label,
            self.memory_type.MemoryOptions[self.memory_type.memory].label,
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        lmm: LLMPortObject,
        tool_list: ToolListPortObject,
    ):

        return AgentPortObject(
            AgentPortObjectSpec(
                self.agent_type.agent_type, self.memory_type.memory_type
            ),
            lmm,
            tool_list,
        )


# TODO: Better descriptions
# TODO: Change name of node
@knext.node(
    "ChatBot Agent Executor2",
    knext.NodeType.PREDICTOR,
    agent_icon,
    category=agent_category,
)
@knext.input_port("Agent", "Configured agent", agent_port_type)
@knext.input_table("Chat History", "Table containing the chat history for the agent.")
@knext.output_table("Chat History", "Table containing the chat history for the agent.")
class ChatBotAgentExecutor2:
    """
    Executes a chat agent equipped with tools and memory

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
            raise ValueError("Please provide at least two string columns")

        if self.history_settings.type_column == self.history_settings.message_column:
            raise ValueError("Message Type and Messages columns can not be the same")

        for c in input_table_spec:
            if (
                c.name == self.history_settings.type_column
                or c.name == self.history_settings.message_column
            ):
                if not util.is_nominal(c):
                    raise ValueError(f"{c.name} has to be a string column.")

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

        if isinstance(agent.spec.memory_type, ConversationBufferMemory):
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )
        else:
            memory = ConversationBufferWindowMemory(
                k=self.history_settings.history_length
            )

        memory.save_context(
            {"system": self.message_settings.system_prefix}, {"outputs": ""}
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

        tool_list = agent.tool_list
        tools = [tool.create(ctx) for tool in tool_list]

        model = agent.model.create_model(ctx)
        chat_history = MessagesPlaceholder(variable_name="chat_history")

        agent_chain = initialize_agent(
            tools,
            model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            agent_kwargs={
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"],
            },
            handle_parsing_errors="Check your output and make sure it conforms!",
        )

        answer = agent_chain.run(input=self.message_settings.chat_message)

        new_df = chat_history_df[
            [self.history_settings.type_column, self.history_settings.message_column]
        ].copy()

        user_input_row = ["input", self.message_settings.chat_message]
        agent_output_row = ["output", answer]

        new_df.loc[f"Row{len(new_df)}"] = user_input_row
        new_df.loc[f"Row{len(new_df)}"] = agent_output_row

        return knext.Table.from_pandas(new_df)
