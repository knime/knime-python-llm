import knime.extension as knext
import util


from models.base import LLMPortObjectSpec, LLMPortObject, ChatConversationSettings
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


@knext.parameter_group(label="Prompt Settings")
class ChatMessageSettings:
    message = knext.MultilineStringParameter(
        "New message", "Specify the new message to be sent to the agent."
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

    memory_window_size = knext.IntParameter(
        "Memory window size",
        """Specifies the number chat messages the Conversation Buffer Window Memory includes. Only active for Conersation Buffer Window Memory.""",
        default_value=5,
        is_advanced=True,
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
    def deserialize_llm_spec(cls, data: dict, java_callback):
        llm_data = data["llm"]
        spec_type = get_port_type_for_id(llm_data["type"])
        try:
            return spec_type.spec_class.deserialize(llm_data["spec"], java_callback)
        except TypeError:
            return spec_type.spec_class.deserialize(llm_data["spec"])

    @classmethod
    def deserialize(cls, data: dict, java_callback) -> "AgentPortObjectSpec":
        return cls(cls.deserialize_llm_spec(data, java_callback))


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


# @knext.node(
#     "LLM Agent Creator",
#     knext.NodeType.SOURCE,
#     agent_icon,
#     category=agent_category,
# )
# @knext.input_port("LLM", "A large language model.", llm_port_type)
# @knext.input_port(
#     "Tool List", "A list of tools for the agent to use.", tool_list_port_type
# )
# @knext.output_port("Agent", "A configured agent.", agent_port_type)
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
        llm_spec.validate_context(ctx)
        tools_spec.validate_context(ctx)
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

        # if agent.spec.memory_type == "CONVERSATION_BUFFER_MEMORY":
        #     memory = ConversationBufferMemory(
        #         memory_key="chat_history",
        #         input_key="input",
        #         return_messages=True,
        #     )
        # else:
        #     memory = ConversationBufferWindowMemory(
        #         k=self.history_settings.history_length,
        #         memory_key="chat_history",
        #         input_key="input",
        #         return_messages=True,
        #     )

        # TODO more memory variants
        # TODO return messages might depend on the type of model (i.e. chat needs it llm doesn't)?

        import langchain
        import langchain.agents
        from langchain.memory import ConversationBufferMemory

        langchain.debug = self.enable_debug_output
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        messages = self.conversation_settings.create_messages(chat_history_df)

        for message in messages:
            memory.chat_memory.add_message(message)

        tools = tools_obj.create_tools(ctx)
        agent = agent_obj.create_agent(ctx, tools)

        agent_exec = langchain.agents.AgentExecutor(
            memory=memory, agent=agent, tools=tools
        )

        response = agent_exec.run(input=self.message_settings.message)

        user_input_row = ["Human", self.message_settings.message]
        agent_output_row = ["AI", response]

        chat_history_df.loc[f"Row{len(chat_history_df)}"] = user_input_row
        chat_history_df.loc[f"Row{len(chat_history_df)}"] = agent_output_row

        return knext.Table.from_pandas(chat_history_df)
