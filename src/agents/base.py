# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods

import knime.extension as knext
import pandas as pd


from models.base import (
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    chat_model_port_type,
)
from indexes.base import (
    tool_list_port_type,
    ToolListPortObject,
    ToolListPortObjectSpec,
)
import util

from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import  initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

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

@knext.node(
    "ChatBot Agent",
    knext.NodeType.PREDICTOR,
    agent_icon,
    category=agent_category,
)
@knext.input_port("Chat", "The large language model to chat with.", chat_model_port_type)
@knext.input_port("Tool List", "Vectorstore input.", tool_list_port_type)
@knext.input_table("Chat History", "Table containing the chat history for the agent.")
@knext.output_table("Chat History", "Table containing the chat history for the agent.")
class ChatBotAgent:

    chat_message = knext.StringParameter(
        label="Chat message",
        description="Message to send to the Chat Bot"
    )

    system_prefix = knext.StringParameter(
        label="Agent prompt prefix",
        description="The prefix will be used for better control what its doing. Example: 'You are a friendly assisstant'"
    )

    def load_messages_from_input_table(self, memory: ConversationBufferMemory, chat_history_df: pd.DataFrame):
        
        for index in range(0, len(chat_history_df), 2):
            memory.save_context(
                {"input": chat_history_df.loc[f"Row{index}"].at["Message"]}, 
                {"output": chat_history_df.loc[f"Row{index+1}"].at["Message"]}
        )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chatmodel: ChatModelPortObjectSpec,
        tool_list_spec: ToolListPortObjectSpec,
        input_table_spec: knext.Schema,
    ):

        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Type"),
            knext.Column(knext.string(), "Message")
        ])

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chatmodel_port: ChatModelPortObject,
        tool_list_port: ToolListPortObject,
        input_table: knext.Table
    ):
        
        chat_history_df = input_table.to_pandas()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        memory.save_context(
                {"system": self.system_prefix}, {"outputs": ""}
            )
        
        if all(col_name in ["Type", "Message"] for col_name in input_table.column_names):
            self.load_messages_from_input_table(memory, chat_history_df)
        else:
            chat_history_df[["Type", "Message"]] = None

        LOGGER.info(chat_history_df)
        tool_list = tool_list_port.tool_list
        tools = [ tool.create(ctx) for tool in tool_list ]

        chatmodel = chatmodel_port.create_model(ctx)
        chat_history = MessagesPlaceholder(variable_name="chat_history")

        agent_chain = initialize_agent(
            tools, 
            chatmodel, 
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            agent_kwargs = {
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"]
            },
            handle_parsing_errors="Check your output and make sure it conforms!"
        )

        answer = agent_chain.run(input=self.chat_message)

        user_input_row = ["", "input", self.chat_message]
        agent_output_row = ["", "output", answer]

        chat_history_df.loc[f'Row{len(chat_history_df)}'] = user_input_row
        chat_history_df.loc[f'Row{len(chat_history_df)}'] = agent_output_row

        return knext.Table.from_pandas(chat_history_df)