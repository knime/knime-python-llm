import knime.extension as knext
import pandas as pd
import os

from indexes.faiss import (
    FAISSVectorstorePortObjectContent,
    FAISSVectorstorePortObjectSpecContent,
)
from models.base import (
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    chat_model_port_type,
)
from indexes.base import (
    VectorStorePortObjectSpec,
    VectorStorePortObject,
    vector_store_port_type,
    tool_list_port_type,
    ToolListPortObject,
    ToolListPortObjectSpec,
)

from langchain import LLMMathChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, Tool, AgentType, ZeroShotAgent, AgentExecutor, ConversationalChatAgent, ConversationalAgent
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory


langchain_icon = ""
agent_category = ""

import logging

LOGGER = logging.getLogger(__name__)


class AgentConnectionSpec(knext.PortObjectSpec):
    def __init__(self) -> None:
        super().__init__()

    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: dict) -> "AgentConnectionSpec":
        return cls()


class AgentConnectionObject(knext.ConnectionPortObject):
    def __init__(self, spec: AgentConnectionSpec, agent) -> None:
        super().__init__(spec)
        self._agent = agent

    def serialize(self):
        return {"agent": self._agent}

    @property
    def spec(self) -> AgentConnectionSpec:
        return super().spec

    @property
    def agent(self):
        return self._agent

    @classmethod
    def deserialize(cls, spec: AgentConnectionSpec, data) -> "AgentConnectionObject":
        return cls(spec, data["agent"])


agent_connection_port_type = knext.port_type(
    "Agent Port Type",
    AgentConnectionObject,
    AgentConnectionSpec,
)


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing Google Search API key",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


@knext.node(
    "ChatBot Agent Creator",
    knext.NodeType.PREDICTOR,
    langchain_icon,
    category=agent_category,
)
@knext.input_port("LLM", "The large language model to chat with.", llm_port_type)
@knext.input_port("Vectorstore", "Vectorstore input.", vector_store_port_type)
@knext.output_port("Agent", "Outputs a chatbot agent.", agent_connection_port_type)
class ChatBotAgentCreator:
    def configure(
        self,
        ctx: knext.ConfigurationContext,
        vectorstore: VectorStorePortObjectSpec,
        chatmodel: ChatModelPortObjectSpec,
    ):
        return AgentConnectionSpec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chatmodel_port: ChatModelPortObject,
        vectorstore: VectorStorePortObject,
    ):
        chatmodel = chatmodel_port.create_model(ctx)
        memory = ConversationBufferMemory(memory_key="chat_history")

        db = vectorstore.load_store(ctx)
        node_descriptions = RetrievalQA.from_chain_type(
            llm=chatmodel,
            chain_type="refine",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
        )
        tools = []

        # TODO: remove and get it from the vectorstoreportobject
        tools.append(
            Tool(
                name="Node Descriptions QA System",
                func=node_descriptions.run,
                description="useful for when you need to answer questions about how to configure a node in KNIME. Input should be a fully formed question.",
            )
        )

        agent = initialize_agent(
            tools,
            chatmodel,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
        )

        return AgentConnectionObject(AgentConnectionSpec(), agent)


@knext.node(
    "ChatBot Agent Executor",
    knext.NodeType.PREDICTOR,
    langchain_icon,
    category=agent_category,
)
@knext.input_port("Chatbot", "Chatbot agent", agent_connection_port_type)
@knext.output_port("Chatbot", "Outputs a chatbot agent.", agent_connection_port_type)
@knext.output_table("Response", "Outputs chatbot agent's response.")
class ChatBotAgentExecutor:
    message = knext.StringParameter("Chat input", "Human chat message")

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chatbot_spec: AgentConnectionSpec,
    ):
        response_table = knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Message"),
                knext.Column(knext.string(), "Response"),
            ]
        )
        return chatbot_spec, response_table

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chatbot: AgentConnectionObject,
    ):
        response = chatbot.agent.run(input=self.message)
        response_table = pd.DataFrame()

        response_table["Message"] = self.message
        response_table["Response"] = response

        return chatbot, knext.Table.from_pandas(response_table)


@knext.node(
    "ChatBot Agent Creator 2",
    knext.NodeType.PREDICTOR,
    langchain_icon,
    category=agent_category,
)
@knext.input_port("Chat", "The large language model to chat with.", chat_model_port_type)
@knext.input_port("Tool List", "Vectorstore input.", tool_list_port_type)
@knext.output_port("Agent", "Outputs a chatbot agent.", agent_connection_port_type)
class ChatBotAgentCreatorTwo:
    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chatmodel: ChatModelPortObjectSpec,
        tool_list_spec: ToolListPortObjectSpec,
    ):
        return AgentConnectionSpec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chatmodel_port: ChatModelPortObject,
        tool_list_port: ToolListPortObject,
    ):
        chatmodel = chatmodel_port.create_model(ctx)

        tools = []
        for tool in tool_list_port.tool_list:
            tools.append(tool.create_tool(ctx))


        from langchain.memory import ChatMessageHistory, ConversationBufferMemory
        prefix = """Have a conversation with a human, answering the following questions as best you can and like a pirate. You have access to the following tools:"""
        suffix = """Begin!"
            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)    

        llm_chain = LLMChain(llm=chatmodel_port.create_model(ctx), prompt=prompt)

        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=["Youtube Transcript QA System", "Node Description QA System"], verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

        agent_chain.run(input="How to filter KNIME columns with regex for columns which have names starting with a letter 's' and which node would we need for that?")
        #LOGGER.info(agent_chain.run(input=""))

        agent_chain.run(input="Using the node you suggested, what columns will stay in the table if my columns are named 'germany', 'austria', 'switzerland'")
        # LOGGER.info(agent_chain.memory.buffer)


        #agent = initialize_agent(
        #    tools,
        #    chatmodel,
        #    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        #    verbose=True,
        #    memory=memory_list
        #)



        #LOGGER.info(agent.run(
        #    "How to filter KNIME columns with regex?"))
        #LOGGER.info(agent.run("What did biden say about ketanji brown jackson in the state of the union address?"))

        return AgentConnectionObject(AgentConnectionSpec(), agent)
    

