import knime.extension as knext
import pandas as pd
import os

from indexes.faiss import (
    FAISSVectorstorePortObjectContent,
    FAISSVectorstorePortObjectSpecContent,
)
from models.base import LLMPortObjectSpec, LLMPortObject, llm_port_type
from indexes.base import (
    VectorStorePortObjectSpec,
    VectorStorePortObject,
    vector_store_port_type,
)

from langchain import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, Tool, AgentType
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
        llm: LLMPortObjectSpec,
    ):
        return AgentConnectionSpec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm_port: LLMPortObject,
        vectorstore: VectorStorePortObject,
    ):
        llm = llm_port.create_model(ctx)
        memory = ConversationBufferMemory(memory_key="chat_history")

        db = vectorstore.load_store(ctx)
        node_descriptions = RetrievalQA.from_chain_type(
            llm=llm,
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
            llm,
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
