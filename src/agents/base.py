import knime.extension as knext
from models.base import LLMPortObjectSpec, LLMPortObject, llm_port_type
from indexes.base import (
    VectorStorePortObjectSpec,
    VectorStorePortObject,
    vector_store_port_type,
)

import pandas as pd
import os

from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

langchain_icon = "./icons/langchain.png"
agent_category = ""


@knext.node(
    "Agent", knext.NodeType.PREDICTOR, langchain_icon, category=agent_category
)
@knext.input_port("LLM", "The large language model to chat with.", llm_port_type)
@knext.input_port("Vector Store","The vector store to get context for the agent.",vector_store_port_type,)
@knext.input_table("Queries", "Table containing a string column with the queries for the vectordb.")
@knext.output_table("Queries and answers", "todo")
#@knext.input_table("Chat", "The chat history.")
#@knext.output_table("Reply", "The agents reply.")
class ChatBotAgent:

    query_column = knext.ColumnParameter(
        "Queries",
        "Column containing the queries",
        port_index=2
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm: LLMPortObjectSpec,
        vectorstore: VectorStorePortObjectSpec,
        table_spec: knext.Schema,
    ):
        return knext.Schema.from_columns([
                knext.Column(knext.string(), "Queries"),
                knext.Column(knext.string(), "Answers"),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm_port: LLMPortObject,
        vectorstore: VectorStorePortObject,
        input_table: knext.Table
    ):
        from langchain.chains import RetrievalQA
        from langchain.vectorstores import FAISS
        
        os.environ["SERPAPI_API_KEY"] = "a0ced4f64a04953c0d922fec371071c3e5f1d344325e27b219be836792baf8a1"
        
        llm = llm_port.create_model(ctx)
        db = vectorstore.load_store(ctx)

        node_descriptions = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

        tools = load_tools(["serpapi", "llm-math"], llm=llm)
        tools.append(
            Tool(
                name = "Node Descriptions QA System",
                func=node_descriptions.run,
                description="useful for when you need to answer questions about how to configure a node in KNIME. Input should be a fully formed question."
                )
        )

        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        queries = input_table.to_pandas()
        df = pd.DataFrame(queries)

        answers = []

        for query in df[self.query_column]:
            answers.append(agent.run(query))
        
        result_table = pd.DataFrame()
        result_table["Queries"] = queries
        result_table["Answers"] = answers

        return knext.Table.from_pandas(result_table)
