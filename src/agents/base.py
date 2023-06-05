import knime.extension as knext
from models.base import LLMPortObjectSpec, LLMPortObject, llm_port_type
from indexes.base import (
    VectorStorePortObjectSpec,
    VectorStorePortObject,
    vectorstore_port_type,
)

langchain_icon = "./icons/langchain.png"
agent_category = ""


@knext.node(
    "Chat Bot Agent", knext.NodeType.PREDICTOR, langchain_icon, category=agent_category
)
@knext.input_port("LLM", "The large language model to chat with.", llm_port_type)
@knext.input_port(
    "Vectorstore",
    "The vectorstore to get context for the user questions.",
    vectorstore_port_type,
)
@knext.input_table("Chat", "The chat history.")
@knext.output_table("Reply", "The agents reply.")
class ChatBotAgent:
    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm: LLMPortObjectSpec,
        vectorstore: VectorStorePortObjectSpec,
        table: knext.Schema,
    ) -> knext.Schema:
        return table

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm: LLMPortObject,
        vectorstore: VectorStorePortObject,
        table: knext.Table,
    ) -> knext.Table:
        return table
