import knime.extension as knext
from .base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
)

openai_icon = "./icons/openai.png"
openai_category = knext.category(
    path="/community",
    level_id="top",
    name="Model Loaders",
    description="All nodes related to models",
    icon="./icons/ml.svg"
)


@knext.node(
    "OpenAI Embeddings Loader",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.output_port(
    "OpenAI Embeddings", "An embeddings model from OpenAI.", embeddings_port_type
)
class OpenAIEmbeddingsLoader:
    def configure(self, ctx: knext.ConfigurationContext) -> EmbeddingsPortObjectSpec:
        return EmbeddingsPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext) -> EmbeddingsPortObject:
        return EmbeddingsPortObject(EmbeddingsPortObjectSpec())


@knext.node(
    "OpenAI LLM Loader", knext.NodeType.SOURCE, openai_icon, category=openai_category
)
@knext.output_port("OpenAI LLM", "A large language model from OpenAI.", llm_port_type)
class OpenAILLMLoader:
    def configure(self, ctx: knext.ConfigurationContext) -> LLMPortObjectSpec:
        return LLMPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext) -> LLMPortObject:
        return LLMPortObject(LLMPortObjectSpec())
