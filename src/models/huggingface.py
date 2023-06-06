import knime.extension as knext
from .base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
)

huggingface_icon = "./icons/huggingface.png"
huggingface_category = knext.category(
    path="/community",
    level_id="top",
    name="Model Loaders",
    description="All nodes related to models",
    icon="./icons/ml.svg",
)


@knext.node(
    "Hugging Face Embeddings Loader",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_category,
)
@knext.output_port(
    "Hugging Face Embeddings",
    "An embeddings model from Hugging Face.",
    embeddings_port_type,
)
class HuggingFaceEmbeddingsLoader:
    def configure(self, ctx: knext.ConfigurationContext) -> EmbeddingsPortObjectSpec:
        return EmbeddingsPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext) -> EmbeddingsPortObject:
        return EmbeddingsPortObject(EmbeddingsPortObjectSpec())


@knext.node(
    "Hugging Face LLM Loader",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_category,
)
@knext.output_port(
    "Hugging Face LLM", "A large language model from Hugging Face.", llm_port_type
)
class HuggingFaceLLMLoader:
    def configure(self, ctx: knext.ConfigurationContext) -> LLMPortObjectSpec:
        return LLMPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext) -> LLMPortObject:
        return LLMPortObject(LLMPortObjectSpec())
