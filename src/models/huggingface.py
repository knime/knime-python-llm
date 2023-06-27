#TODO: Implement me

import knime.extension as knext
from .base import (
    ModelPortObjectSpecContent,
    ModelPortObjectContent,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
)


class HuggingFaceLLMPortObjectSpecContent(ModelPortObjectSpecContent):
    def __init__(self, credentials, model_name) -> None:
        super().__init__()
        self._credentials = credentials
        self._model_name = model_name

    def serialize(self) -> dict:
        return {"credentials": self._credentials, "model": self._model_name}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"], data["model"])


LLMPortObjectSpec.register_content_type(HuggingFaceLLMPortObjectSpecContent)


class HuggingFaceLLMPortObjectContent(ModelPortObjectContent):
    def create_model(self, ctx):
        return HuggingFaceTextGenInference(
            inference_server_url="http://localhost:8010/",
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
        )
    

LLMPortObject.register_content_type(HuggingFaceLLMPortObjectContent)

huggingface_icon = ""
huggingface_category = knext.category(
    path="/community",
    level_id="llm",
    name="Model Loaders",
    description="All nodes related to models",
    icon="",
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
