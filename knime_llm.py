import knime.extension as knext

# TODO split up into multiple files e.g. core, open_ai and so on


class EmbeddingsPortObjectSpec(knext.PortObjectSpec):
    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: dict) -> "EmbeddingsPortObjectSpec":
        return cls()


class EmbeddingsPortObject(knext.PortObject):
    def __init__(self, spec: knext.PortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(
        cls, spec: EmbeddingsPortObjectSpec, data: bytes
    ) -> "EmbeddingsPortObject":
        return cls(spec)


embeddings_port_type = knext.port_type(
    "Embeddings", EmbeddingsPortObject, EmbeddingsPortObjectSpec
)


class LLMPortObjectSpec(knext.PortObjectSpec):
    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: dict) -> "LLMPortObjectSpec":
        return cls()


class LLMPortObject(knext.PortObject):
    def __init__(self, spec: knext.PortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: LLMPortObjectSpec, data: bytes) -> "LLMPortObject":
        return cls(spec)


llm_port_type = knext.port_type("LLM", LLMPortObject, LLMPortObjectSpec)


class VectorStorePortObjectSpec(knext.PortObjectSpec):
    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: dict) -> "VectorStorePortObjectSpec":
        return cls()


class VectorStorePortObject(knext.PortObject):
    def __init__(self, spec: VectorStorePortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(
        cls, spec: VectorStorePortObjectSpec, data: bytes
    ) -> "VectorStorePortObject":
        return cls(spec)


vectorstore_port_type = knext.port_type(
    "Vectorstore", VectorStorePortObject, VectorStorePortObjectSpec
)

openai_category = ""
openai_icon = "openai.png"

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
@knext.output_port(
    "OpenAI LLM", "A large language model from OpenAI.", llm_port_type
)
class OpenAILLMLoader:
    def configure(self, ctx: knext.ConfigurationContext) -> LLMPortObjectSpec:
        return LLMPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext) -> LLMPortObject:
        return LLMPortObject(LLMPortObjectSpec())
    

huggingface_icon = "huggingface.png"
huggingface_category = ""

@knext.node(
    "Hugging Face Embeddings Loader",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_category,
)
@knext.output_port(
    "Hugging Face Embeddings", "An embeddings model from Hugging Face.", embeddings_port_type
)
class HuggingFaceEmbeddingsLoader:
    def configure(self, ctx: knext.ConfigurationContext) -> EmbeddingsPortObjectSpec:
        return EmbeddingsPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext) -> EmbeddingsPortObject:
        return EmbeddingsPortObject(EmbeddingsPortObjectSpec())


@knext.node(
    "Hugging Face LLM Loader", knext.NodeType.SOURCE, huggingface_icon, category=huggingface_category
)
@knext.output_port(
    "Hugging Face LLM", "A large language model from Hugging Face.", llm_port_type
)
class HuggingFaceLLMLoader:
    def configure(self, ctx: knext.ConfigurationContext) -> LLMPortObjectSpec:
        return LLMPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext) -> LLMPortObject:
        return LLMPortObject(LLMPortObjectSpec())


vectorstore_category = ""

langchain_icon = "langchain.png"


@knext.node(
    "Vectorstore Loader",
    knext.NodeType.SOURCE,
    "chroma.png",
    category=vectorstore_category,
)
@knext.input_port(
    "Embeddings", "The embeddings to use in the vectorstore.", embeddings_port_type
)
@knext.output_port("Vectorstore", "The loaded vectorstore.", vectorstore_port_type)
class VectorStoreLoader:
    def configure(
            self,
        ctx: knext.ConfigurationContext, embeddings: EmbeddingsPortObjectSpec
    ) -> VectorStorePortObjectSpec:
        return VectorStorePortObjectSpec()

    def execute(self,
        ctx: knext.ExecutionContext, embeddings: EmbeddingsPortObject
    ) -> VectorStorePortObject:
        return VectorStorePortObject(VectorStorePortObjectSpec)


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
    def configure(self,
        ctx: knext.ConfigurationContext,
        llm: LLMPortObjectSpec,
        vectorstore: VectorStorePortObjectSpec,
        table: knext.Schema,
    ) -> knext.Schema:
        return table

    def execute(self,
        ctx: knext.ExecutionContext,
        llm: LLMPortObject,
        vectorstore: VectorStorePortObject,
        table: knext.Table,
    ) -> knext.Table:
        return table
