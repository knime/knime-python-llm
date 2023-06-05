import knime.extension as knext


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
