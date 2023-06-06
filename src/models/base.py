import knime.extension as knext


class EmbeddingsPortObjectSpec(knext.PortObjectSpec):
    def __init__(self, credentials: str, model_name: str) -> None:
        super().__init__()
        self._credentials = credentials
        self._model_name = model_name

    @property
    def cred(self):
        return self._credentials

    @property
    def model_name(self):
        return self._model_name

    def serialize(self) -> dict:
        return {"credentials": self._credentials, "model_name": self._model_name}

    @classmethod
    def deserialize(cls, data: dict) -> "EmbeddingsPortObjectSpec":
        return cls(data["credentials"], data["model_name"])


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
    def __init__(self, credentials: str, model_name: str) -> None:
        super().__init__()
        self._credentials = credentials
        self._model_name = model_name

    @property
    def cred(self):
        return self._credentials

    @property
    def model_name(self):
        return self._model_name

    def serialize(self) -> dict:
        return {"credentials": self._credentials, "model_name": self._model_name}

    @classmethod
    def deserialize(cls, data: dict) -> "LLMPortObjectSpec":
        return cls(data["credentials"], data["model_name"])


class LLMPortObject(knext.PortObject):
    def __init__(self, spec: knext.PortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: LLMPortObjectSpec, data: bytes) -> "LLMPortObject":
        return cls(spec)


llm_port_type = knext.port_type("LLM", LLMPortObject, LLMPortObjectSpec)
