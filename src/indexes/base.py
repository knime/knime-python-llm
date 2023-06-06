import knime.extension as knext
from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
)


class VectorStorePortObjectSpec(knext.PortObjectSpec):
    def __init__(self, persist_directory) -> None:
        super().__init__()
        self._persist_directory = persist_directory

    @property
    def persist_directory(self):
        return self._persist_directory

    def serialize(self) -> dict:
        return {"persist_directory": self._persist_directory}

    @classmethod
    def deserialize(cls, data: dict) -> "VectorStorePortObjectSpec":
        return cls(data["persist_directory"])


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
