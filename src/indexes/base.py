import knime.extension as knext
import pickle

from models.base import (
    EmbeddingsPortObject,
    ModelPortObjectSpecContent,
)


class VectorStorePortObjectSpecContent(knext.PortObjectSpec):
    pass


class VectorStorePortObjectSpec(knext.PortObjectSpec):
    content_registry = {}

    def __init__(self, content: ModelPortObjectSpecContent) -> None:
        super().__init__()
        self._content = content

    def serialize(self):
        return {"type": str(type(self._content)), "content": self._content.serialize()}

    @classmethod
    def register_content_type(cls, content_type: type):
        cls.content_registry[str(content_type)] = content_type

    @classmethod
    def deserialize(cls, data: dict) -> "VectorStorePortObjectSpec":
        content_cls = cls.content_registry[data["type"]]
        return content_cls.deserialize(data["content"])


class VectorStorePortObjectContent(knext.PortObject):
    def __init__(
        self, spec: PortObjectSpec, embeddings_model: EmbeddingsPortObject
    ) -> None:
        super().__init__(spec)
        self._embeddings_model = embeddings_model

    def create_store(self, ctx):
        raise NotImplementedError()

    def load_store(self, ctx):
        raise NotImplementedError()

    def serialize(self):
        config = {"embeddings_model": self._embeddings_model}
        return pickle.dumps(config)

    @classmethod
    def deserialize(
        cls, spec: VectorStorePortObjectSpec, data
    ) -> "VectorStorePortObjectContent":
        config = pickle.loads(data)
        return cls(spec, config["embeddings_model"])


class VectorStorePortObject(knext.PortObject):
    content_registry = {}

    def __init__(
        self, spec: knext.PortObjectSpec, content: VectorStorePortObjectContent
    ) -> None:
        super().__init__(spec)
        self._content = content

    def create_store(self, ctx):
        return self._content.create_store(self, ctx)

    def load_store(self, ctx):
        return self._content.load_store(ctx)

    def serialize(self):
        config = {
            "type": str(type(self._content)),
            "content": self._content.serialize(),
        }
        return pickle.dumps(config)

    @classmethod
    def register_content_type(cls, content_type: type):
        cls.content_registry[str(content_type)] = content_type

    @classmethod
    def deserialize(
        cls, spec: VectorStorePortObjectSpec, data
    ) -> "VectorStorePortObject":
        config = pickle.loads(data)
        content_cls = cls.content_registry[config["type"]]
        return cls(spec, content_cls.deserialize(spec, config["content"]))


vector_store_port_type = knext.port_type(
    "Vectorstore", VectorStorePortObject, VectorStorePortObjectSpec
)