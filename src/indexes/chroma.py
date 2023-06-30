# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods


import knime.extension as knext
import pandas as pd

from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
)

from .base import (
    VectorStorePortObjectSpec,
    VectorStorePortObject,
    store_category,
)

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

chroma_icon = "icons/chroma.png"
chroma_category = knext.category(
    path=store_category,
    level_id="chroma",
    name="Chroma",
    description="",
    icon=chroma_icon,
)


class ChromaVectorstorePortObjectSpec(VectorStorePortObjectSpec):
    def __init__(self, persist_directory) -> None:
        super().__init__()
        self._persist_directory = persist_directory

    @property
    def persist_directory(self):
        return self._persist_directory

    def serialize(self) -> dict:
        return {
            "persist_directory": self._persist_directory,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["persist_directory"])


class ChromaVectorstorePortObject(VectorStorePortObject):
    def __init__(
        self,
        spec: ChromaVectorstorePortObjectSpec,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> None:
        super().__init__(spec, embeddings_port_object)
        self._embeddings_port_object = embeddings_port_object

    def load_store(self, ctx):
        return Chroma(
            persist_directory=self.spec.persist_directory,
        )


chroma_vector_store_port_type = knext.port_type(
    "Chroma Vector Store", ChromaVectorstorePortObject, ChromaVectorstorePortObjectSpec
)


@knext.node(
    "Chroma Vector Store Creator",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
)
@knext.input_port(
    "Embeddings",
    "The embeddings model to use for the vector store.",
    embeddings_model_port_type,
)
@knext.input_table(
    name="Documents",
    description="""Table containing a string column representing documents that will be used in the vector store.""",
)
@knext.output_port(
    "Chroma Vector Store",
    "The created Chroma vector store.",
    chroma_vector_store_port_type,
)
class ChromaVectorStoreCreator:
    document_column = knext.ColumnParameter(
        "Document column",
        """Selection of column used as the document column.""",
        port_index=1,
    )

    persist_directory = knext.StringParameter(
        "Persist directory",
        """Directory in whcih the vector store will be written to.""",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
        input_table: knext.Schema,
    ) -> ChromaVectorstorePortObjectSpec:
        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings: EmbeddingsPortObject,
        input_table: knext.Table,
    ) -> ChromaVectorstorePortObject:
        df = input_table.to_pandas()

        documents = [Document(page_content=text) for text in df[self.document_column]]
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings.create_model(ctx),
            persist_directory=self.persist_directory,
        )

        db.persist()

        return ChromaVectorstorePortObject(self.create_spec(), embeddings)

    def create_spec(self):
        return ChromaVectorstorePortObjectSpec(self.persist_directory)


@knext.node(
    "Chroma Vector Store Loader",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
)
@knext.input_port(
    "Embeddings",
    "The embeddings model to use for the vector store.",
    embeddings_model_port_type,
)
@knext.output_port(
    "Chroma Vector Store", "The loaded vector store.", chroma_vector_store_port_type
)
class ChromaVectorStoreLoader:
    persist_directory = knext.StringParameter(
        "Vectorstore directory",
        """Directory to store the vectordb.""",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
    ) -> ChromaVectorstorePortObjectSpec:
        return self.create_spec_content()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> ChromaVectorstorePortObject:
        # TODO: Add check if Chroma files are here instead of instantiation

        Chroma(self.persist_directory, embeddings_port_object.create_model(ctx))

        return ChromaVectorstorePortObject(self.create_spec(), embeddings_port_object)

    def create_spec(self):
        return ChromaVectorstorePortObjectSpec(self.persist_directory)
