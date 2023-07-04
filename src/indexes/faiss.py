# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods


import knime.extension as knext
import pandas as pd

from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
)

from .base import (
    VectorStorePortObject,
    VectorStorePortObjectSpec,
    store_category,
)

from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

faiss_icon = "icons/ml.png"
faiss_category = knext.category(
    path=store_category,
    level_id="faiss",
    name="FAISS",
    description="",
    icon=faiss_icon,
)


class FAISSVectorstorePortObjectSpec(VectorStorePortObjectSpec):
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


class FAISSVectorstorePortObject(VectorStorePortObject):
    def __init__(
        self, spec: knext.PortObjectSpec, embeddings_port_object: EmbeddingsPortObject
    ) -> None:
        super().__init__(spec, embeddings_port_object)
        self._embeddings_port_object = embeddings_port_object

    def load_store(self, ctx):
        return FAISS.load_local(
            self.spec.persist_directory,
            self._embeddings_port_object.create_model(ctx),
        )


faiss_vector_store_port_type = knext.port_type(
    "FAISS Vector Store", FAISSVectorstorePortObject, FAISSVectorstorePortObjectSpec
)


@knext.node(
    "FAISS Vector Store Creator",
    knext.NodeType.SOURCE,
    faiss_icon,
    category=faiss_category,
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
    "FAISS Vector Store",
    "The created FAISS vector store.",
    faiss_vector_store_port_type,
)
class FAISSVectorStoreCreator:
    """
    Creates a FAISS Vector Store

    A vector store refers to a data structure or storage mechanism that holds
    a collection of vectors. These vectors represent the embeddings or numerical
    representations of objects such as documents, images, or other data points.
    he vector store allows efficient storage, retrieval,
    and similarity search operations on these vectors.FAISS provides indexing methods
    and algorithms optimized for similarity search on large-scale vector collections.

    """

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
    ) -> FAISSVectorstorePortObjectSpec:
        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings: EmbeddingsPortObject,
        input_table: knext.Table,
    ) -> FAISSVectorstorePortObject:
        df = input_table.to_pandas()

        documents = [Document(page_content=text) for text in df[self.document_column]]

        db = FAISS.from_documents(
            documents=documents,
            embedding=embeddings.create_model(ctx),
        )
        db.save_local(self.persist_directory)

        return FAISSVectorstorePortObject(self.create_spec(), embeddings)

    def create_spec(self):
        return FAISSVectorstorePortObjectSpec(self.persist_directory)


@knext.node(
    "FAISS Vector Store Loader",
    knext.NodeType.SOURCE,
    faiss_icon,
    category=faiss_category,
)
@knext.input_port(
    "Embeddings",
    "The embeddings model to use for the vector store.",
    embeddings_model_port_type,
)
@knext.output_port(
    "FAISS Vector Store", "The loaded vector store.", faiss_vector_store_port_type
)
class FAISSVectorStoreLoader:
    """
    Loads a FAISS Vector Store

    Loads the .fiass and .pkl file from a already
    created vectore store into KNIME.

    A vector store refers to a data structure or storage mechanism that holds
    a collection of vectors. These vectors represent the embeddings or numerical
    representations of objects such as documents, images, or other data points.
    he vector store allows efficient storage, retrieval,
    and similarity search operations on these vectors.FAISS provides indexing methods
    and algorithms optimized for similarity search on large-scale vector collections.
    """

    persist_directory = knext.StringParameter(
        "Vectorstore directory",
        """Directory to store the vectordb.""",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
    ) -> FAISSVectorstorePortObjectSpec:
        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> FAISSVectorstorePortObject:
        # TODO: Add check if .fiass and .pkl files are in the directory instead of instatiating as check
        FAISS.load_local(
            self.persist_directory, embeddings_port_object.create_model(ctx)
        )

        return FAISSVectorstorePortObject(self.create_spec(), embeddings_port_object)

    def create_spec(self):
        return FAISSVectorstorePortObjectSpec(self.persist_directory)
