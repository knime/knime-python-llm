import uuid
import pandas as pd
import knime.extension as knext
from typing import Optional

from knime.extension import ExecutionContext
from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
)

from .base import (
    BaseVectorStoreCreator,
    VectorstorePortObjectSpec,
    VectorstorePortObject,
    FilestoreVectorstorePortObjectSpec,
    FilestoreVectorstorePortObject,
    store_category,
)

from langchain_core.embeddings import Embeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document


chroma_icon = "icons/chroma.png"
chroma_category = knext.category(
    path=store_category,
    level_id="chroma",
    name="Chroma",
    description="Contains nodes for working with Chroma vector stores.",
    icon=chroma_icon,
)

# Keeps the Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME value consistent for backwards compatibility
default_collection_name = "langchain"


class ChromaVectorstorePortObjectSpec(VectorstorePortObjectSpec):
    """Super type of the spec of local and remote Chroma vector stores."""


class LocalChromaVectorstorePortObjectSpec(
    ChromaVectorstorePortObjectSpec, FilestoreVectorstorePortObjectSpec
):
    """Spec for local chroma instances run within this process."""

    def __init__(
        self,
        embeddings_spec: EmbeddingsPortObjectSpec,
        metadata_column_names: Optional[list[str]] = None,
        collection_name: str = default_collection_name,
    ) -> None:
        super().__init__(embeddings_spec, metadata_column_names)
        self._collection_name = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def serialize(self) -> dict:
        data = super().serialize()
        data["collection_name"] = self.collection_name
        return data

    @classmethod
    def deserialize(cls, data: dict, java_callback):
        super_cls = super().deserialize(data=data, java_callback=java_callback)
        return cls(
            super_cls.embeddings_spec,
            super_cls.metadata_column_names,
            data.get("collection_name", default_collection_name),
        )


class ChromaVectorstorePortObject(VectorstorePortObject):
    """Super type of Chroma vector stores"""


chroma_vector_store_port_type = knext.port_type(
    "Chroma Vector Store", ChromaVectorstorePortObject, ChromaVectorstorePortObjectSpec
)


class LocalChromaVectorstorePortObject(
    ChromaVectorstorePortObject, FilestoreVectorstorePortObject
):
    def __init__(
        self,
        spec: ChromaVectorstorePortObjectSpec,
        embeddings_port_object: EmbeddingsPortObject,
        folder_path: Optional[str] = None,
        vectorstore: Optional[Chroma] = None,
    ) -> None:
        super().__init__(spec, embeddings_port_object, folder_path, vectorstore)

    def save_vectorstore(self, vectorstore_folder: str, vectorstore: Chroma):
        if (
            vectorstore._persist_directory is None
            or not vectorstore._persist_directory == vectorstore_folder
        ):
            # HACK because Chroma doesn't allow to add or change a persist directory after the fact
            import chromadb
            import chromadb.config

            settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=vectorstore_folder
            )
            existing_collection = vectorstore._collection
            client = chromadb.Client(settings)
            new_collection = client.get_or_create_collection(
                name=existing_collection.name,
                metadata=existing_collection.metadata,
                embedding_function=existing_collection._embedding_function,
            )
            existing_entries = existing_collection.get(
                include=["embeddings", "documents", "metadatas"]
            )

            # replace None (not allowed) metadata values with empty dictionaries (expected value type)
            for i, entry in enumerate(existing_entries["metadatas"]):
                if entry is None:
                    existing_entries["metadatas"][i] = {}

            try:
                new_collection.add(**existing_entries)
            except IndexError:
                raise knext.InvalidParametersError(
                    f"No Chroma collection named '{existing_collection.name}' was found in the specified directory."
                )
            vectorstore = Chroma(
                embedding_function=vectorstore._embedding_function,
                persist_directory=vectorstore_folder,
                client_settings=settings,
                collection_metadata=existing_collection.metadata,
                client=client,
            )
        vectorstore.persist()

    def load_vectorstore(self, embeddings, vectorstore_path) -> Chroma:
        return Chroma(
            collection_name=self.spec.collection_name,
            embedding_function=embeddings,
            persist_directory=vectorstore_path,
        )


local_chroma_vector_store_port_type = knext.port_type(
    "Chroma Vector Store",
    LocalChromaVectorstorePortObject,
    LocalChromaVectorstorePortObjectSpec,
)


@knext.node(
    "Chroma Vector Store Creator",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
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
    local_chroma_vector_store_port_type,
)
class ChromaVectorStoreCreator(BaseVectorStoreCreator):
    """
    Creates a Chroma vector store from a string column and an embeddings model.

    The node generates a Chroma vector store by processing a string column containing documents
    with the provided embeddings model. For each document, the embeddings model extracts a numerical vector that represents
    the semantic meaning of the document. These embeddings are then stored in the vector store, along with their corresponding
    documents. Downstream nodes, such as the **Vector Store Retriever node**, utilize the vector store to find documents with similar
    semantic meaning when given a query.
    """

    # redefined here to enforce the parameter order in the dialog
    document_column = BaseVectorStoreCreator.document_column

    embeddings_column = BaseVectorStoreCreator.embeddings_column

    collection_name = knext.StringParameter(
        "Collection name",
        "Specify the collection name of the vector store.",
        default_collection_name,
        since_version="5.3.2",
    )

    def _configure(
        self,
        embeddings_spec: EmbeddingsPortObjectSpec,
        metadata_column_names: list[str],
    ) -> LocalChromaVectorstorePortObjectSpec:
        if self.collection_name == "":
            raise knext.InvalidParametersError("The collection name must not be empty.")
        return LocalChromaVectorstorePortObjectSpec(
            embeddings_spec=embeddings_spec,
            metadata_column_names=metadata_column_names,
            collection_name=self.collection_name,
        )

    def _create_port_object(
        self,
        ctx: ExecutionContext,
        embeddings_obj: EmbeddingsPortObject,
        documents: list[Document],
        metadata_column_names: list[str],
        embeddings: pd.Series | None,
    ) -> LocalChromaVectorstorePortObject:
        embeddings_model = embeddings_obj.create_model(ctx)
        if embeddings is None:
            db = Chroma.from_documents(
                documents=documents,
                embedding=embeddings_model,
                collection_name=self.collection_name,
            )
        else:
            db = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings_model,
            )

            db._collection.add(
                documents=[doc.page_content for doc in documents],
                embeddings=[arr.tolist() for arr in embeddings],
                ids=[str(uuid.uuid4()) for _ in range(len(documents))],
                metadatas=[doc.metadata for doc in documents],
            )

        return LocalChromaVectorstorePortObject(
            LocalChromaVectorstorePortObjectSpec(
                embeddings_spec=embeddings_obj.spec,
                metadata_column_names=metadata_column_names,
                collection_name=self.collection_name,
            ),
            embeddings_port_object=embeddings_obj,
            vectorstore=db,
        )


@knext.node(
    "Chroma Vector Store Reader",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
)
@knext.input_port(
    "Embeddings",
    "The embeddings model to use for the vector store.",
    embeddings_model_port_type,
)
@knext.output_port(
    "Chroma Vector Store",
    "The loaded vector store.",
    local_chroma_vector_store_port_type,
)
class ChromaVectorStoreReader:
    """
    Reads a Chroma vector store created with LangChain from a local path.

    This node allows you to read a Chroma vector store created with [LangChain](https://python.langchain.com/docs/integrations/vectorstores/chroma) from a local path. If you want to create a new vector store, use the Chroma Vector Store Creator instead.

    A vector store is a data structure or storage mechanism that stores a collection of numerical vectors
    along with their corresponding documents. The vector store enables efficient storage, retrieval, and similarity
    search operations on these vectors and their associated data.

    If the vector store was created with LangChain in Python, the embeddings model is not stored with the vectorstore, so it has to be provided separately via the matching Embeddings Model Connector node.

    On execution, the node will extract a document from the store to obtain information about the document's metadata. This assumes that each document in the vector store has the same kind of metadata attached to it.
    """

    persist_directory = knext.StringParameter(
        "Vector store directory",
        """The local directory in which the vector store is stored.""",
    )

    collection_name = knext.StringParameter(
        "Collection name",
        "Specify the collection name of the vector store.",
        default_collection_name,
        since_version="5.3.2",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
    ) -> LocalChromaVectorstorePortObjectSpec:
        embeddings_spec.validate_context(ctx)
        if not self.persist_directory:
            raise knext.InvalidParametersError("No vector store directory specified.")

        if self.collection_name == "":
            raise knext.InvalidParametersError("The collection name must not be empty.")

        return LocalChromaVectorstorePortObjectSpec(
            embeddings_spec, collection_name=self.collection_name
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> ChromaVectorstorePortObject:
        chroma = Chroma(
            self.collection_name,
            embeddings_port_object.create_model(ctx),
            self.persist_directory,
        )

        document_list = chroma.similarity_search("a", k=1)
        metadata_keys = (
            [key for key in document_list[0].metadata] if len(document_list) > 0 else []
        )

        return LocalChromaVectorstorePortObject(
            LocalChromaVectorstorePortObjectSpec(
                embeddings_port_object.spec, metadata_keys, self.collection_name
            ),
            embeddings_port_object,
            vectorstore=chroma,
        )
