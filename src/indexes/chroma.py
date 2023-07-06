import knime.extension as knext
from typing import Optional

from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
)

from .base import (
    FilestoreVectorstorePortObjectSpec,
    FilestoreVectorstorePortObject,
    store_category,
    validate_creator_document_column,
)

import util

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


class ChromaVectorstorePortObjectSpec(FilestoreVectorstorePortObjectSpec):
    # placeholder to enable us to add Chroma specific stuff later on
    pass


# TODO consider abstracting the chroma vectorstore because it is also possible to connect to a cloud hosted chroma instance
class ChromaVectorstorePortObject(FilestoreVectorstorePortObject):
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
            existing_entries = existing_collection.get()
            new_collection.add(**existing_entries)
            vectorstore = Chroma(
                embedding_function=vectorstore._embedding_function,
                persist_directory=vectorstore_folder,
                client_settings=settings,
                collection_metadata=existing_collection.metadata,
                client=client,
            )
        vectorstore.persist()

    def load_vectorstore(self, embeddings, vectorstore_path) -> Chroma:
        return Chroma(embedding_function=embeddings, persist_directory=vectorstore_path)


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
    """
    Creates a Chroma vector store from a string column and an embeddings model.

    The node generates a Chroma vector store by processing a string column containing documents
    with the provided embeddings model. For each document, the embeddings model extracts a numerical vector that represents
    the semantic meaning of the document. These embeddings are then stored in the vector store, along with their corresponding
    documents. Downstream nodes, such as the **Vector Store Retriever node**, utilize the vector store to find documents with similar
    semantic meaning when given a query.
    """

    document_column = knext.ColumnParameter(
        "Document column",
        """Select the column containing the documents to be embedded.""",
        port_index=1,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
        input_table: knext.Schema,
    ) -> ChromaVectorstorePortObjectSpec:
        if self.document_column is None:
            self.document_column = util.pick_default_column(input_table, knext.string())
        else:
            validate_creator_document_column(input_table, self.document_column)
        return ChromaVectorstorePortObjectSpec(embeddings_spec)

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
        )

        return ChromaVectorstorePortObject(
            ChromaVectorstorePortObjectSpec(embeddings.spec), embeddings, vectorstore=db
        )


@knext.node(
    "Chroma Vector Store Reader",
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
class ChromaVectorStoreReader:
    """
    Reads a Chroma vector store from a local path.

    This node allows you to read a Chroma vector store from a local path. It combines the Chroma vector store with
    an input embeddings model that is used in downstream nodes, such as the **Vector Store Retriever node**. The embeddings
    model is responsible for embedding documents, enabling the vector store to retrieve documents that share similar embeddings,
    facilitating tasks like document clustering or recommendation systems.

    If the vector store was created with another tool i.e. outside of KNIME, the embeddings model is not stored with the vectorstore, so it has to be provided separately (<Provider> Embeddings Connector Node).
    """

    persist_directory = knext.StringParameter(
        "Vectorstore directory",
        """Directory to store the vectordb.""",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
    ) -> ChromaVectorstorePortObjectSpec:
        return ChromaVectorstorePortObjectSpec(embeddings_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> ChromaVectorstorePortObject:
        # TODO: Add check if Chroma files are here instead of instantiation
        chroma = Chroma(
            self.persist_directory, embeddings_port_object.create_model(ctx)
        )
        return ChromaVectorstorePortObject(
            ChromaVectorstorePortObjectSpec(embeddings_port_object.spec),
            embeddings_port_object,
            vectorstore=chroma,
        )
