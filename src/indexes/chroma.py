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
    pick_default_column,
)

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
        if vectorstore._persist_directory is None or not vectorstore._persist_directory == vectorstore_folder:
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

    The Chroma Vector Store Creator creates a Chroma vector store from a string column containing documents and an embeddings model.
    For each document an embedding i.e. a numerical vector representing the document is extracted by the embeddings model and the embedding is
    stored together with the document in the vector store. Down-stream nodes such as the Vector Store Retriever use the vector store to find documents
    with similar semantic meaning given a query.
    """

    document_column = knext.ColumnParameter(
        "Document column",
        """Selection of column used as the document column.""",
        port_index=1,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
        input_table: knext.Schema,
    ) -> ChromaVectorstorePortObjectSpec:
        if self.document_column is None:
            self.document_column = pick_default_column(input_table, knext.string())
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

    This node allows to read a Chroma vector store from a local path and combines it with an embeddings model that
    is used in down-stream nodes (e.g. the Vector Store Retriever) to embed documents such that the vector store
    can find documents with similar embeddings.
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
