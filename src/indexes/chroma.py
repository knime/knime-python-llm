import knime.extension as knext
from typing import Optional

from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
)

from .base import (
    VectorstorePortObjectSpec,
    VectorstorePortObject,
    FilestoreVectorstorePortObjectSpec,
    FilestoreVectorstorePortObject,
    MetadataSettings,
    store_category,
    get_metadata_columns,
    handle_missing_metadata_values,
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


class ChromaVectorstorePortObjectSpec(VectorstorePortObjectSpec):
    """Super type of the spec of local and remote Chroma vector stores."""


class LocalChromaVectorstorePortObjectSpec(
    ChromaVectorstorePortObjectSpec, FilestoreVectorstorePortObjectSpec
):
    """Spec for local chroma instances run within this process."""


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
        column_filter=util.create_type_filer(knext.string()),
    )

    missing_value_handling = knext.EnumParameter(
        "Handle missing values in the document column",
        """Define whether missing values in the document column should be skipped or whether the 
        node execution should fail on missing values.""",
        default_value=lambda v: util.MissingValueHandlingOptions.Fail.name
        if v < knext.Version(5, 2, 0)
        else util.MissingValueHandlingOptions.SkipRow.name,
        enum=util.MissingValueHandlingOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.2.0",
    )

    metadata_settings = MetadataSettings(since_version="5.2.0")

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
        input_table: knext.Schema,
    ) -> LocalChromaVectorstorePortObjectSpec:
        embeddings_spec.validate_context(ctx)
        if self.document_column:
            validate_creator_document_column(input_table, self.document_column)
        else:
            self.document_column = util.pick_default_column(input_table, knext.string())

        metadata_cols = get_metadata_columns(
            self.metadata_settings.metadata_columns, self.document_column, input_table
        )

        return LocalChromaVectorstorePortObjectSpec(
            embeddings_spec=embeddings_spec,
            metadata_column_names=metadata_cols,
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings: EmbeddingsPortObject,
        input_table: knext.Table,
    ) -> ChromaVectorstorePortObject:
        meta_data_columns = get_metadata_columns(
            self.metadata_settings.metadata_columns,
            self.document_column,
            input_table.schema,
        )

        document_column = self.document_column

        df = input_table[[document_column] + meta_data_columns].to_pandas()

        def to_document(row) -> Document:
            metadata = {name: row[name] for name in meta_data_columns}
            return Document(page_content=row[document_column], metadata=metadata)

        # Skip rows with missing values if "SkipRow" option is selected
        # or fail execution if "Fail" is selected and there are missing documents
        missing_value_handling_setting = util.MissingValueHandlingOptions[
            self.missing_value_handling
        ]

        df = util.handle_missing_and_empty_values(
            df, self.document_column, missing_value_handling_setting, ctx
        )

        # Replaces missing values with empty string since the allowed metadata value
        # types are string, integer, float or bool
        df = handle_missing_metadata_values(df, meta_data_columns)

        documents = df.apply(to_document, axis=1).tolist()

        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings.create_model(ctx),
        )

        return LocalChromaVectorstorePortObject(
            LocalChromaVectorstorePortObjectSpec(embeddings.spec, meta_data_columns),
            embeddings,
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

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
    ) -> LocalChromaVectorstorePortObjectSpec:
        embeddings_spec.validate_context(ctx)
        if not self.persist_directory:
            raise knext.InvalidParametersError("No vector store directory specified.")
        return LocalChromaVectorstorePortObjectSpec(embeddings_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> ChromaVectorstorePortObject:
        chroma = Chroma(
            self.persist_directory, embeddings_port_object.create_model(ctx)
        )

        document_list = chroma.similarity_search("a", k=1)
        metadata_keys = (
            [key for key in document_list[0].metadata] if len(document_list) > 0 else []
        )

        return LocalChromaVectorstorePortObject(
            LocalChromaVectorstorePortObjectSpec(
                embeddings_port_object.spec, metadata_keys
            ),
            embeddings_port_object,
            vectorstore=chroma,
        )
