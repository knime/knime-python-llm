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
    MetadataSettings,
    store_category,
    get_metadata_columns,
    validate_creator_document_column,
)

import util

from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

faiss_icon = "icons/ml.png"
faiss_category = knext.category(
    path=store_category,
    level_id="faiss",
    name="FAISS",
    description="Contains nodes for working with FAISS vector stores.",
    icon=faiss_icon,
)


class FAISSVectorstorePortObjectSpec(FilestoreVectorstorePortObjectSpec):
    # placeholder to enable use to add FAISS specific stuff later on
    pass


class FAISSVectorstorePortObject(FilestoreVectorstorePortObject):
    def __init__(
        self,
        spec: FAISSVectorstorePortObjectSpec,
        embeddings_port_object: EmbeddingsPortObject,
        folder_path: Optional[str] = None,
        vectorstore: Optional[FAISS] = None,
    ) -> None:
        super().__init__(
            spec, embeddings_port_object, folder_path, vectorstore=vectorstore
        )

    def load_vectorstore(self, embeddings, vectorstore_path) -> FAISS:
        return FAISS.load_local(
            embeddings=embeddings,
            folder_path=vectorstore_path,
            allow_dangerous_deserialization=True,
        )

    def save_vectorstore(self, vectorstore_folder, vectorstore: FAISS):
        vectorstore.save_local(vectorstore_folder)


faiss_vector_store_port_type = knext.port_type(
    "FAISS Vector Store", FAISSVectorstorePortObject, FAISSVectorstorePortObjectSpec
)


@knext.node(
    "FAISS Vector Store Creator (Labs)",
    knext.NodeType.SOURCE,
    faiss_icon,
    category=faiss_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
)
@knext.input_port(
    "Embeddings Model",
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
    Creates a FAISS vector store from a string column and an embeddings model.

    The node generates a FAISS vector store by processing a string column containing documents
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
    ) -> FAISSVectorstorePortObjectSpec:
        embeddings_spec.validate_context(ctx)

        if self.document_column:
            validate_creator_document_column(input_table, self.document_column)
        else:
            self.document_column = util.pick_default_column(input_table, knext.string())

        metadata_cols = get_metadata_columns(
            self.metadata_settings.metadata_columns, self.document_column, input_table
        )
        return FAISSVectorstorePortObjectSpec(
            embeddings_spec=embeddings_spec,
            metadata_column_names=metadata_cols,
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings: EmbeddingsPortObject,
        input_table: knext.Table,
    ) -> FAISSVectorstorePortObject:
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

        documents = df.apply(to_document, axis=1).tolist()

        db = FAISS.from_documents(
            documents=documents,
            embedding=embeddings.create_model(ctx),
        )

        return FAISSVectorstorePortObject(
            FAISSVectorstorePortObjectSpec(embeddings.spec, meta_data_columns),
            embeddings,
            vectorstore=db,
        )


@knext.node(
    "FAISS Vector Store Reader (Labs)",
    knext.NodeType.SOURCE,
    faiss_icon,
    category=faiss_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
)
@knext.input_port(
    "Embeddings",
    "The embeddings model that the vector store uses for embedding documents.",
    embeddings_model_port_type,
)
@knext.output_port(
    "FAISS Vector Store", "The loaded FAISS vector store.", faiss_vector_store_port_type
)
class FAISSVectorStoreReader:
    """
    Reads a FAISS vector store created with LangChain from a local path.

    This node reads a FAISS vector store create with [LangChain](https://python.langchain.com/docs/integrations/vectorstores/faiss#saving-and-loading) from a local path.
    If you want to create a new vector store, use the FAISS Vector Store Creator instead.

    A vector store is a data structure or storage mechanism that stores a collection of numerical vectors
    along with their corresponding documents. The vector store enables efficient storage, retrieval, and similarity
    search operations on these vectors and their associated data.

    If the vector store was created with LangChain in Python, the embeddings model is not stored with the vectorstore, so it has to be provided separately via the matching Embeddings Model Connector node.

    On execution, the node will extract a document from the store to obtain information about the document's metadata. This assumes that each document in the vector store has the same kind of metadata attached to it.
    """

    persist_directory = knext.StringParameter(
        "Vector store directory",
        "The local directory in which the vector store is stored.",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
    ) -> FAISSVectorstorePortObjectSpec:
        embeddings_spec.validate_context(ctx)
        if not self.persist_directory:
            raise knext.InvalidParametersError("Select the vector store directory.")
        return FAISSVectorstorePortObjectSpec(embeddings_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> FAISSVectorstorePortObject:
        # TODO: Add check if .fiass and .pkl files are in the directory instead of instatiating as check
        db = FAISS.load_local(
            self.persist_directory, embeddings_port_object.create_model(ctx)
        )

        document_list = db.similarity_search("a", k=1)
        metadata_keys = (
            [key for key in document_list[0].metadata] if len(document_list) > 0 else []
        )

        return FAISSVectorstorePortObject(
            FAISSVectorstorePortObjectSpec(embeddings_port_object.spec, metadata_keys),
            embeddings_port_object,
            vectorstore=db,
        )
