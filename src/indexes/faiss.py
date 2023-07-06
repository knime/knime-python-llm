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

from langchain.vectorstores import FAISS
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
        return FAISS.load_local(embeddings=embeddings, folder_path=vectorstore_path)

    def save_vectorstore(self, vectorstore_folder, vectorstore: FAISS):
        vectorstore.save_local(vectorstore_folder)


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
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
        input_table: knext.Schema,
    ) -> FAISSVectorstorePortObjectSpec:
        if self.document_column is None:
            self.document_column = util.pick_default_column(input_table, knext.string())
        else:
            validate_creator_document_column(input_table, self.document_column)
        return FAISSVectorstorePortObjectSpec(embeddings_spec=embeddings_spec)

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

        return FAISSVectorstorePortObject(
            FAISSVectorstorePortObjectSpec(embeddings.spec), embeddings, vectorstore=db
        )


@knext.node(
    "FAISS Vector Store Reader",
    knext.NodeType.SOURCE,
    faiss_icon,
    category=faiss_category,
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
    Reads a FAISS vector store from a local path.

    FAISS is a library that provides indexing methods and algorithms optimized
    for performing similarity search on large-scale vector collections.

    A vector store is a data structure or storage mechanism that stores a collection of numerical vectors
    along with their corresponding documents. The vector store enables efficient storage, retrieval, and similarity
    search operations on these vectors.

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
    ) -> FAISSVectorstorePortObjectSpec:
        return FAISSVectorstorePortObjectSpec(embeddings_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> FAISSVectorstorePortObject:
        # TODO: Add check if .fiass and .pkl files are in the directory instead of instatiating as check
        faiss = FAISS.load_local(
            self.persist_directory, embeddings_port_object.create_model(ctx)
        )
        return FAISSVectorstorePortObject(
            FAISSVectorstorePortObjectSpec(embeddings_port_object.spec),
            embeddings_port_object,
            vectorstore=faiss,
        )
