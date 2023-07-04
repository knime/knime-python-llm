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
        # HACK because Chroma doesn't allow to set the path in another way
        vectorstore._persist_directory = vectorstore_folder
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
    # TODO add node description

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
        # TODO validate table
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

        return ChromaVectorstorePortObject(ChromaVectorstorePortObjectSpec(embeddings.spec), embeddings, vectorstore=db)


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
# TODO rename to reader
class ChromaVectorStoreLoader:
    # TODO add node description
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
        chroma = Chroma(self.persist_directory, embeddings_port_object.create_model(ctx))
        return ChromaVectorstorePortObject(ChromaVectorstorePortObjectSpec(embeddings_port_object.spec), embeddings_port_object, vectorstore=chroma)

