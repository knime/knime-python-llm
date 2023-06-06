import knime.extension as knext
from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
)
from .base import (
    VectorStorePortObjectSpec,
    VectorStorePortObject,
    vectorstore_port_type,
)

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import logging

LOGGER = logging.getLogger(__name__)

vectorstore_category = ""
chroma_icon = "./icons/chroma.png"


@knext.node(
    "Vectorstore Creator",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=vectorstore_category,
)
@knext.input_port(
    "Embeddings", "The embeddings to use in the vectorstore.", embeddings_port_type
)
@knext.input_table(
    name="Documents",
    description="""Table containing the documents that will be input for the vectorspace.""",
)
@knext.output_port("Vectorstore", "The loaded vectorstore.", vectorstore_port_type)
class VectorStoreCreator:
    document_column = knext.ColumnParameter(
        "Document column",
        """Selection of column used as the document column.""",
        port_index=1,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings: EmbeddingsPortObjectSpec,
        input_table: knext.Schema,
    ) -> VectorStorePortObjectSpec:
        return VectorStorePortObjectSpec(None)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings: EmbeddingsPortObject,
        input_table: knext.Table,
    ) -> VectorStorePortObject:
        # Convert input table to pandas dataframes
        df = input_table.to_pandas()

        # Load document objects from rows of document_column
        documents = [Document(page_content=text) for text in df[self.document_column]]

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        LOGGER.info(docs)

        openai_embeddings = OpenAIEmbeddings(
            model=embeddings.spec.model_name,
            openai_api_key=ctx.get_credentials(embeddings.spec.credentials).password,
        )

        # Embed and store the texts
        persist_directory = "db"
        vectordb = Chroma.from_documents(
            documents, openai_embeddings, persist_directory=persist_directory
        )

        # # for testing purposes
        # query = "What did the president say about Ketanji Brown Jackson"
        # docs = vectordb.similarity_search(query)
        # LOGGER.info(docs[0].page_content)

        vectordb.persist()
        vectordb = None

        return VectorStorePortObject(VectorStorePortObjectSpec(persist_directory))


@knext.node(
    "Vectorstore Reader",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=vectorstore_category,
)
@knext.input_port(
    "Embeddings", "The embeddings to use in the vectorstore.", embeddings_port_type
)
@knext.output_port("Vectorstore", "The loaded vectorstore.", vectorstore_port_type)
class VectorStoreLoader:
    directory = knext.StringParameter(
        "Directory of the vector store",
        "",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings: EmbeddingsPortObjectSpec,
    ) -> VectorStorePortObjectSpec:
        return VectorStorePortObjectSpec(None)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings: EmbeddingsPortObject,
    ) -> VectorStorePortObject:
        persist_directory = self.directory

        openai_embeddings = OpenAIEmbeddings(
            model=embeddings.spec.model_name,
            openai_api_key=ctx.get_credentials(embeddings.spec.credentials).password,
        )

        # Now we can load the persisted database from disk, and use it as normal
        vectordb = Chroma(
            persist_directory=persist_directory, embedding_function=openai_embeddings
        )

        # # for testing purposes
        # query = "What did the president say about Ketanji Brown Jackson"
        # docs = vectordb.similarity_search(query)
        # LOGGER.info(docs[0].page_content)

        return VectorStorePortObject(VectorStorePortObjectSpec(persist_directory))
