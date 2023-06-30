# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods



import knime.extension as knext
import pandas as pd

from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
)

from .base import (
    VectorStorePortObject,
    VectorStorePortObjectSpec,
    VectorStorePortObjectSpecContent,
    VectorStorePortObjectContent,
    vector_store_port_type,
    store_category
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

class ChromaVectorstorePortObjectSpecContent(VectorStorePortObjectSpecContent):
    def __init__(self, persist_directory) -> None:
        super().__init__()
        self._persist_directory = persist_directory

    def serialize(self) -> dict:
        return {
            "persist_directory": self._persist_directory,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["persist_directory"])


VectorStorePortObjectSpec.register_content_type(ChromaVectorstorePortObjectSpecContent)


class ChromaVectorstorePortObjectContent(VectorStorePortObjectContent):
    def __init__(
        self, spec: knext.PortObjectSpec, embeddings_port_object: EmbeddingsPortObject
    ) -> None:
        super().__init__(spec, embeddings_port_object)
        self._embeddings_port_object = embeddings_port_object

    def load_store(self, ctx):
        return Chroma(
            persist_directory= self.spec.serialize()["persist_directory"],
        )


VectorStorePortObject.register_content_type(ChromaVectorstorePortObjectContent)


@knext.node(
    "Chroma Vector Store Creator",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
)
@knext.input_port(
    "Embeddings",
    "The embeddings model to use for the vector store.",
    embeddings_port_type,
)
@knext.input_table(
    name="Documents",
    description="""Table containing a string column representing documents that will be used in the vector store.""",
)
@knext.output_port(
    "FAISS Vector Store", "The created FAISS vector store.", vector_store_port_type
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
    ):
        return VectorStorePortObjectSpec(self.create_spec_content())

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings: EmbeddingsPortObject,
        input_table: knext.Table,
    ) -> VectorStorePortObject:
        df = input_table.to_pandas()
        
        documents = [Document(page_content=text) for text in df[self.document_column]]
        db = Chroma.from_documents(
            documents= documents, 
            embedding= embeddings.create_model(ctx), 
            persist_directory=self.persist_directory
            )
        
        db.persist()

        return VectorStorePortObject(
            spec=VectorStorePortObjectSpec(self.create_spec_content()),
            content=ChromaVectorstorePortObjectContent(
                self.create_spec_content(), embeddings
            ),
        )

    def create_spec_content(self):
        return ChromaVectorstorePortObjectSpecContent(self.persist_directory)


@knext.node(
    "Chroma Vector Store Loader",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
)
@knext.input_port(
    "Embeddings",
    "The embeddings model to use for the vector store.",
    embeddings_port_type,
)
@knext.output_port(
    "Chroma Vector Store", "The loaded vector store.", vector_store_port_type
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
    ) -> VectorStorePortObjectSpec:
        return VectorStorePortObjectSpec(self.create_spec_content())

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> VectorStorePortObject:
        # TODO: Add check if Chroma files are here instead of instantiation

        Chroma(self.persist_directory,)

        return VectorStorePortObject(
            spec=VectorStorePortObjectSpec(self.create_spec_content()),
            content=ChromaVectorstorePortObjectContent(
                self.create_spec_content(), embeddings_port_object
            ),
        )

    def create_spec_content(self):
        return ChromaVectorstorePortObjectSpecContent(self.persist_directory)


@knext.node(
    "Chroma Vector Store Retriever",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
)
@knext.input_port("Vector Store", "A vector store port object.", vector_store_port_type)
@knext.input_table(
    "Queries", "Table containing a string column with the queries for the vector store."
)
@knext.output_table(
    "Result table", "Table containing the queries and their closest match from the db."
)
class ChromaVectorStoreRetriever:
    query_column = knext.ColumnParameter(
        "Queries", "Column containing the queries", port_index=1
    )

    top_k = knext.IntParameter(
        "Number of results",
        "Number of top results to get from vector store search. Ranking from best to worst",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        vectorstore_spec: VectorStorePortObjectSpec,
        table_spec: knext.Schema,
    ):
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Queries"),
                knext.Column(knext.ListType(knext.string()), "Documents"),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        vectorstore: VectorStorePortObject,
        input_table: knext.Table,
    ):
        db = vectorstore.load_store(ctx)

        queries = input_table.to_pandas()
        df = pd.DataFrame(queries)

        doc_collection = []

        for query in df[self.query_column]:
            similar_documents = db.similarity_search(query, k=self.top_k)

            relevant_documents = []

            for document in similar_documents:
                relevant_documents.append(document.page_content)

            doc_collection.append(relevant_documents)

        result_table = pd.DataFrame()
        result_table["Queries"] = queries
        result_table["Documents"] = doc_collection

        return knext.Table.from_pandas(result_table)