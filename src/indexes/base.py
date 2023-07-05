# KNIME / own imports
import knime.extension as knext
from knime.extension.nodes import (
    FilestorePortObject,
    load_port_object,
    save_port_object,
    PortType,
    get_port_type_for_spec_type,
    get_port_type_for_id
)
from models.base import (
    EmbeddingsPortObject,
    EmbeddingsPortObjectSpec,
)

import pandas as pd
import util
from typing import Optional, Any
import os
import shutil

store_icon = "icons/store.png"
store_category = knext.category(
    path=util.main_category,
    level_id="stores",
    name="Vector Stores",
    description="",
    icon=store_icon,
)


class VectorStorePortObjectSpec(knext.PortObjectSpec):

    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: dict):
        return cls()


class VectorStorePortObject(knext.PortObject):
    def __init__(
        self, spec: VectorStorePortObjectSpec, embeddings_model: EmbeddingsPortObject
    ) -> None:
        super().__init__(spec)
        self._embeddings_model = embeddings_model

    @property
    def embeddings_model(self):
        return self._embeddings_model

    def load_store(self, ctx):
        raise NotImplementedError()


vector_store_port_type = knext.port_type(
    "Vectorstore", VectorStorePortObject, VectorStorePortObjectSpec
)


class FilestoreVectorstorePortObjectSpec(VectorStorePortObjectSpec):
    def __init__(
        self, embeddings_spec: EmbeddingsPortObjectSpec
    ) -> None:
        super().__init__()
        self._embeddings_port_type = get_port_type_for_spec_type(type(embeddings_spec))
        self._embeddings_spec = embeddings_spec

    @property
    def embeddings_port_type(self) -> PortType:
        return self._embeddings_port_type

    @property
    def embeddings_spec(self) -> EmbeddingsPortObjectSpec:
        return self._embeddings_spec

    def serialize(self) -> dict:
        return {
            "embeddings_port_type": self.embeddings_port_type.id,
            "embeddings_spec": self.embeddings_spec.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict):
        embeddings_port_type: PortType = get_port_type_for_id(data["embeddings_port_type"])
        embeddings_spec = embeddings_port_type.spec_class.deserialize(
            data["embeddings_spec"]
        )
        return cls(embeddings_spec)


class FilestoreVectorstorePortObject(FilestorePortObject, VectorStorePortObject):
    def __init__(
        self,
        spec: FilestoreVectorstorePortObjectSpec,
        embeddings_port_object: EmbeddingsPortObject,
        folder_path: Optional[str] = None,
        vectorstore: Optional[Any] = None,
    ):
        super().__init__(spec, embeddings_port_object)
        self._folder_path = folder_path
        self._vectorstore = vectorstore

    def load_store(self, ctx):
        if self._vectorstore is None:
            embeddings = self.embeddings_model.create_model(ctx)
            self._vectorstore = self.load_vectorstore(
                embeddings, self._vectorstore_path
            )
        return self._vectorstore

    def load_vectorstore(self, embeddings, vectorstore_path):
        raise NotImplementedError()

    @classmethod
    def _embeddings_path(cls, folder_path: str) -> str:
        return os.path.join(folder_path, "embeddings")

    @property
    def _vectorstore_path(self) -> str:
        return os.path.join(self._folder_path, "vectorstore")

    def write_to(self, file_path):
        os.makedirs(file_path)
        save_port_object(self.embeddings_model, self._embeddings_path(file_path))
        if self._folder_path is not None and not self._folder_path == file_path:
            # copy the folder structures if we have a folder path from read_from
            shutil.copytree(self._vectorstore_path, os.path.join(file_path, "vectorstore"))
        else:
            self.save_vectorstore(os.path.join(file_path, "vectorstore"), self._vectorstore)

    def save_vectorstore(self, vectorstore_folder, vectorstore):
        raise NotImplementedError()

    @classmethod
    def _read_embeddings(
        cls, spec: FilestoreVectorstorePortObjectSpec, file_path: str
    ) -> EmbeddingsPortObject:
        return load_port_object(
            spec.embeddings_port_type.object_class,
            spec.embeddings_spec,
            cls._embeddings_path(file_path),
        )

    @classmethod
    def read_from(cls, spec: FilestoreVectorstorePortObjectSpec, file_path: str):
        embeddings_obj = cls._read_embeddings(spec, file_path)
        return cls(spec, embeddings_obj, file_path)

def pick_default_column(input_table: knext.Schema, ktype: knext.KnimeType):
    string_type = knext.string()
    for column in input_table:
        if column.ktype == string_type:
            return column.name
    raise knext.InvalidParametersError(f"The input table does not contain any columns of type '{str(ktype)}'.")

def validate_creator_document_column(input_table: knext.Schema, column: str):
    check_column(input_table, column, knext.string(), "document")
    
def check_column(input_table: knext.Schema, column: str, expected_type:knext.KnimeType, column_purpose: str):
    if not column in input_table.column_names:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column}' is missing in the input table."
        )
    ktype = input_table[column].ktype
    if ktype != expected_type:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column}' is of type {str(ktype)} but should be of type {str(expected_type)}."
        )


@knext.node(
    "Vector Store Retriever",
    knext.NodeType.SOURCE,
    store_icon,
    category=store_category,
)
@knext.input_port("Vector Store", "A vector store containing document embeddings.", vector_store_port_type)
@knext.input_table(
    "Queries", "Table containing a string column with the queries for the vector store."
)
@knext.output_table(
    "Result table", "Table containing the queries and their closest match from the db."
)
class VectorStoreRetriever:
    """
    Performs a similarity search on a vector store.

    A vector store retriever is a component or module that
    specializes in retrieving vectors from a vector store
    based on user queries. It works in conjunction with a
    vector store to facilitate efficient vector
    retrieval and similarity search operations.
    """

    query_column = knext.ColumnParameter(
        "Queries", "Column containing the queries.", port_index=1
    )

    top_k = knext.IntParameter(
        "Number of results",
        "Number of top results to get from vector store search. Ranking from best to worst.",
        default_value=3,
    )

    # TODO: Add options to retrieve meta data from the store
    def configure(
        self,
        ctx: knext.ConfigurationContext,
        vectorstore_spec: VectorStorePortObjectSpec,
        table_spec: knext.Schema,
    ):
        if self.query_column is None:
            self.query_column = pick_default_column(table_spec, knext.string())
        else:
            check_column(table_spec, self.query_column, knext.string(), "queries")
            
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

