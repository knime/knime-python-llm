# KNIME / own imports
import knime.extension as knext
import util
from knime.extension.nodes import (
    FilestorePortObject,
    load_port_object,
    save_port_object,
    PortType,
    get_port_type_for_spec_type,
    get_port_type_for_id,
)
from knime.extension.parameter import ManualFilterConfig
from models.base import (
    EmbeddingsPortObject,
    EmbeddingsPortObjectSpec,
)
from base import AIPortObjectSpec

import pandas as pd
from typing import Optional, Any
import os
import shutil
import logging

LOGGER = logging.getLogger(__name__)

store_icon = "icons/store.png"
store_category = knext.category(
    path=util.main_category,
    level_id="stores",
    name="Vector Stores",
    description="",
    icon=store_icon,
)


class MissingValueHandlingOptions(knext.EnumParameterOptions):
    SkipRow = (
        "Skip rows",
        "Rows with missing values will be ignored.",
    )
    Fail = (
        "Fail",
        "This node will fail during the execution.",
    )


def skip_missing_values(df: pd.DataFrame, col_name: str, ctx: knext.ExecutionContext):
    # Drops rows with missing values
    df_cleaned = df.dropna(subset=[col_name], how="any")
    n_skipped_rows = len(df) - len(df_cleaned)

    if n_skipped_rows > 0:
        ctx.set_warning(f"{n_skipped_rows} / {len(df)} rows are skipped.")

    return df_cleaned


def handle_missing_and_empty_documents(
    df: pd.DataFrame,
    document_column: str,
    missing_value_handling_setting: MissingValueHandlingOptions,
    ctx: knext.ExecutionContext,
):
    # Drops rows if SkipRow option is selected, otherwise fails
    # if there are any missing documents in the document column (=Fail option is selected)
    has_missing_values = df[document_column].isna().any()
    if (
        missing_value_handling_setting == MissingValueHandlingOptions.SkipRow
        and has_missing_values
    ):
        df = skip_missing_values(df, document_column, ctx)
    elif has_missing_values:
        missing_row_id = df[df[document_column].isnull()].index[0]
        raise knext.InvalidParametersError(
            f"There are missing documents in the document column. See row ID <{missing_row_id}> for the first row that contains a missing document."
        )

    if df.empty:
        raise knext.InvalidParametersError(
            f"""All rows are skipped due to missing documents."""
        )

    # Check for empty documents
    for id, document in df[document_column].items():
        if not document.strip():
            raise ValueError(
                f"Empty documents are not supported. See row ID {id} for the first empty document."
            )

    return df


def handle_missing_metadata_values(
    df: pd.DataFrame,
    metadatas: Optional[list[str]] = None,
):
    # Fills missing metadata values with empty string
    for metadata in metadatas:
        df[metadata] = df[metadata].fillna("")

    return df


class VectorstorePortObjectSpec(AIPortObjectSpec):
    """Marker interface for vector store specs. Used to define the most generic vector store PortType."""

    def __init__(self, metadata_column_names: Optional[list[str]] = None) -> None:
        super().__init__()
        self._metadata_column_names = (
            metadata_column_names if metadata_column_names is not None else []
        )

    @property
    def metadata_column_names(self) -> list[str]:
        return self._metadata_column_names

    def serialize(self) -> dict:
        return {"metadata_column_names": self.metadata_column_names}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data.get("metadata_column_names"))


class VectorstorePortObject(knext.PortObject):
    def __init__(
        self, spec: VectorstorePortObjectSpec, embeddings_model: EmbeddingsPortObject
    ) -> None:
        super().__init__(spec)
        self._embeddings_model = embeddings_model

    @property
    def embeddings_model(self):
        return self._embeddings_model

    def load_store(self, ctx):
        raise NotImplementedError()


vector_store_port_type = knext.port_type(
    "Vectorstore", VectorstorePortObject, VectorstorePortObjectSpec
)


class FilestoreVectorstorePortObjectSpec(VectorstorePortObjectSpec):
    def __init__(
        self,
        embeddings_spec: EmbeddingsPortObjectSpec,
        metadata_column_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__(metadata_column_names)
        self._embeddings_port_type = get_port_type_for_spec_type(type(embeddings_spec))
        self._embeddings_spec = embeddings_spec

    @property
    def embeddings_port_type(self) -> PortType:
        return self._embeddings_port_type

    @property
    def embeddings_spec(self) -> EmbeddingsPortObjectSpec:
        return self._embeddings_spec

    def validate_context(self, ctx: knext.ConfigurationContext):
        self._embeddings_spec.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "embeddings_port_type": self.embeddings_port_type.id,
            "embeddings_spec": self.embeddings_spec.serialize(),
            **super().serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict):
        embeddings_port_type: PortType = get_port_type_for_id(
            data["embeddings_port_type"]
        )
        embeddings_spec = embeddings_port_type.spec_class.deserialize(
            data["embeddings_spec"]
        )
        return cls(embeddings_spec, data.get("metadata_column_names"))


class FilestoreVectorstorePortObject(FilestorePortObject, VectorstorePortObject):
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
            shutil.copytree(
                self._vectorstore_path, os.path.join(file_path, "vectorstore")
            )
        else:
            self.save_vectorstore(
                os.path.join(file_path, "vectorstore"), self._vectorstore
            )

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


def validate_creator_document_column(input_table: knext.Schema, column: str):
    util.check_column(input_table, column, knext.string(), "document")


@knext.parameter_group(label="Metadata")
class MetadataSettings:
    metadata_columns = knext.ColumnFilterParameter(
        "Metadata columns",
        """Selection of columns used as metadata for each document. The documents column will be ignored.""",
        port_index=1,
        column_filter=util.create_type_filer(knext.string()),  # TODO fix typo
        default_value=lambda v: knext.ColumnFilterConfig(
            manual_filter=ManualFilterConfig(include_unknown_columns=False)
        )
        if v < knext.Version(5, 2, 0)
        else knext.ColumnFilterConfig(),
    )


def get_metadata_columns(
    metadata_columns, document_column, schema: knext.Schema
) -> list[str]:
    # metadata was introduced in 5.1.1 and the parameter is None for older versions
    if not metadata_columns:
        return []
    meta_data_columns = [column.name for column in metadata_columns.apply(schema)]
    try:
        meta_data_columns.remove(document_column)
    except:
        pass
    return meta_data_columns


@knext.node(
    "Vector Store Retriever",
    knext.NodeType.SOURCE,
    store_icon,
    category=store_category,
)
@knext.input_port(
    "Vector Store",
    "A vector store containing document embeddings.",
    vector_store_port_type,
)
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
        "Queries",
        "Column containing the queries.",
        port_index=1,
        column_filter=util.create_type_filer(knext.string()),
    )

    top_k = knext.IntParameter(
        "Number of results",
        "Number of top results to get from vector store search. Ranking from best to worst.",
        default_value=3,
    )

    retrieved_docs_column_name = knext.StringParameter(
        "Retrieved document column name",
        "The name for the appended column containing the retrieved documents.",
        "Retrieved documents",
    )

    retrieve_metadata = knext.BoolParameter(
        "Retrieve metadata from documents",
        "Whether or not to retrieve document metadata if provided.",
        default_value=False,
        since_version="5.2.0",
    )

    retrieve_similarity_scores = knext.BoolParameter(
        "Retrieve similarity scores",
        """Whether or not to retrieve similarity scores for the retrieved documents. 
        FAISS uses L2 distance and Chroma uses cosine distance to calculate similarity scores. 
        Lower score represents more similarity.""",
        default_value=False,
        since_version="5.3.0",
    )

    similarity_scores_column_name = knext.StringParameter(
        "Similarity scores column name",
        "The name for the appended column containing the similarity scores.",
        "Similarity scores",
        since_version="5.3.0",
    ).rule(knext.OneOf(retrieve_similarity_scores, [True]), knext.Effect.SHOW)

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        vectorstore_spec: VectorstorePortObjectSpec,
        table_spec: knext.Schema,
    ):
        vectorstore_spec.validate_context(ctx)

        if self.query_column:
            util.check_column(table_spec, self.query_column, knext.string(), "queries")
        else:
            self.query_column = util.pick_default_column(table_spec, knext.string())

        if not self.retrieved_docs_column_name:
            raise knext.InvalidParametersError(
                "No name for the column holding the retrieved documents is provided."
            )

        if not self.similarity_scores_column_name:
            raise knext.InvalidParametersError(
                "No name for the column holding the similarity scores is provided."
            )
        output_column_name = util.handle_column_name_collision(
            ctx, table_spec, self.retrieved_docs_column_name
        )

        table_spec = table_spec.append(
            self._create_column_list(vectorstore_spec, output_column_name)
        )
        return table_spec

    def execute(
        self,
        ctx: knext.ExecutionContext,
        vectorstore: VectorstorePortObject,
        input_table: knext.Table,
    ):
        db = vectorstore.load_store(ctx)
        num_rows = input_table.num_rows
        i = 0
        output_table: knext.BatchOutputTable = knext.BatchOutputTable.create()

        for batch in input_table.batches():
            doc_collection = []
            similarity_scores = []
            metadata_dict = {}

            df = batch.to_pandas()

            for query in df[self.query_column]:
                util.check_canceled(ctx)

                documents = db.similarity_search_with_score(query, k=self.top_k)

                doc_collection.append(
                    [document[0].page_content for document in documents]
                )

                if self.retrieve_metadata:

                    def to_str_or_none(metadata):
                        return str(metadata) if metadata is not None else None

                    for key in vectorstore.spec.metadata_column_names:
                        if key not in metadata_dict:
                            metadata_dict[key] = []
                        metadata_dict[key].append(
                            [
                                to_str_or_none(document[0].metadata.get(key))
                                for document in documents
                            ]
                        )
                if self.retrieve_similarity_scores:
                    similarity_scores.append([document[1] for document in documents])

                i += 1
                ctx.set_progress(i / num_rows)

            output_column_name = util.handle_column_name_collision(
                ctx, input_table.schema, self.retrieved_docs_column_name
            )

            df[output_column_name] = doc_collection

            for key in metadata_dict.keys():
                df[key] = metadata_dict[key]

            if self.retrieve_similarity_scores:
                df[self.similarity_scores_column_name] = similarity_scores

            output_table.append(df)

        return output_table

    def _create_column_list(
        self, vectorstore_spec, output_column_name
    ) -> list[knext.Column]:
        result_columns = [
            knext.Column(knext.ListType(knext.string()), output_column_name)
        ]

        if self.retrieve_metadata:
            for column_name in vectorstore_spec.metadata_column_names:
                result_columns.append(
                    knext.Column(knext.ListType(knext.string()), column_name)
                )

        if self.retrieve_similarity_scores:
            result_columns.append(
                knext.Column(
                    knext.ListType(knext.double()),
                    self.similarity_scores_column_name,
                )
            )

        return result_columns
