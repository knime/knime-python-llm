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
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
    EmbeddingsPortObject,
    EmbeddingsPortObjectSpec,
)

# Langchain imports
from langchain.chains import RetrievalQA
from langchain.tools import Tool

import pandas as pd
import pickle
import util
from typing import Optional, Any
import os

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
        self._folder_path = file_path
        save_port_object(self.embeddings_model, self._embeddings_path(file_path))
        self.save_vectorstore(self._vectorstore_path, self._vectorstore)

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

class ToolPortObjectSpec(knext.PortObjectSpec):
    def __init__(self, name, description) -> None:
        super().__init__()
        self._name = name
        self._description = description

    def serialize(self) -> dict:
        return {"name": self._name, "description": self._description}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["name"], data["description"])


class ToolPortObject(knext.PortObject):
    def __init__(self, spec: ToolPortObjectSpec) -> None:
        super().__init__(spec)

    def create(self):
        raise NotImplementedError()


class VectorToolPortObjectSpec(ToolPortObjectSpec):
    def __init__(self, name, description, top_k) -> None:
        super().__init__(name, description)
        self._top_k = top_k

    def serialize(self) -> dict:
        return {
            "name": self._name,
            "description": self._description,
            "top_k": self._top_k,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["name"], data["description"], data["top_k"])


class VectorToolPortObject(ToolPortObject):
    def __init__(
        self,
        spec: ToolPortObjectSpec,
        llm_port: LLMPortObject,
        vectorstore_port: VectorStorePortObject,
    ) -> None:
        super().__init__(spec)
        self._lmm_port = llm_port
        self._vectorestore_port = vectorstore_port

    def serialize(self) -> bytes:
        port_objects = {"llm": self._lmm_port, "vectorstore": self._vectorestore_port}
        return pickle.dumps(port_objects)

    @classmethod
    def deserialize(cls, spec: knext.PortObjectSpec, data):
        port_objects = pickle.loads(data)
        return cls(spec, port_objects["llm"], port_objects["vectorstore"])

    def _create_function(self, ctx):
        llm = self._lmm_port.create_model(ctx)
        vectorstore = self._vectorestore_port.load_store(ctx)

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": self.spec.serialize()["top_k"]}
            ),
        )

    def create(self, ctx):
        return Tool(
            name=self.spec.serialize()["name"],
            func=self._create_function(ctx).run,
            description=self.spec.serialize()["description"],
        )


class ToolListPortObjectSpec(knext.PortObjectSpec):
    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data):
        return cls()


class ToolListPortObject(knext.PortObject):
    def __init__(self, spec: ToolListPortObjectSpec, tool_list) -> None:
        super().__init__(spec)
        self._tool_list = tool_list

    @property
    def tool_list(self):
        return self._tool_list

    def serialize(self) -> bytes:
        tool_list = {"tool_list": self._tool_list}
        return pickle.dumps(tool_list)

    @classmethod
    def deserialize(cls, spec: ToolListPortObjectSpec, data):
        tool_list = pickle.loads(data)
        return cls(spec, tool_list["tool_list"])


tool_list_port_type = knext.port_type(
    "Tool list", ToolListPortObject, ToolListPortObjectSpec
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


# TODO: Add better descriptions
@knext.node(
    "Vector Store to Agent Tool",
    knext.NodeType.SOURCE,
    icon_path="icons/store.png",
    category=store_category,
)
@knext.input_port("LLM Port", "A large language model.", llm_port_type)
@knext.input_port("Vector Store Port", "A loaded vector store.", vector_store_port_type)
@knext.output_port("Agent Tool", "A tool for an agent to use.", tool_list_port_type)
class VectorStoreToTool:
    """
    Creates an agent tool from a vector store.

    The power of an agent is that it can decide whether it needs to
    make use of an provided tool (e.g. looking for data in a vector store) to
    answer questions.

    An agent needs to be provided with the store and the information of its content.
    A meaningful name and description are very important:

    Example:

    Name: KNIME Node Description QA System

    Description: Use this tool whenever you need information about which nodes a user would need in a given
    situation or if you need information about nodes' configuration options.
    """

    tool_name = knext.StringParameter(
        label="Tool name", description="The name for the Tool."
    )

    tool_description = knext.StringParameter(
        label="Tool description",
        description="""The description for the tool through which an agent decides whether to use the tool or not. 
        Provide a meaningful description to make the agent decide more optimally.""",
    )

    top_k = knext.IntParameter(
        label="Retrieved documents",
        description="The number of top results that the tool will provide from the vector store.",
        default_value=5,
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm_spec: LLMPortObjectSpec,
        vectorstore_spec: VectorStorePortObjectSpec,
    ):
        return ToolListPortObjectSpec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm_port: LLMPortObject,
        vectorstore: VectorStorePortObject,
    ):
        tool_list = []
        tool_list.append(
            VectorToolPortObject(
                spec=VectorToolPortObjectSpec(
                    self.tool_name, self.tool_description, self.top_k
                ),
                llm_port=llm_port,
                vectorstore_port=vectorstore,
            )
        )

        return ToolListPortObject(ToolListPortObjectSpec(), tool_list)


@knext.node(
    "Tool Concatinator",
    knext.NodeType.SOURCE,
    icon_path="icons/store.png",
    category=store_category,
)
@knext.input_port(
    "Agent Tool(s)", "One or more tools for an agent to use.", tool_list_port_type
)
@knext.input_port(
    "Agent Tool(s)", "One or more tools for an agent to use.", tool_list_port_type
)
@knext.output_port(
    "Agent Tools",
    "The concatenated tool list for an agent to use.",
    tool_list_port_type,
)
class ToolCombiner:
    """
    Concatinates two Tools.

    A agent can be provided with a list of tools to choose from. Use this
    node to concatinate existing tools into a list and provide an agent with the tool list.
    """

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        spec_one: ToolListPortObjectSpec,
        spec_two: ToolListPortObjectSpec,
    ):
        return ToolListPortObjectSpec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        object_one: ToolListPortObject,
        object_two: ToolListPortObject,
    ):
        object_one._tool_list = object_one._tool_list + object_two.tool_list

        return object_one
