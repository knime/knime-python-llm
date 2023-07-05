import knime.extension as knext
from knime.extension.nodes import (
    get_port_type_for_id,
    get_port_type_for_spec_type,
    load_port_object,
    save_port_object,
    FilestorePortObject,
)
from .base import ToolPortObjectSpec, ToolPortObject
from ..models.base import LLMPortObject, LLMPortObjectSpec, llm_port_type
from ..indexes.base import (
    VectorStorePortObject,
    VectorStorePortObjectSpec,
    FilestoreVectorstorePortObjectSpec,
    FilestoreVectorstorePortObject,
    store_category,
    vector_store_port_type,
)
from .base import ToolListPortObject, ToolListPortObjectSpec, tool_list_port_type
from langchain.chains import RetrievalQA
from langchain.tools import Tool
import os


class VectorToolPortObjectSpec(ToolPortObjectSpec):
    def __init__(self, name, description, top_k) -> None:
        super().__init__(name, description)
        self._top_k = top_k

    @property
    def top_k(self):
        return self.top_k

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
        spec: VectorToolPortObjectSpec,
        llm_port: LLMPortObject,
        vectorstore_port: VectorStorePortObject,
    ) -> None:
        super().__init__(spec)
        self._llm = llm_port
        self._vectorstore = vectorstore_port

    @property
    def spec(self) -> VectorToolPortObjectSpec:
        return super().spec

    @property
    def llm(self) -> LLMPortObject:
        return self._llm

    @property
    def vectorstore(self) -> VectorStorePortObject:
        return self._vectorstore

    def _create_function(self, ctx):
        llm = self._llm.create_model(ctx)
        vectorstore = self._vectorestore_port.load_store(ctx)

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": self.spec.top_k}),
        )

    def create(self, ctx):
        return Tool(
            name=self.spec.serialize()["name"],
            func=self._create_function(ctx).run,
            description=self.spec.serialize()["description"],
        )


class FilestoreVectorToolPortObjectSpec(VectorToolPortObjectSpec):
    def __init__(
        self,
        name,
        description,
        top_k,
        llm_spec: LLMPortObjectSpec,
        vectorstore_spec: VectorStorePortObjectSpec,
    ) -> None:
        super().__init__(name, description, top_k)
        self._llm_spec = llm_spec
        self._llm_type = get_port_type_for_spec_type(llm_spec)
        self._vectorstore_spec = vectorstore_spec
        self._vectorstore_type = get_port_type_for_spec_type(vectorstore_spec)

    @property
    def llm_spec(self) -> LLMPortObjectSpec:
        return self._llm_spec

    @property
    def llm_type(self) -> knext.PortType:
        return self._llm_type

    @property
    def vectorstore_spec(self) -> VectorStorePortObjectSpec:
        return self._vectorstore_spec

    @property
    def vectorstore_type(self) -> knext.PortType:
        return self._vectorstore_type

    def serialize(self) -> dict:
        data = super().serialize()
        data["llm_spec"] = self.llm_spec.serialize()
        data["llm_type"] = self.llm_type.id
        data["vectorstore_spec"] = self.vectorstore_spec.serialize()
        data["vectorstore_type"] = self.vectorstore_type.id
        return data

    @classmethod
    def deserialize(cls, data: dict) -> FilestoreVectorstorePortObject:
        llm_type: knext.PortType = get_port_type_for_id(data["llm_type"])
        llm_spec = llm_type.spec_class.deserialize(data["llm_spec"])
        vectorstore_type: knext.PortType = get_port_type_for_id(
            data["vectorstore_type"]
        )
        vectorstore_spec = vectorstore_type.spec_class.deserialize(
            data["vectorstore_spec"]
        )
        return cls(
            data["name"], data["description"], data["top_k"], llm_spec, vectorstore_spec
        )


class FilestoreVectorToolPortObject(VectorToolPortObject, FilestorePortObject):
    def write_to(self, file_path: str) -> None:
        llm_path = os.path.join(file_path, "llm")
        save_port_object(self.llm, llm_path)
        vectorstore_path = os.path.join(file_path, "vectorstore")
        save_port_object(self._vectorstore, vectorstore_path)

    @classmethod
    def read_from(
        cls, spec: FilestoreVectorToolPortObjectSpec, file_path: str
    ) -> "FilestoreVectorToolPortObject":
        llm_path = os.path.join(file_path, "llm")
        llm = load_port_object(spec.llm_port_type, spec.llm_spec, llm_path)
        vectorstore_path = os.path.join(file_path, "vectorstore")
        vectorstore = load_port_object(
            spec.vectorstore_port_type, spec.vectorstore_spec, vectorstore_path
        )
        return cls(spec, llm, vectorstore)


# not actually output by any node but needs to be registered in the framework,
# such that the ToolListPortObject can load FilestoreVectorstorePortObjects via load_port_object
_filestore_vector_tool_port_type = knext.port_type(
    "Filestore Vector Store Tool",
    FilestoreVectorstorePortObject,
    FilestoreVectorstorePortObjectSpec,
)


# TODO: Add better descriptions
@knext.node(
    "Vector Store to Agent Tool",
    knext.NodeType.MANIPULATOR,
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
    ) -> ToolListPortObjectSpec:
        tool_spec = self._create_tool_spec(llm_spec, vectorstore_spec)
        return ToolListPortObjectSpec([tool_spec])

    def _create_tool_spec(
        self, llm_spec: LLMPortObjectSpec, vectorstore_spec: VectorStorePortObjectSpec
    ) -> FilestoreVectorstorePortObjectSpec:
        return FilestoreVectorToolPortObjectSpec(
            self.tool_name,
            self.tool_description,
            self.top_k,
            llm_spec,
            vectorstore_spec,
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm: LLMPortObject,
        vectorstore: VectorStorePortObject,
    ) -> ToolListPortObject:
        tool = FilestoreVectorToolPortObject(
            spec=self._create_tool_spec(llm.spec, vectorstore.spec),
            llm=llm,
            vectorstore=vectorstore,
        )
        return ToolListPortObject(ToolListPortObjectSpec([tool.spec]), [tool])
