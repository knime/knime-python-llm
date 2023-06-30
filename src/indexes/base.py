# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods


from typing import Dict
import knime.extension as knext
import pickle

from models.base import (
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,

    EmbeddingsPortObject,
)
import util

from langchain.chains import RetrievalQA
from langchain.tools import Tool


store_category = knext.category(
    path=util.main_cat,
    level_id="stores",
    name="Vector Stores",
    description="",
    icon="icons/store.png",
)

class VectorStorePortObjectSpec(knext.PortObjectSpec):
    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: Dict):
        return cls()
    
class VectorStorePortObject(knext.PortObject):
    def __init__(self, spec: VectorStorePortObjectSpec, embeddings_model: EmbeddingsPortObject) -> None:
        super().__init__(spec)
        self._embeddings_model = embeddings_model

    def load_store(self, ctx):
        raise NotImplementedError()

    def serialize(self):
        config = {"embeddings_model": self._embeddings_model}
        return pickle.dumps(config)

    @classmethod
    def deserialize(cls, spec: VectorStorePortObjectSpec, data) -> "VectorStorePortObject":
        config = pickle.loads(data)
        return cls(spec, config["embeddings_model"])
    

vector_store_port_type = knext.port_type(
    "Vectorstore", VectorStorePortObject, VectorStorePortObjectSpec
)


class ToolPortObjectSpec(knext.PortObjectSpec):
    def __init__(self, name, description) -> None:
        super().__init__()
        self._name = name
        self._description = description

    def serialize(self) -> dict:
        return {
            "name": self._name,
            "description": self._description
        }
    
    @classmethod
    def deserialize(cls, data: Dict):
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
            "top_k": self._top_k
        }

    @classmethod
    def deserialize(cls, data: Dict):
        return cls(data["name"], data["description"], data["top_k"])

class VectorToolPortObject(ToolPortObject):

    def __init__(self, spec: ToolPortObjectSpec, llm_port: LLMPortObject, vectorstore_port: VectorStorePortObject) -> None:
        super().__init__(spec)
        self._lmm_port = llm_port
        self._vectorestore_port = vectorstore_port

    def serialize(self) -> bytes:
        port_objects = {
            "llm": self._lmm_port,
            "vectorstore": self._vectorestore_port
        }
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
                retriever= vectorstore.as_retriever(search_kwargs = {"k": self.spec.serialize()["top_k"]})
            )

    def create(self, ctx):
        return Tool(
            name= self.spec.serialize()["name"],
            func= self._create_function(ctx).run,
            description= self.spec.serialize()["description"]
        )

class ToolListPortObjectSpec(knext.PortObjectSpec):
    
    def serialize(self) -> dict:
        return {}
    
    @classmethod
    def deserialize(cls, data):
        return cls()

class ToolListPortObject(knext.PortObject):
    def __init__(self, spec: knext.PortObjectSpec, tool_list) -> None:
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

tool_list_port_type = knext.port_type("Tool list", ToolListPortObject, ToolListPortObjectSpec)

@knext.node(
    "Vector Store to Toollist",
    knext.NodeType.SOURCE,
    icon_path="icons/store.png",
    category=store_category
)
@knext.input_port("LLM", "A llm to search through the vector store.", llm_port_type)
@knext.input_port("Vector Store", "A vector store transform into a Tool object for an agent to use.", vector_store_port_type)
@knext.output_port("Tool","A tool object for an agent to use", tool_list_port_type)
class VectorStoreToTool:

    tool_name = knext.StringParameter(
        label="Tool name",
        description="The name for the Tool"
    )

    tool_description = knext.StringParameter(
        label="Tool description",
        description="""The descripton for the tool through which an agent decides whether to use the tool. Provide a meaningful
        description under which circumstances the agent should try to use it."""
    )

    top_k = knext.IntParameter(
        label="Retrieved documents",
        description="The number of top results from the vector store that the tool will provide",
        default_value=5,
        is_advanced=True
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm_spec: LLMPortObjectSpec,
        vectorstore_spec: VectorStorePortObjectSpec
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
                spec= VectorToolPortObjectSpec(
                    self.tool_name, self.tool_description, self.top_k
                ),
                llm_port= llm_port,
                vectorstore_port= vectorstore
            )
        )

        return ToolListPortObject(
            ToolListPortObjectSpec(),
            tool_list
        )
    
@knext.node(
    "Tool List Combiner",
    knext.NodeType.SOURCE,
    icon_path="icons/store.png",
    category=store_category
)
@knext.input_port("Tool List", "A list of tools for an agent to use.", tool_list_port_type)
@knext.input_port("Tool List", "A list of tools for an agent to use.", tool_list_port_type)
@knext.output_port("Tool List","The concatenated tools from both lists.", tool_list_port_type)
class ToolCombiner:

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        tool_list_one: ToolListPortObjectSpec,
        tool_list_twp: ToolListPortObjectSpec
    ):
        
        return ToolListPortObjectSpec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        tool_list_one: ToolListPortObject,
        tool_list_two: ToolListPortObject

    ):
        tool_list_one._tool_list = tool_list_one._tool_list + tool_list_two.tool_list

        return tool_list_one