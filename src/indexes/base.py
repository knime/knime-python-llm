from typing import Dict
import knime.extension as knext
import pickle

from models.base import (
    LLMPortObject,
    EmbeddingsPortObject,
    LLMPortObjectSpec,
    ModelPortObjectSpecContent,
    llm_port_type
)

from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents.agent_toolkits import (
    VectorStoreToolkit,
    VectorStoreInfo,
)

import logging

LOGGER = logging.getLogger(__name__)

class VectorStorePortObjectSpecContent(knext.PortObjectSpec):
    pass


class VectorStorePortObjectSpec(knext.PortObjectSpec):
    content_registry = {}

    def __init__(self, content: ModelPortObjectSpecContent) -> None:
        super().__init__()
        self._content = content

    def serialize(self):
        return {"type": str(type(self._content)), "content": self._content.serialize()}

    @classmethod
    def register_content_type(cls, content_type: type):
        cls.content_registry[str(content_type)] = content_type

    @classmethod
    def deserialize(cls, data: dict) -> "VectorStorePortObjectSpec":
        content_cls = cls.content_registry[data["type"]]
        return content_cls.deserialize(data["content"])


class VectorStorePortObjectContent(knext.PortObject):
    def __init__(
        self, spec: knext.PortObjectSpec, 
        embeddings_model: EmbeddingsPortObject
    ) -> None:
        super().__init__(spec)
        self._embeddings_model = embeddings_model

    def load_store(self, ctx):
        raise NotImplementedError()

    def serialize(self):
        config = {"embeddings_model": self._embeddings_model}
        return pickle.dumps(config)

    @classmethod
    def deserialize(
        cls, spec: VectorStorePortObjectSpec, data
    ) -> "VectorStorePortObjectContent":
        config = pickle.loads(data)
        return cls(spec, config["embeddings_model"])


class VectorStorePortObject(knext.PortObject):
    content_registry = {}

    def __init__(
        self, spec: knext.PortObjectSpec, content: VectorStorePortObjectContent
    ) -> None:
        super().__init__(spec)
        self._content = content
    
    def load_store(self, ctx):
        return self._content.load_store(ctx)

    def serialize(self):
        config = {
            "type": str(type(self._content)),
            "content": self._content.serialize(),
        }
        return pickle.dumps(config)

    @classmethod
    def register_content_type(cls, content_type: type):
        cls.content_registry[str(content_type)] = content_type

    @classmethod
    def deserialize(
        cls, spec: VectorStorePortObjectSpec, data
    ) -> "VectorStorePortObject":
        config = pickle.loads(data)
        content_cls = cls.content_registry[config["type"]]
        return cls(spec, content_cls.deserialize(spec, config["content"]))


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

    def create_tool(self):
        raise NotImplementedError()
    
class VectorToolPortObjectSpec(ToolPortObjectSpec):
    pass

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
                retriever= vectorstore.as_retriever()
            )

    def create_tool(self, ctx):
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
    "Vector Store to Tool",
    knext.NodeType.SOURCE,
    icon_path="icons/chroma.png",
    category=""
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
                spec=VectorToolPortObjectSpec(
                    self.tool_name, self.tool_description
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
    "Tool Combiner",
    knext.NodeType.SOURCE,
    icon_path="icons/chroma.png",
    category=""
)
@knext.input_port("LLM", "A llm to search through the vector store.", tool_list_port_type)
@knext.input_port("Vector Store", "A vector store transform into a Tool object for an agent to use.", tool_list_port_type)
@knext.output_port("Tool","A tool object for an agent to use", tool_list_port_type)
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
        tool_list_twp: ToolListPortObject

    ):
        tool_list = []
        tool_list = tool_list + tool_list_one.tool_list
        tool_list = tool_list + tool_list_twp.tool_list

        return ToolListPortObject(
            ToolListPortObjectSpec(),
            tool_list
        )