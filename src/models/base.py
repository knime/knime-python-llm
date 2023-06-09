import knime.extension as knext
import pickle

class EmbeddingsPortObjectSpec(knext.PortObjectSpec):
    def __init__(self, credentials: str, model_name: str) -> None:
        super().__init__()
        self._credentials = credentials
        self._model_name = model_name

    @property
    def credentials(self):
        return self._credentials

    @property
    def model_name(self):
        return self._model_name

    def serialize(self) -> dict:
        return {"credentials": self._credentials, "model_name": self._model_name}

    @classmethod
    def deserialize(cls, data: dict) -> "EmbeddingsPortObjectSpec":
        return cls(data["credentials"], data["model_name"])


class EmbeddingsPortObject(knext.PortObject):
    def __init__(self, spec: knext.PortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(
        cls, spec: EmbeddingsPortObjectSpec, data: bytes
    ) -> "EmbeddingsPortObject":
        return cls(spec)


embeddings_port_type = knext.port_type(
    "Embeddings", EmbeddingsPortObject, EmbeddingsPortObjectSpec
)

class LLMPortObjectSpecContent(knext.PortObjectSpec):
    pass

class LLMPortObjectSpec(knext.PortObjectSpec):

    content_registry = {}

    def __init__(self, content: LLMPortObjectSpecContent) -> None:
        super().__init__()
        self._content = content

    def serialize(self):
        return {
            "type": str(type(self.content)),
            "content": self._content.serialize()
        }
    
    @property
    def content(self):
        return self._content

    @classmethod
    def register_content_type(cls, content_type: type):
        cls.content_registry[str(type)] = type

    @classmethod
    def deserialize(cls, data: dict) -> "LLMPortObjectSpec":
        content_cls = cls.content_registry[data["type"]]
        return content_cls.deserialize(data["content"])
    


class LLMPortObjectContent(knext.PortObject):

    def create_llm(self, ctx):
        pass

class LLMPortObject(knext.PortObject):

    content_registry = {}

    def __init__(self, spec: knext.PortObjectSpec, content: LLMPortObjectContent) -> None:
        super().__init__(spec)
        self._content = content
    
    def create_llm(self, ctx):
        return self._content.create_llm(ctx)


    def serialize(self):
        config = {
            "type": type(self._content), 
            "content": self._content.serialize()
        }
        return pickle.dumps(config)
    
    @classmethod
    def register_content_type(cls, content_type: type):
        cls.content_registry[str(content_type)] = content_type

    @classmethod
    def deserialize(cls, spec: LLMPortObjectSpec, data) -> "LLMPortObject":
        config = pickle.loads(data)
        content_cls = cls.content_registry[config["type"]]
        return cls(spec, content_cls.deserialize(spec.content, config["content"]))


llm_port_type = knext.port_type("LLM", LLMPortObject, LLMPortObjectSpec)

@knext.node(
    "LLM Prompter", knext.NodeType.SOURCE, "icons/ml.svg", ""
)
@knext.input_port("LLM", "A large language model.", llm_port_type)
#TODO: Output column with promt and answer instead of log
#@knext.output_table("Answer Table", "A table containing the answers to the promts.")
class LLMPrompter:
    # TODO: Take in column of promts
    prompt = knext.StringParameter("Prompt", "The prompt that wil be send", "")

    def configure(
        self, 
        ctx: knext.ConfigurationContext, 
        llm_spec: LLMPortObjectSpec
    ):
        #return knext.Schema.from_columns([
        #        knext.Column(knext.string(), "Promts"),
        #        knext.Column(knext.string(), "Results"),
        #    ]
        #)
        return llm_spec

    def execute(
        self, 
        ctx: knext.ExecutionContext, 
        llm_port: LLMPortObject
    ):
        
        llm = llm_port.create_llm(ctx)
        
        LOGGER.info(llm("What is an apple?"))
        
        return