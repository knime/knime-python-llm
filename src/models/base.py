from knime.api.schema import PortObjectSpec
import knime.extension as knext
import pickle
import logging

LOGGER = logging.getLogger(__name__)
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
            "type": str(type(self._content)),
            "content": self._content.serialize()
        }

    @classmethod
    def register_content_type(cls, content_type: type):
        cls.content_registry[str(content_type)] = content_type

    @classmethod
    def deserialize(cls, data: dict) -> "LLMPortObjectSpec":
        content_cls = cls.content_registry[data["type"]]
        return content_cls.deserialize(data["content"])
    


class LLMPortObjectContent(knext.PortObject):

        def create_llm(self, ctx):
            raise NotImplementedError()
        
        #TODO: Why do i need to name 'serialize' and 'deserialize' here?
        def serialize(self):
            pass

        def deserialize():
            pass

    

class LLMPortObject(knext.PortObject):

    content_registry = {}

    def __init__(self, spec: knext.PortObjectSpec, content: LLMPortObjectContent) -> None:
        super().__init__(spec)
        self._content = content
    
    def create_llm(self, ctx):
        return self._content.create_llm(self, ctx)


    def serialize(self):
        config = {
            "type": str(type(self._content)),
            "content": self._content.serialize()
        }
        return pickle.dumps(config)
    
    @classmethod
    def register_content_type(cls, content_type: type):
        cls.content_registry[str(content_type)] = content_type

    @classmethod
    def deserialize(cls, spec: LLMPortObjectSpec, data) -> "LLMPortObject":
        config = pickle.loads(data)
        # TODO understand why str is necessary here. Look at the type of config["type"]
        content_cls = cls.content_registry[config["type"]]
        return cls(spec, content_cls)

#.deserialize(spec.content, config["content"])
llm_port_type = knext.port_type("LLM", LLMPortObject, LLMPortObjectSpec)

@knext.node(
    "LLM Prompter", knext.NodeType.SOURCE, "icons/ml.svg", ""
)
@knext.input_port("LLM", "A large language model.", llm_port_type)
@knext.input_table("Prompt Table", "A table containing prompts.")
@knext.output_table("Answer Table", "A table containing prompts and answers.")
class LLMPrompter:

    promt_column = knext.ColumnParameter(
        "Prompt column",
        """Selection of column used as the prompts column.""",
        port_index=1,
    )

    def configure(
        self, 
        ctx: knext.ConfigurationContext, 
        llm_spec: LLMPortObjectSpec,
        input_table: knext.Table
    ):
        #TODO: Check for string column in input

        return knext.Schema.from_columns([
                knext.Column(knext.string(), "Prompts"),
                knext.Column(knext.string(), "Answers"),
            ]
        )

    def execute(
        self, 
        ctx: knext.ExecutionContext, 
        llm_port: LLMPortObject,
        input_table: knext.Table
    ):
        import pandas as pd

        prompts = input_table.to_pandas()
        df = pd.DataFrame(prompts)


        llm = llm_port.create_llm(ctx)
        answers = []

        for prompt in df[self.promt_column]:
            answers.append(llm(prompt))
        
        result_table = pd.DataFrame()
        result_table["Prompts"] = prompts
        result_table["Answers"] = answers
        
        return knext.Table.from_pandas(result_table)