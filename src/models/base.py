import knime.extension as knext
import pickle
import pandas as pd


class ModelPortObjectSpecContent(knext.PortObjectSpec):
    pass


class ModelPortObjectSpec(knext.PortObjectSpec):
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
    def deserialize(cls, data: dict) -> "ModelPortObjectSpec":
        content_cls = cls.content_registry[data["type"]]
        return content_cls.deserialize(data["content"])


class ModelPortObjectContent(knext.PortObject):
    def create_model(self, ctx):
        raise NotImplementedError()

    # TODO: Why do i need to name 'serialize' and 'deserialize' here?
    def serialize(self):
        pass

    def deserialize():
        pass


class ModelPortObject(knext.PortObject):
    content_registry = {}

    def __init__(
        self, spec: knext.PortObjectSpec, content: ModelPortObjectContent
    ) -> None:
        super().__init__(spec)
        self._content = content

    def create_model(self, ctx):
        return self._content.create_model(self, ctx)

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
    def deserialize(cls, spec: ModelPortObjectSpec, data) -> "ModelPortObject":
        config = pickle.loads(data)
        content_cls = cls.content_registry[config["type"]]
        return cls(spec, content_cls)


class LLMPortObjectSpec(ModelPortObjectSpec):
    def __init__(self, content: ModelPortObjectSpecContent) -> None:
        super().__init__(content)

    @classmethod
    def deserialize(cls, data: dict) -> "LLMPortObjectSpec":
        content_cls = cls.content_registry[data["type"]]
        return content_cls.deserialize(data["content"])


class LLMPortObject(ModelPortObject):
    @classmethod
    def deserialize(cls, spec: LLMPortObjectSpec, data) -> "LLMPortObject":
        config = pickle.loads(data)
        content_cls = cls.content_registry[config["type"]]
        return cls(spec, content_cls)


class ChatModelPortObjectSpec(ModelPortObjectSpec):
    def __init__(self, content: ModelPortObjectSpecContent) -> None:
        super().__init__(content)

    @classmethod
    def deserialize(cls, data: dict) -> "ChatModelPortObjectSpec":
        content_cls = cls.content_registry[data["type"]]
        return content_cls.deserialize(data["content"])


class ChatModelPortObject(ModelPortObject):
    @classmethod
    def deserialize(cls, spec: ChatModelPortObjectSpec, data) -> "ChatModelPortObject":
        config = pickle.loads(data)
        content_cls = cls.content_registry[config["type"]]
        return cls(spec, content_cls)


class EmbeddingsPortObjectSpec(ModelPortObjectSpec):
    def __init__(self, content: ModelPortObjectSpecContent) -> None:
        super().__init__(content)

    @classmethod
    def deserialize(cls, data: dict) -> "EmbeddingsPortObjectSpec":
        content_cls = cls.content_registry[data["type"]]
        return content_cls.deserialize(data["content"])


class EmbeddingsPortObject(ModelPortObject):
    @classmethod
    def deserialize(
        cls, spec: EmbeddingsPortObjectSpec, data
    ) -> "EmbeddingsPortObject":
        config = pickle.loads(data)
        content_cls = cls.content_registry[config["type"]]
        return cls(spec, content_cls)


llm_port_type = knext.port_type("LLM", LLMPortObject, LLMPortObjectSpec)
chat_model_port_type = knext.port_type(
    "Chat Model", ChatModelPortObject, ChatModelPortObjectSpec
)
embeddings_port_type = knext.port_type(
    "Embeddings", EmbeddingsPortObject, EmbeddingsPortObjectSpec
)


@knext.node("LLM Prompter", knext.NodeType.SOURCE, "", "")
@knext.input_port("LLM", "A large language model.", llm_port_type)
@knext.input_table("Prompt Table", "A table containing a string column with prompts.")
@knext.output_table(
    "Answer Table", "A table containing prompts and their respective answer."
)
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
        input_table: knext.Table,
    ):
        # TODO: Check for string column in input

        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Prompts"),
                knext.Column(knext.string(), "Answers"),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm_port: LLMPortObject,
        input_table: knext.Table,
    ):
        prompts = input_table.to_pandas()
        df = pd.DataFrame(prompts)

        llm = llm_port.create_model(ctx)
        answers = []

        for prompt in df[self.promt_column]:
            answers.append(llm(prompt))

        result_table = pd.DataFrame()
        result_table["Prompts"] = prompts
        result_table["Answers"] = answers

        return knext.Table.from_pandas(result_table)
