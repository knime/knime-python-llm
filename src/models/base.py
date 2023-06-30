# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods


from typing import Dict
import knime.extension as knext
import pandas as pd

import util

model_category = knext.category(
    path=util.main_cat,
    level_id="models",
    name="Models",
    description="",
    icon="icons/ml.svg",
)

class LLMPortObjectSpec(knext.PortObjectSpec):
    def serialize(self):
        return {}

    @classmethod
    def deserialize(cls, data: Dict):
        return cls()
    
class LLMPortObject(knext.PortObject):

    def __init__(self, spec: LLMPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""
    
    @classmethod
    def deserialize(cls, spec: LLMPortObjectSpec, storage: bytes):
        return cls(spec)

    def create_model(self, ctx):
        raise NotImplementedError()
    
class ChatModelPortObjectSpec(knext.PortObjectSpec):
    def serialize(self):
        return {}

    @classmethod
    def deserialize(cls, data: Dict):
        return cls()

class ChatModelPortObject(knext.PortObject):

    def __init__(self, spec: ChatModelPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self):
        return b""

    @classmethod
    def deserialize(cls, spec, data: Dict):
        return cls(spec)

    def create_model(self, ctx):
        raise NotImplementedError()

class EmbeddingsPortObjectSpec(knext.PortObjectSpec):
    def serialize(self):
        return {}

    @classmethod
    def deserialize(cls, data: Dict):
        return cls()
    
class EmbeddingsPortObject(knext.PortObject):

    def __init__(self, spec: EmbeddingsPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self):
        return b""

    @classmethod
    def deserialize(cls, spec, data: Dict):
        return cls(spec)

    def create_model(self, ctx):
        raise NotImplementedError()
    
llm_port_type = knext.port_type("LLM Port Type", LLMPortObject, LLMPortObjectSpec)
chat_model_port_type = knext.port_type("Chat Model Port Type", ChatModelPortObject, ChatModelPortObjectSpec)
embeddings_model_port_type = knext.port_type("Embeddings Port Type", EmbeddingsPortObject, EmbeddingsPortObjectSpec)


@knext.node("LLM Prompter", knext.NodeType.SOURCE, "", model_category)
@knext.input_port("LLM", "A large language model.", llm_port_type)
@knext.input_table("Prompt Table", "A table containing a string column with prompts.")
@knext.output_table("Result Table", "A table containing prompts and their respective answer.")
class LLMPrompter:
    prompt_column = knext.ColumnParameter(
        "Prompt column",
        """Selection of column used as the prompts column.""",
        port_index=1,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm_spec: LLMPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        nominal_columns = [
            (c.name, c.ktype) for c in input_table_spec if util.is_nominal(c)
        ]

        if len(nominal_columns) == 0:
            raise knext.InvalidParametersError("""The number of nominal columns are 0. Expected at least 
                one nominal column for prompts."""
            )
        
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), self.prompt_column),
                knext.Column(knext.string(), "Prompt Result"),
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

        for prompt in df[self.prompt_column]:
            answers.append(llm(prompt))

        df["Prompt Result"] = answers

        return knext.Table.from_pandas(df)
