# TODO: Done Alex -- check?
# TODO: Node idea: Chat Prompter that gets a conversation table

# KNIME / own imports
import knime.extension as knext
import util

# TODO: Add category description?
# TODO: Get someone to do new icons
model_category = knext.category(
    path=util.main_cat,
    level_id="models",
    name="Models",
    description="",
    icon="icons/ml.png",
)


class LLMPortObjectSpec(knext.PortObjectSpec):
    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: dict):
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


llm_port_type = knext.port_type("LLM Port", LLMPortObject, LLMPortObjectSpec)


class ChatModelPortObjectSpec(knext.PortObjectSpec):
    def serialize(self):
        return {}

    @classmethod
    def deserialize(cls, data: dict):
        return cls()


class ChatModelPortObject(knext.PortObject):
    def __init__(self, spec: ChatModelPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self):
        return b""

    @classmethod
    def deserialize(cls, spec, data: dict):
        return cls(spec)

    def create_model(self, ctx):
        raise NotImplementedError()


chat_model_port_type = knext.port_type(
    "Chat Model Port", ChatModelPortObject, ChatModelPortObjectSpec
)


class EmbeddingsPortObjectSpec(knext.PortObjectSpec):
    def serialize(self):
        return {}

    @classmethod
    def deserialize(cls, data: dict):
        return cls()


class EmbeddingsPortObject(knext.PortObject):
    def __init__(self, spec: EmbeddingsPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self):
        return b""

    @classmethod
    def deserialize(cls, spec, data: dict):
        return cls(spec)

    def create_model(self, ctx):
        raise NotImplementedError()


embeddings_model_port_type = knext.port_type(
    "Embeddings Port", EmbeddingsPortObject, EmbeddingsPortObjectSpec
)


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="OpenAI API Key",
        description="""
        Credentials parameter for accessing the OpenAI API key
        """,
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


@knext.parameter_group(label="Model Settings")
class GeneralSettings:

    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0 and 1. 
        Higher values like 0.8 will make the output more random, 
        while lower values like 0.2 will make it more focused and deterministic.
        
        It is generally recommend altering this or top_p but not both at once.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True,
    )

    top_p = knext.DoubleParameter(
        label="top_p",
        description="""
        An alternative to sampling with temperature, 
        where the model considers the results of the tokens (words) 
        with top_p probability mass. So 0.1 means only the tokens 
        comprising the top 10% probability mass are considered.

        It is generally recommend altering this or top_p but not both at once.
        """,
        default_value=0.1,
        min_value=0.01,
        max_value=1.0,
        is_advanced=True,
    )

    max_tokens = knext.IntParameter(
        label="Max tokens",
        description="""
        The maximum number of tokens to generate in the completion.

        The token count of your prompt plus 
        max_tokens cannot exceed the model's context length.
        """,
        default_value=512,
        min_value=1,
    )

    n = knext.IntParameter(
        label="n",
        description="""
        How many chat completion choices to generate for each input message.
        This parameter generates many completions and
        can quickly consume your token quota. 
        """,
        default_value=1,
        min_value=1,
        is_advanced=True,
    )


# TODO: Add configuration dialog to enable templates e.g. https://python.langchain.com/docs/modules/model_io/models/llms/integrations/openai
# TODO: Add configuration dialog to more general options to configure how LLM is prompted
# TODO: Write better text
@knext.node("LLM Prompter", knext.NodeType.PREDICTOR, "", model_category)
@knext.input_port("LLM Port", "A large language model.", llm_port_type)
@knext.input_table("Prompt Table", "A table containing a string column with prompts.")
@knext.output_table(
    "Result Table", "A table containing prompts and their respective answer."
)
class LLMPrompter:
    """
    Prompt a given Large Language Model.

    This node takes a string column of prompts and prompts the
    provided Large Language Model with each of the prompts.
    """

    prompt_column = knext.ColumnParameter(
        "Prompt column",
        "Column that contains prompts for the LLM.",
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
            raise knext.InvalidParametersError(
                """
                The number of nominal columns are 0. Expected at least 
                one nominal column for prompts.
                """
            )

        if not self.prompt_column:
            raise ValueError("No column selected")

        # TODO: Append the column to the given table instead of creating a new one
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
        llm = llm_port.create_model(ctx)

        prompts = input_table.to_pandas()

        answers = [llm(prompt) for prompt in prompts[self.prompt_column]]

        prompts["Prompt Result"] = answers

        return knext.Table.from_pandas(prompts)
