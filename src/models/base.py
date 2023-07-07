# KNIME / own imports
import knime.extension as knext
import util

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.embeddings.base import Embeddings
from base import AIPortObjectSpec

model_category = knext.category(
    path=util.main_category,
    level_id="models",
    name="Models",
    description="",
    icon="icons/ml.png",
)


@knext.parameter_group(label="Model Settings")
class GeneralSettings:
    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 100.0. 
        Higher values will make the output more random, 
        while lower values will make it more focused and deterministic.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=100.0,
        is_advanced=True,
    )

    top_p = knext.DoubleParameter(
        label="top_p",
        description="""
        An alternative to sampling with temperature, 
        where the model considers the results of the tokens (words) 
        with top_p probability mass. Hence, 0.1 means only the tokens 
        comprising the top 10% probability mass are considered.
        """,
        default_value=0.15,
        min_value=0.01,
        max_value=1.0,
        is_advanced=True,
    )


@knext.parameter_group(label="Conversation Settings")
class ChatConversationSettings:
    type_column = knext.ColumnParameter(
        "Type", "Column that specifies the sender type of the messages", port_index=1
    )

    message_column = knext.ColumnParameter(
        "Messages",
        "Column containing the messages that have been sent to and from the model.",
        port_index=1,
    )


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    def __init__(self, label, description):
        def _get_default_credentials(identifier):
            result = knext.DialogCreationContext.get_credential_names(identifier)
            if not result:
                # single choice with an empty string to still render the configuration dialogue if no credentials are provided
                return [""]
            return result

        self.credentials_param = knext.StringParameter(
            label=label,
            description=description,
            choices=lambda a: _get_default_credentials(a),
        )


class LLMPortObjectSpec(AIPortObjectSpec):
    """Most generic spec of LLMs. Used to define the most generic LLM PortType"""


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


class ChatModelPortObjectSpec(LLMPortObjectSpec):
    """Most generic chat model spec. Used to define the most generic chat model PortType."""


class ChatModelPortObject(LLMPortObject):
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


class EmbeddingsPortObjectSpec(AIPortObjectSpec):
    """Most generic embeddings model spec. Used to define the most generic embeddings model PortType."""

class EmbeddingsPortObject(knext.PortObject):
    def __init__(self, spec: EmbeddingsPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self):
        return b""

    @classmethod
    def deserialize(cls, spec, data: dict):
        return cls(spec)

    def create_model(self, ctx: knext.ExecutionContext) -> Embeddings:
        raise NotImplementedError()


embeddings_model_port_type = knext.port_type(
    "Embeddings Port", EmbeddingsPortObject, EmbeddingsPortObjectSpec
)


# TODO: Add configuration dialog to enable templates, e.g. https://python.langchain.com/docs/modules/model_io/models/llms/integrations/openai
@knext.node("LLM Prompter", knext.NodeType.PREDICTOR, "icons/ml.png", model_category)
@knext.input_port("LLM Port", "A large language model.", llm_port_type)
@knext.input_table("Prompt Table", "A table containing a string column with prompts.")
@knext.output_table(
    "Result Table", "A table containing prompts and their respective answer."
)
class LLMPrompter:
    """
    Prompts a Large Language Model.
    
    For each row in the input table, the LLM Prompter sends one prompt to the LLM and receives a corresponding response.
    Rows are treated independently i.e. the LLM can not *remember* the content of previous rows or how it responded to them.
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
            raise knext.InvalidParametersError("No column selected.")
        
        llm_spec.validate_context(ctx)

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

        answers = [llm.predict(prompt) for prompt in prompts[self.prompt_column]]

        prompts["Prompt Result"] = answers

        return knext.Table.from_pandas(prompts)


# TODO: Add configuration dialog to more general options to configure how LLM is prompted
# TODO: Write better text
@knext.node(
    "Chat Model Prompter", knext.NodeType.PREDICTOR, "icons/ml.png", model_category
)
@knext.input_port("Chat Model Port", "A chat model model.", chat_model_port_type)
@knext.input_table(
    "Conversation", "A table containing an empty or filled conversation."
)
@knext.output_table("Conversation", "A table containing the conversation history.")
class ChatModelPrompter:
    """
    Prompts a Chat Model.

    The Chat Model Prompter takes a statement (prompt) and the conversation
    history with human and AI messages. It then generates a response for the prompt with the knowledge of the previous
    conversation.

    If you want to reduce the amount of tokens being used, consider reducing
    the conversation table length to a reasonable (e.g. 5 conversation steps) length
    before feeding it into the Chat Model Prompter.

    """

    conversation_settings = ChatConversationSettings()

    system_message = knext.StringParameter(
        "System Message",
        """
        The first message given to the model describing how it should behave.

        Example: You are a helpful assistant that has to answer questions truthfully, and
        if you do not know an answer to a question, you should state that.
        """,
    )

    chat_message = knext.StringParameter(
        "Message",
        "The (next) message that will be added to the conversation",
        default_value="",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chat_model_spec: ChatModelPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        nominal_columns = [
            (c.name, c.ktype) for c in input_table_spec if util.is_nominal(c)
        ]

        if len(nominal_columns) < 2:
            raise knext.InvalidParametersError(
                """
                At least two nominal columns have to provided ('Type', 'Message').
                """
            )

        chat_model_spec.validate_context(ctx)
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Type"),
                knext.Column(knext.string(), "Message"),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        input_table: knext.Table,
    ):
        table = input_table.to_pandas()

        conversation_messages = []

        if len(table.index) == 0:
            if self.system_message:
                conversation_messages.append(SystemMessage(content=self.system_message))
                table.loc[f"Row{len(table)}"] = ["SystemMessage", self.system_message]

        else:
            for index, row in table.iterrows():
                match row[self.conversation_settings.type_column]:
                    case "AIMessage":
                        conversation_messages.append(
                            AIMessage(
                                content=row[self.conversation_settings.message_column]
                            )
                        )
                    case "HumanMessage":
                        conversation_messages.append(
                            HumanMessage(
                                content=row[self.conversation_settings.message_column]
                            )
                        )
                    case "SystemMessage":
                        conversation_messages.append(
                            SystemMessage(
                                content=row[self.conversation_settings.message_column]
                            )
                        )

        if self.chat_message:
            table.loc[f"Row{len(table)}"] = ["HumanMessage", self.chat_message]
            conversation_messages.append(HumanMessage(content=self.chat_message))

        chat = chat_model.create_model(ctx)

        answer = chat(conversation_messages)

        table.loc[f"Row{len(table)}"] = ["AIMessage", answer.content]

        return knext.Table.from_pandas(table)

def _string_col_filter(column: knext.Column):
    return column.ktype == knext.string()

@knext.node(
    "Text Embedder",
    knext.NodeType.PREDICTOR,
    util.ai_icon,
    model_category
)
@knext.input_port("Embeddings Model", "Used to embed the texts from the input table into numerical vectors.", embeddings_model_port_type)
@knext.input_table("Input Table", "Input table containing a text column to embed.")
@knext.output_table("Output Table", "The input table with the appended embeddings column.")
class TextEmbedder:
    """
    Embeds text in a string column using an embedding model.

    This node applies the provided embeddings model to create embeddings for the texts contained in a string column of the input table.
    At its core, a text embedding is a dense vector of floating point values capturing the semantic meaning of the text.
    Thus these embeddings are often used to find semantically similar documents e.g. in vector stores.
    How exactly the embeddings are derived depends on the used embeddings model but typically the embeddings are the internal representations
    used by deep language models e.g. GPTs.
    """

    text_column = knext.ColumnParameter("Text column", "The string column containing the texts to embed.", port_index=1, column_filter=_string_col_filter)

    embeddings_column_name = knext.StringParameter("Embeddings column name", "Name for output column that will hold the embeddings.", "embeddings")
    
    def configure(self, ctx: knext.ConfigurationContext, embeddings_spec: EmbeddingsPortObjectSpec, table_spec: knext.Schema) -> knext.Schema:
        if self.text_column is None:
            self.text_column = util.pick_default_column(table_spec, knext.string())
        else:
            util.check_column(table_spec, self.text_column, knext.string(), "text column")
        
        embeddings_spec.validate_context(ctx)
        return table_spec.append(self._create_output_column())

    def _create_output_column(self) -> knext.Column:
        return knext.Column(knext.list_(knext.double()), self.embeddings_column_name)

    def execute(self, ctx: knext.ExecutionContext, embeddings_obj: EmbeddingsPortObject, table: knext.Table) -> knext.Table:
        embeddings_model = embeddings_obj.create_model(ctx)
        output_table = knext.BatchOutputTable.create()
        for batch in table.batches():
            if ctx.is_canceled():
                raise RuntimeError("Execution was canceled.")
            data_frame = batch.to_pandas()
            texts = data_frame[self.text_column]
            embeddings = embeddings_model.embed_documents(texts)
            data_frame[self.embeddings_column_name] = embeddings
            output_table.append(knext.Table.from_pandas(data_frame))
        return output_table
