# TODO: Done Alex -- check?
# TODO: Node idea: Chat Prompter that gets a conversation table

# KNIME / own imports
import knime.extension as knext
import util
import pandas as pd

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# TODO: Add category description?
# TODO: Get someone to do new icons
model_category = knext.category(
    path=util.main_cat,
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
        Sampling temperature to use, between 0 and 1. 
        Higher values like 0.8 will make the output more random, 
        while lower values like 0.2 will make it more focused and deterministic.
        
        It is generally recommend altering this or top_p but not both at once.
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
        with top_p probability mass. So 0.1 means only the tokens 
        comprising the top 10% probability mass are considered.

        It is generally recommend altering this or top_p but not both at once.
        """,
        default_value=0.15,
        min_value=0.01,
        max_value=1.0,
        is_advanced=True,
    )


@knext.parameter_group(label="Conversation History Settings")
class ChatConversationSettings:
    type_column = knext.ColumnParameter(
        "Message Type", "Column that specifies the sender of the messages", port_index=1
    )

    message_column = knext.ColumnParameter(
        "Messages",
        "Column containing the messages that have been sent to and from the model",
        port_index=1,
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
    Prompt a Large Language Model.

    The LLM Prompter takes a statement (prompt) and creates a
    response (e.g. generates text) for that single prompt. It has
    no knowledge of the other prompts or any kind of conversation
    memory.

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


import logging

LOGGER = logging.getLogger(__name__)


# TODO: Add configuration dialog to more general options to configure how LLM is prompted
# TODO: Write better text
@knext.node("Chat Model Prompter", knext.NodeType.PREDICTOR, "", model_category)
@knext.input_port("Chat Model Port", "A chat model model.", chat_model_port_type)
@knext.input_table(
    "Conversation Table", "A table containing a empty or filled conversation."
)
@knext.output_table(
    "Conversation Table", "A table containing the conversation history."
)
class ChatModelPrompter:
    """
    Prompt a Chat Model.

    The Chat Model Prompter takes a statement (prompt) and the conversation
    history with user, ai and system messages. It then can generate a response
    (e.g. generates text) for the prompt with the knowledge of the previous
    conversation.

    If you want to reduce the amount of tokens being used, consider cutting
    the conversation table at a reasonable (e.g. 5 conversation steps) length
    before re-providing it to the Chat Model Prompter.

    """

    conversation_settings = ChatConversationSettings()

    system_message = knext.StringParameter(
        "Chat System Prefix",
        """
        The first message given to the model describing how it should behave.

        E.g. You are a helpfull assissant that needs answer questions truthfully and
        state if you do not know a answer to a question.
        """,
    )

    chat_message = knext.StringParameter(
        "Chat message", "The (next) message to send to the chat model", default_value=""
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
                The number of nominal columns have to be at least 2. ('Type', 'Message')
                """
            )

        # TODO: Append the column to the given table instead of creating a new one
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

        table.loc[f"Row{len(table)}"] = ["HumanMessage", self.chat_message]
        conversation_messages.append(HumanMessage(content=self.chat_message))

        chat = chat_model.create_model(ctx)

        answer = chat(conversation_messages)

        table.loc[f"Row{len(table)}"] = ["AIMessage", answer.content]

        return knext.Table.from_pandas(table)
