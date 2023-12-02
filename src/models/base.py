# KNIME / own imports
import knime.extension as knext
import util
import pandas as pd
from base import AIPortObjectSpec
from typing import Any, List, Optional, Sequence


# Langchain imports
from langchain.schema import HumanMessage, SystemMessage, ChatMessage, AIMessage
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain.base_language import BaseLanguageModel
from langchain.llms.base import BaseLLM
from langchain.schema.messages import (
    BaseMessage,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)

model_category = knext.category(
    path=util.main_category,
    level_id="models",
    name="Models",
    description="",
    icon="icons/ml.png",
)


@knext.parameter_group(label="Model Parameters")
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
        label="Top-p sampling",
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
    def __init__(self, port_index=1) -> None:
        self.role_column = knext.ColumnParameter(
            "Message role",
            "Column that specifies the sender role of the messages. The usual values are Human and AI.",
            port_index=port_index,
            column_filter=util.create_type_filer(knext.string()),
        )

        self.content_column = knext.ColumnParameter(
            "Messages",
            "Column containing the message contents that have been sent to and from the model.",
            port_index=port_index,
            column_filter=util.create_type_filer(knext.string()),
        )

    def configure(self, input_table_spec: knext.Schema):
        if self.content_column:
            util.check_column(
                input_table_spec,
                self.content_column,
                knext.string(),
                "content",
            )
        else:
            self.content_column = util.pick_default_column(
                input_table_spec, knext.string()
            )

        if self.role_column:
            util.check_column(
                input_table_spec,
                self.role_column,
                knext.string(),
                "role",
            )
        else:
            spec_without_content = knext.Schema.from_columns(
                [c for c in input_table_spec if c.name != self.content_column]
            )
            try:
                self.role_column = util.pick_default_column(
                    spec_without_content, knext.string()
                )
            except:
                raise knext.InvalidParametersError(
                    "The conversation table must contain at least two string columns. "
                    "One for the message roles and one for the message contents."
                )

        if self.role_column == self.content_column:
            raise knext.InvalidParametersError(
                "The role and content column can not be the same."
            )

    _role_to_message_type = {
        "ai": AIMessage,
        "assistant": AIMessage,
        "user": HumanMessage,
        "human": HumanMessage,
    }

    def _create_message(self, role: str, content: str):
        if not role:
            raise ValueError("No role provided.")
        message_type = self._role_to_message_type.get(role.lower(), None)
        if message_type:
            return message_type(content=content)
        else:
            # fallback to be used if the user provides other roles
            # which may or may not work in subsequent calls
            return ChatMessage(content=content, role=role)

    def create_messages(self, data_frame: pd.DataFrame):
        role_column = data_frame[self.role_column]
        content_column = data_frame[self.content_column]
        return [
            self._create_message(role, content)
            for role, content in zip(role_column.values, content_column.values)
        ]


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

    def create_model(self, ctx: knext.ExecutionContext) -> BaseLanguageModel:
        raise NotImplementedError()


llm_port_type = knext.port_type("LLM", LLMPortObject, LLMPortObjectSpec)


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

    def create_model(self, ctx: knext.ExecutionContext) -> BaseChatModel:
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


@knext.node("LLM Prompter", knext.NodeType.PREDICTOR, "icons/ml.png", model_category)
@knext.input_port(
    "LLM or chat model", "A large language model or chat model.", llm_port_type
)
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
        "Column containing prompts for the LLM.",
        port_index=1,
        column_filter=util.create_type_filer(knext.string()),
    )

    response_column_name = knext.StringParameter(
        "Response column name",
        "Name for the column holding the LLM's responses.",
        default_value="Response",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm_spec: LLMPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        if self.prompt_column:
            util.check_column(
                input_table_spec, self.prompt_column, knext.string(), "prompt"
            )
        else:
            self.prompt_column = util.pick_default_column(
                input_table_spec, knext.string()
            )

        llm_spec.validate_context(ctx)

        if not self.response_column_name:
            raise knext.InvalidParametersError(
                "The response column name must not be empty."
            )

        return input_table_spec.append(
            knext.Column(ktype=knext.string(), name=self.response_column_name)
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm_port: LLMPortObject,
        input_table: knext.Table,
    ):
        llm = llm_port.create_model(ctx)
        num_rows = input_table.num_rows

        output_table: knext.BatchOutputTable = knext.BatchOutputTable.create()
        i = 0
        for batch in input_table.batches():
            data_frame = batch.to_pandas()
            responses = []
            for prompt in data_frame[self.prompt_column]:
                util.check_canceled(ctx)
                responses.append(llm.predict(prompt))
                ctx.set_progress(i / num_rows)
                i += 1

            data_frame[self.response_column_name] = responses
            output_table.append(data_frame)

        return output_table


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

    system_message = knext.MultilineStringParameter(
        "System Message",
        """
        The first message given to the model describing how it should behave.

        Example: You are a helpful assistant that has to answer questions truthfully, and
        if you do not know an answer to a question, you should state that.
        """,
        default_value="",
    )

    chat_message = knext.MultilineStringParameter(
        "Message",
        "The (next) message that will be added to the conversation",
        default_value="",
    )

    conversation_settings = ChatConversationSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chat_model_spec: ChatModelPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        self.conversation_settings.configure(input_table_spec)

        chat_model_spec.validate_context(ctx)
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), self.conversation_settings.role_column),
                knext.Column(knext.string(), self.conversation_settings.content_column),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model: ChatModelPortObject,
        input_table: knext.Table,
    ):
        data_frame = input_table[
            [
                self.conversation_settings.role_column,
                self.conversation_settings.content_column,
            ]
        ].to_pandas()

        conversation_messages = []
        if self.system_message:
            conversation_messages.append(SystemMessage(content=self.system_message))
        conversation_messages += self.conversation_settings.create_messages(data_frame)
        if self.chat_message:
            human_message = HumanMessage(content=self.chat_message)
            conversation_messages.append(human_message)
            data_frame.loc[f"Row{len(data_frame)}"] = [
                human_message.type,
                human_message.content,
            ]

        chat = chat_model.create_model(ctx)

        answer = chat(conversation_messages)

        data_frame.loc[f"Row{len(data_frame)}"] = [answer.type, answer.content]

        return knext.Table.from_pandas(data_frame)


def _string_col_filter(column: knext.Column):
    return column.ktype == knext.string()


@knext.node("Text Embedder", knext.NodeType.PREDICTOR, util.ai_icon, model_category)
@knext.input_port(
    "Embeddings Model",
    "Used to embed the texts from the input table into numerical vectors.",
    embeddings_model_port_type,
)
@knext.input_table("Input Table", "Input table containing a text column to embed.")
@knext.output_table(
    "Output Table", "The input table with the appended embeddings column."
)
class TextEmbedder:
    """
    Embeds text in a string column using an embedding model.

    This node applies the provided embeddings model to create embeddings for the texts contained in a string column of the input table.
    At its core, a text embedding is a dense vector of floating point values capturing the semantic meaning of the text.
    Thus these embeddings are often used to find semantically similar documents e.g. in vector stores.
    How exactly the embeddings are derived depends on the used embeddings model but typically the embeddings are the internal representations
    used by deep language models e.g. GPTs.
    If this node fails to execute and gives an unhelpful error message such as 'Execute failed: Error while sending a command.', then refer to the
    description of the node that provided the embeddings model.
    """

    text_column = knext.ColumnParameter(
        "Text column",
        "The string column containing the texts to embed.",
        port_index=1,
        column_filter=_string_col_filter,
    )

    embeddings_column_name = knext.StringParameter(
        "Embeddings column name",
        "Name for output column that will hold the embeddings.",
        "Embeddings",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
        table_spec: knext.Schema,
    ) -> knext.Schema:
        if self.text_column is None:
            self.text_column = util.pick_default_column(table_spec, knext.string())
        else:
            util.check_column(
                table_spec, self.text_column, knext.string(), "text column"
            )

        embeddings_spec.validate_context(ctx)
        return table_spec.append(self._create_output_column())

    def _create_output_column(self) -> knext.Column:
        return knext.Column(knext.list_(knext.double()), self.embeddings_column_name)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_obj: EmbeddingsPortObject,
        table: knext.Table,
    ) -> knext.Table:
        embeddings_model = embeddings_obj.create_model(ctx)
        output_table = knext.BatchOutputTable.create()
        num_rows = table.num_rows
        i = 0
        for batch in table.batches():
            util.check_canceled(ctx)
            data_frame = batch.to_pandas()
            texts = data_frame[self.text_column].tolist()
            embeddings = embeddings_model.embed_documents(texts)
            data_frame[self.embeddings_column_name] = embeddings
            output_table.append(knext.Table.from_pandas(data_frame))
            i += batch.num_rows
            ctx.set_progress(i / num_rows)
        return output_table


class LLMChatModelAdapter(BaseChatModel):
    """
    This class adapts LLMs as chat models, allowing LLMs that have been fined tuned for
    chat applications to be used as chat models with the Chat Model Prompter.
    """

    llm: BaseLLM
    system_prompt_template: str
    prompt_template: str

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prediction = self._predict_messages(
            messages=messages,
            system_prompt_template=self.system_prompt_template,
            prompt_template=self.prompt_template,
            stop=stop,
            **kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=prediction)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prediction = await self._apredict_messages(
            messages=messages,
            system_prompt_template=self.system_prompt_template,
            prompt_template=self.prompt_template,
            stop=stop,
            **kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=prediction)])

    def _apply_prompt_templates(
        self,
        messages: Sequence[BaseMessage],
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> str:
        string_messages = []

        message_templates = {
            HumanMessage: prompt_template,
            AIMessage: "%1",
            SystemMessage: system_prompt_template,
        }

        for m in messages:
            if type(m) not in message_templates:
                raise ValueError(f"Got unsupported message type: {m}")

            template = message_templates[type(m)]

            # if the template doesn't include the predefined pattern "%1",
            # the template input will be ignored and only the entered message will be passed
            if "%1" in template:
                message = template.replace(
                    "%1",
                    m.content
                    if isinstance(m, (HumanMessage, SystemMessage))
                    else m.content,
                )
            else:
                message = m.content

            string_messages.append(message)

        return "\n".join(string_messages)

    def _predict_messages(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = self._apply_prompt_templates(
            messages=messages,
            system_prompt_template=system_prompt_template,
            prompt_template=prompt_template,
        )
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = self.llm(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    async def _apredict_messages(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = self._apply_prompt_templates(
            messages=messages,
            system_prompt_template=system_prompt_template,
            prompt_template=prompt_template,
        )
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = self.llm._call_async(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    def _llm_type(self) -> str:
        """Return type of llm."""
        return "LLMChatModelAdapter"
