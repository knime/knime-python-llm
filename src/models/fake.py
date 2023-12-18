# KNIME / own imports
import knime.extension as knext
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    model_category,
    util,
)

# Langchain imports
from langchain.llms.base import LLM
from langchain.chat_models.base import SimpleChatModel
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema import BaseMessage

# Other imports
from typing import Any, List, Optional, Mapping, Dict
from pydantic import BaseModel
import pickle

fake_icon = "icons/fake.png"
fake_category = knext.category(
    path=model_category,
    level_id="fake",
    name="Fake",
    description="",
    icon=fake_icon,
)

# == Settings ==


class MissingValueHandlingOptions(knext.EnumParameterOptions):
    DefaultAnswer = (
        "Default response",
        "Return a predefined response.",
    )
    Fail = (
        "Fail",
        "Fail if no matching query is found.",
    )


@knext.parameter_group(label="Fake Configuration")
class FakeGeneralSettings:
    fake_prompt_col = knext.ColumnParameter(
        "Prompt column",
        "Column containing the fake prompts.",
        port_index=0,
        column_filter=util.create_type_filer(knext.string()),
    )

    fake_response_col = knext.ColumnParameter(
        "Responses",
        "Column containing the fake responses for each prompt.",
        port_index=0,
        column_filter=util.create_type_filer(knext.string()),
    )

    delay = knext.IntParameter(
        "Answer delay",
        "Delays the time the fake LLM will take to respond.",
        0,
        min_value=0,
        is_advanced=True,
    )


@knext.parameter_group(label="Prompt mismatch")
class MismatchSettings:
    prompt_mismatch = knext.EnumParameter(
        "Handle prompt mismatch",
        """Specify whether the LLM should provide a default response or fail when a given prompt cannot 
        be found in the prompt column.""",
        default_value=lambda v: MissingValueHandlingOptions.DefaultAnswer.name,
        enum=MissingValueHandlingOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    default_response = knext.StringParameter(
        "Default response",
        "The fake LLM's response when a query is not found in the prompt column.",
        "Could not find an answer to the query.",
    ).rule(
        knext.OneOf(prompt_mismatch, [MissingValueHandlingOptions.DefaultAnswer.name]),
        knext.Effect.SHOW,
    )


@knext.parameter_group(label="Embeddings Model Configuration")
class FakeEmbeddingsSettings:
    document_column = knext.ColumnParameter(
        "Document column",
        "Column containing the fake documents as strings.",
        port_index=0,
        column_filter=util.create_type_filer(knext.string()),
    )

    vector_column = knext.ColumnParameter(
        "Vector column",
        """Column containing the fake vectors. The column is expected to be a list of doubles. You may create the list 
        using the 
        [Table Creator](https://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.node.io.tablecreator.TableCreator2NodeFactory) and
        the [Create Collection Column](https://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.collection.list.create2.CollectionCreate2NodeFactoryhttps://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.collection.list.create2.CollectionCreate2NodeFactory) 
        nodes.""",
        port_index=0,
        column_filter=util.create_type_filer(knext.ListType(knext.double())),
    )

    delay = knext.IntParameter(
        "Answer delay",
        "Delays the time the fake LLM will take to respond.",
        0,
        min_value=0,
        is_advanced=True,
    )


# == Fake Implementations ==


def _warn_if_same_columns(ctx, f_prompt_col: str, response_col: str) -> None:
    if f_prompt_col == response_col:
        ctx.set_warning("Query and response column are set to be the same column.")


def generate_response(
    response_dict: dict[str, str],
    default_response: str,
    prompt: str,
    missing_value_strategy: str,
):
    response = response_dict.get(prompt)

    if not response:
        if missing_value_strategy == MissingValueHandlingOptions.Fail.name:
            raise knext.InvalidParametersError(
                f"Could not find matching response for prompt: '{prompt}'. Please make sure, that the prompt exactly matches one from the given prompt column."
            )
        else:
            return default_response

    return response


class FakeDictLLM(LLM):
    """Self implemented Fake LLM wrapper for testing purposes."""

    response_dict: dict[str, str]
    default_response: str
    missing_value_strategy: str
    sleep: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-dict"

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
        )

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class FakeChatModel(SimpleChatModel):
    """Fake ChatModel for testing purposes."""

    response_dict: Dict[str, str]
    default_response: str
    missing_value_strategy: str
    sleep: Optional[int] = 0

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = messages[len(messages) - 1].content
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class FakeEmbeddings(Embeddings, BaseModel):
    embeddings_dict: dict[str, list[float]]

    def embed_documents(self, _: any) -> List[float]:
        return list(self.embeddings_dict.values())

    def embed_query(self, text: str) -> List[float]:
        if text is None:
            raise ValueError("None values are not supported.")
        elif not text.strip():
            raise ValueError("Empty documents are not supported.")

        try:
            return self.embeddings_dict[text]
        except KeyError:
            raise KeyError(f"Could not find document '{text}' in dictionary.")


# == Port Objects ==


class FakeLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(self, sleep: float, missing_value_strategy: str) -> None:
        super().__init__()
        self._sleep = sleep
        self._missing_value_strategy = missing_value_strategy

    @property
    def sleep(self) -> float:
        return self._sleep

    @property
    def missing_value_strategy(self) -> str:
        return self._missing_value_strategy

    def serialize(self) -> dict:
        return {
            "sleep": self._sleep,
            "missing_value_strategy": self._missing_value_strategy,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return FakeLLMPortObjectSpec(data["sleep"], data["missing_value_strategy"])


class FakeLLMPortObject(LLMPortObject):
    def __init__(
        self,
        spec: FakeLLMPortObjectSpec,
        responses: dict[str, str],
        default_response: str,
    ) -> None:
        super().__init__(spec)
        self._responses = responses
        self._default_response = default_response

    @property
    def spec(self) -> FakeLLMPortObjectSpec:
        return super().spec

    @property
    def responses(self):
        return self._responses

    @property
    def default_response(self):
        return self._default_response

    def serialize(self):
        return pickle.dumps((self._responses, self._default_response))

    @classmethod
    def deserialize(cls, spec, data):
        (responses, default_response) = pickle.loads(data)
        return cls(spec, responses, default_response)

    def create_model(self, ctx) -> FakeDictLLM:
        return FakeDictLLM(
            response_dict=self.responses,
            default_response=self.default_response,
            missing_value_strategy=self.spec.missing_value_strategy,
            sleep=self.spec.sleep,
        )


fake_llm_port_type = knext.port_type(
    "Fake LLM", FakeLLMPortObject, FakeLLMPortObjectSpec
)


class FakeChatPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(self, sleep: float, missing_value_strategy: str) -> None:
        super().__init__()
        self._sleep = sleep
        self._missing_value_strategy = missing_value_strategy

    @property
    def sleep(self) -> int:
        return self._sleep

    @property
    def missing_value_strategy(self) -> str:
        return self._missing_value_strategy

    def serialize(self) -> dict:
        return {
            "sleep": self._sleep,
            "missing_value_strategy": self._missing_value_strategy,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return FakeLLMPortObjectSpec(data["sleep"], data["missing_value_strategy"])


class FakeChatModelPortObject(ChatModelPortObject):
    def __init__(
        self,
        spec: FakeLLMPortObjectSpec,
        responses: dict[str, str],
        default_response: str,
    ) -> None:
        super().__init__(spec)
        self._responses = responses
        self._default_response = default_response

    @property
    def spec(self) -> FakeChatPortObjectSpec:
        return super().spec

    @property
    def responses(self):
        return self._responses

    @property
    def default_response(self):
        return self._default_response

    def serialize(self):
        return pickle.dumps((self._responses, self._default_response))

    @classmethod
    def deserialize(cls, spec, data):
        (responses, default_response) = pickle.loads(data)
        return cls(spec, responses, default_response)

    def create_model(self, ctx: knext.ExecutionContext) -> FakeChatModel:
        return FakeChatModel(
            response_dict=self.responses,
            default_response=self.default_response,
            missing_value_strategy=self.spec.missing_value_strategy,
            sleep=self.spec.sleep,
        )


fake_chat_model_port_type = knext.port_type(
    "Fake Chat Model", FakeChatModelPortObject, FakeChatPortObjectSpec
)


class FakeEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def deserialize(cls, data: dict):
        return FakeEmbeddingsPortObjectSpec()


class FakeEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(
        self,
        spec: FakeEmbeddingsPortObjectSpec,
        embeddings_dict: dict[str, list[float]],
    ):
        super().__init__(spec)
        self._embeddings_dict = embeddings_dict

    @property
    def embeddings_dict(self):
        return self._embeddings_dict

    def serialize(self):
        return pickle.dumps(self._embeddings_dict)

    @classmethod
    def deserialize(cls, spec, data: dict):
        embeddings_dict = pickle.loads(data)
        return FakeEmbeddingsPortObject(FakeEmbeddingsPortObjectSpec(), embeddings_dict)

    def create_model(self, ctx):
        return FakeEmbeddings(embeddings_dict=self.embeddings_dict)


fake_embeddings_port_type = knext.port_type(
    "Fake Embeddings", FakeEmbeddingsPortObject, FakeEmbeddingsPortObjectSpec
)


def _configure_model_generation_columns(
    table_spec: knext.Schema,
    fake_prompt_col: knext.Column,
    fake_response_col: knext.Column,
    response_includes_vectors: bool = None,
):
    if fake_prompt_col and fake_response_col:
        util.check_column(
            table_spec,
            fake_prompt_col,
            knext.string(),
            "fake prompt",
        )

        if response_includes_vectors:
            util.check_column(
                table_spec,
                fake_response_col,
                knext.ListType(knext.double()),
                "fake embeddings",
            )
        else:
            util.check_column(
                table_spec,
                fake_response_col,
                knext.string(),
                "fake response",
            )

        return fake_prompt_col, fake_response_col

    else:
        if response_includes_vectors:
            query_col = util.pick_default_column(table_spec, knext.string())
            vector_col = util.pick_default_column(
                table_spec, knext.ListType(knext.double())
            )
            return query_col, vector_col

        default_columns = util.pick_default_columns(table_spec, knext.string(), 2)
        return default_columns[0], default_columns[1]


# == Nodes ==


def to_dictionary(table, columns: list[str]):
    df = table.to_pandas()

    for col in columns:
        is_missing = df[col].isnull().any()

        if is_missing:
            raise knext.InvalidParametersError(
                f"Missing value found in column '{col}'. Please make sure, that the query and response column do not have missing values."
            )

    response_dict = dict(
        map(
            lambda i, j: (i, j),
            df[columns[0]],
            df[columns[1]],
        )
    )

    return response_dict


@knext.node(
    "Fake LLM Connector",
    knext.NodeType.SOURCE,
    fake_icon,
    category=fake_category,
)
@knext.input_table(
    "Prompt and response table",
    "A table with columns for fake prompts and their corresponding responses.",
)
@knext.output_port(
    "Fake LLM",
    "Configured Fake LLM",
    fake_llm_port_type,
)
class FakeLLMConnector:
    """
    Creates a fake Large Language Model.

    This node creates a fake Large Language Model (LLM) implementation for testing
    purposes without the need for heavy computing power. Provide a column with expected
    prompts and their respective answers in a second column. When the 'Fake Model'
    is prompted with a matching prompt, it will return the provided answer.
    """

    settings = FakeGeneralSettings()
    mismatch = MismatchSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> FakeLLMPortObjectSpec:
        fake_prompt_col, response_col = _configure_model_generation_columns(
            table_spec, self.settings.fake_prompt_col, self.settings.fake_response_col
        )

        _warn_if_same_columns(ctx, fake_prompt_col, response_col)

        self.settings.fake_prompt_col = fake_prompt_col
        self.settings.fake_response_col = response_col

        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> FakeLLMPortObject:
        _warn_if_same_columns(
            ctx, self.settings.fake_prompt_col, self.settings.fake_response_col
        )

        response_dict = to_dictionary(
            input_table,
            [self.settings.fake_prompt_col, self.settings.fake_response_col],
        )

        return FakeLLMPortObject(
            self.create_spec(),
            response_dict,
            self.mismatch.default_response,
        )

    def create_spec(self) -> FakeLLMPortObjectSpec:
        return FakeLLMPortObjectSpec(self.settings.delay, self.mismatch.prompt_mismatch)


@knext.node(
    "Fake Chat Model Connector",
    knext.NodeType.SOURCE,
    fake_icon,
    category=fake_category,
)
@knext.input_table(
    "Prompt and response table",
    "A table with columns for fake prompts and their corresponding responses.",
)
@knext.output_port(
    "Fake Chat Model",
    "Configured Fake Chat Model",
    fake_chat_model_port_type,
)
class FakeChatConnector:
    """
    Creates a fake Chat Model.

    This node creates a fake Chat Model implementation for testing purposes without the
    need for heavy computing power. Provide a column with prompts and their respective
    responses in a second column. When the 'Fake Chat Model' is prompted,
    it will return the provided answer as an AI message.
    """

    settings = FakeGeneralSettings()
    mismatch = MismatchSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> FakeChatPortObjectSpec:
        fake_prompt_col, response_col = _configure_model_generation_columns(
            table_spec, self.settings.fake_prompt_col, self.settings.fake_response_col
        )

        _warn_if_same_columns(ctx, fake_prompt_col, response_col)

        self.settings.fake_prompt_col = fake_prompt_col
        self.settings.fake_response_col = response_col

        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> FakeChatModelPortObject:
        df = input_table.to_pandas()

        _warn_if_same_columns(
            ctx, self.settings.fake_prompt_col, self.settings.fake_response_col
        )

        response_dict = dict(
            map(
                lambda i, j: (i, j),
                df[self.settings.fake_prompt_col],
                df[self.settings.fake_response_col],
            )
        )

        return FakeChatModelPortObject(
            self.create_spec(),
            response_dict,
            self.mismatch.default_response,
        )

    def create_spec(self) -> FakeChatPortObjectSpec:
        return FakeChatPortObjectSpec(
            self.settings.delay, self.mismatch.prompt_mismatch
        )


@knext.node(
    "Fake Embeddings Connector",
    knext.NodeType.SOURCE,
    fake_icon,
    category=fake_category,
)
@knext.input_table(
    "Document and vector table",
    "A table containing a column with documents and one with the respective vectors.",
)
@knext.output_port(
    "Fake Embeddings",
    "Configured Fake Embeddings Model",
    fake_embeddings_port_type,
)
class FakeEmbeddingsConnector:
    """
    Creates a fake Embeddings Model.

    This node creates a fake Embeddings Model implementation for testing purposes without
    the need for heavy computing power. Provide a set of documents and queries along with their
    corresponding vectors for the following nodes, e.g., Vector Store Creators and Vector Store Retriever,
    to use.

    With this node you simulate exactly with which vectors documents will be stored and retrieved.
    """

    settings = FakeEmbeddingsSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> FakeEmbeddingsPortObjectSpec:
        fake_document_col, fake_vector_col = _configure_model_generation_columns(
            table_spec,
            self.settings.document_column,
            self.settings.vector_column,
            response_includes_vectors=True,
        )

        _warn_if_same_columns(ctx, fake_document_col, fake_vector_col)

        self.settings.document_column = fake_document_col
        self.settings.vector_column = fake_vector_col

        return FakeEmbeddingsPortObjectSpec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> FakeEmbeddingsPortObject:
        df = input_table.to_pandas()

        response_dict = dict(
            map(
                lambda i, j: (i, j.tolist()),
                df[self.settings.document_column],
                df[self.settings.vector_column],
            )
        )

        return FakeEmbeddingsPortObject(
            FakeEmbeddingsPortObjectSpec(),
            response_dict,
        )
