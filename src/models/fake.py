# KNIME / own imports
import knime.extension as knext
from .base import (
    AIPortObjectSpec,
    LLMPortObjectSpec,
    LLMPortObject,
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    model_category,
    util,
)

# Langchain imports
from langchain.llms.base import LLM
from langchain.chat_models.base import SimpleChatModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema import BaseMessage

# Other imports
from typing import Any, List, Optional, Mapping, Dict

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
        "Return a predefined default response.",
    )
    Fail = (
        "Fail",
        "Fail if no matching query is found.",
    )


@knext.parameter_group(label="Fake Configuration")
class FakeGeneralSettings:
    query_column = knext.ColumnParameter(
        "Queries",
        "Column containing the fake queries.",
        port_index=0,
        column_filter=util.create_type_filer(knext.string()),
    )

    response_column = knext.ColumnParameter(
        "Responses",
        "Column containing the fake respnses.",
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


@knext.parameter_group(label="Key mismatch")
class MismatchSettings:
    handle_key_mismatch = knext.EnumParameter(
        "Handle key mismatch",
        """Define whether the LLM should 'answer' with a predefined answer or fail when the prompt can not be found in the provided queries.""",
        default_value=lambda v: MissingValueHandlingOptions.DefaultAnswer.name,
        enum=MissingValueHandlingOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    default_response = knext.StringParameter(
        "Default response",
        "The response of the fake LLM when no querie matches",
        "Could not find an answer to the querie.",
    ).rule(
        knext.OneOf(
            handle_key_mismatch, [MissingValueHandlingOptions.DefaultAnswer.name]
        ),
        knext.Effect.SHOW,
    )


# == Fake Implementations ==


def handle_missing_value(option: str, prompt: str, response: str):
    if option == MissingValueHandlingOptions.Fail.name:
        raise knext.InvalidParametersError(
            f"Could not find matching response for prompt: '{prompt}'. Please make sure, that the prompt exactly matches the provided answers."
        )
    return response


class FakeDictLLM(LLM):
    """Self implemented Fake LLM wrapper for testing purposes."""

    response_dict: dict[str, str]
    default_response: str
    option: str
    sleep: int = 0

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-dict"

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return self.response_dict.get(
            prompt, handle_missing_value(self.option, prompt, self.default_response)
        )

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        return self.response_dict.get(
            prompt, handle_missing_value(self.option, prompt, self.default_response)
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class FakeChatModel(SimpleChatModel):
    """Fake ChatModel for testing purposes."""

    response_dict: Dict[str, str]
    default_response: str
    option: str
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
        """First try to lookup in response_dict, else return 'foo' or 'bar'."""
        prompt = messages[len(messages) - 1].content
        return self.response_dict.get(
            prompt, handle_missing_value(self.option, prompt, self.default_response)
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


# == Port Objects ==


class FakeModelPortObjectSpec(AIPortObjectSpec):
    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: dict):
        return FakeModelPortObjectSpec()


class FakeLLMPortObjectSpec(FakeModelPortObjectSpec, LLMPortObjectSpec):
    def __init__(self, sleep: float, missing_value_option: str) -> None:
        super().__init__()
        self._sleep = sleep
        self._missing_value_option = missing_value_option

    @property
    def sleep(self) -> float:
        return self._sleep

    @property
    def missing_value_option(self) -> str:
        return self._missing_value_option

    def serialize(self) -> dict:
        return {
            "sleep": self._sleep,
            "missing_value_option": self._missing_value_option,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return FakeLLMPortObjectSpec(data["sleep"], data["missing_value_option"])


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
            option=self.spec.missing_value_option,
            sleep=self.spec.sleep,
        )


fake_llm_port_type = knext.port_type(
    "Fake LLM", FakeLLMPortObject, FakeLLMPortObjectSpec
)


class FakeChatPortObjectSpec(FakeLLMPortObjectSpec, ChatModelPortObjectSpec):
    def __init__(self, sleep: float, missing_value_option: str) -> None:
        super().__init__(sleep, missing_value_option)


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
            option=self.spec.missing_value_option,
            sleep=self.spec.sleep,
        )


fake_chat_model_port_type = knext.port_type(
    "Fake Chat Model", FakeChatModelPortObject, FakeChatPortObjectSpec
)


# == Nodes ==


@knext.node(
    "Fake LLM Connector",
    knext.NodeType.SOURCE,
    fake_icon,
    category=fake_category,
)
@knext.input_table(
    "Prompt and answer table",
    "A table containing a column with expected prompts and one with the matching responses.",
)
@knext.output_port(
    "Fake LLM",
    "Configured Fake LLM",
    fake_llm_port_type,
)
class FakeLLMConnector:
    """
    Creates a fake Large Language Model.

    This node creates a fake Large Language Model (LLM) implementation for testing purposes without
    the need to use heavy compute power. Provide a column with expected prompts and their respective
    answer in a second column. When the 'Fake Model' will be prompted with a matching prompt, it will
    return the provided answer.
    """

    settings = FakeGeneralSettings()
    mismatch = MismatchSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> FakeLLMPortObjectSpec:
        if self.settings.query_column:
            util.check_column(
                table_spec,
                self.settings.query_column,
                knext.string(),
                "String column with prompts",
            )
        else:
            self.settings.query_column = util.pick_default_column(
                table_spec, knext.string()
            )

        if self.settings.response_column:
            util.check_column(
                table_spec,
                self.settings.response_column,
                knext.string(),
                "String column with responses",
            )
        else:
            self.settings.query_column = util.pick_default_column(
                table_spec, knext.string()
            )

        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> FakeLLMPortObject:
        df = input_table.to_pandas()

        response_dict = dict(
            map(
                lambda i, j: (i, j),
                df[self.settings.query_column],
                df[self.settings.response_column],
            )
        )

        return FakeLLMPortObject(
            self.create_spec,
            response_dict,
            self.mismatch.default_response,
        )

    def create_spec(self) -> FakeLLMPortObjectSpec:
        return FakeLLMPortObjectSpec(
            self.settings.delay, self.mismatch.handle_key_mismatch
        )


@knext.node(
    "Fake Chat Model Connector",
    knext.NodeType.SOURCE,
    fake_icon,
    category=fake_category,
)
@knext.input_table(
    "Prompt and answer table",
    "A table containing a column with expected prompts and one with the matching responses.",
)
@knext.output_port(
    "Fake Chat Model",
    "Configured Fake Chat Model",
    fake_chat_model_port_type,
)
class FakeChatConnector:
    """
    Creates a fake Chat Model.

    This node creates a fake Chat Model implementation for testing purposes without
    the need to use heavy compute power. Provide a column with expected prompts and their respective
    answer in a second column. When the 'Fake Chat Model' will be prompted with a matching prompt, it will
    return the provided answer as an AI message.
    """

    settings = FakeGeneralSettings()
    mismatch = MismatchSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> FakeChatPortObjectSpec:
        if self.settings.query_column:
            util.check_column(
                table_spec,
                self.settings.query_column,
                knext.string(),
                "String column with prompts",
            )
        else:
            self.settings.query_column = util.pick_default_column(
                table_spec, knext.string()
            )

        if self.settings.response_column:
            util.check_column(
                table_spec,
                self.settings.response_column,
                knext.string(),
                "String column with responses",
            )
        else:
            self.settings.response_column = util.pick_default_column(
                table_spec, knext.string()
            )

        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> FakeChatModelPortObject:
        df = input_table.to_pandas()

        response_dict = dict(
            map(
                lambda i, j: (i, j),
                df[self.settings.query_column],
                df[self.settings.response_column],
            )
        )

        return FakeChatModelPortObject(
            self.create_spec(),
            response_dict,
            self.mismatch.default_response,
        )

    def create_spec(self) -> FakeChatPortObjectSpec:
        return FakeChatPortObjectSpec(
            self.settings.delay, self.mismatch.handle_key_mismatch
        )
