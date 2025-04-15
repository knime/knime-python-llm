import knime.extension as knext
from ..base import ChatModelPortObject, ChatModelPortObjectSpec, OutputFormatOptions
from ._base import anthropic_icon, anthropic_category
from ._auth import (
    AnthropicAuthenticationPortObjectSpec,
    AnthropicAuthenticationPortObject,
    anthropic_auth_port_type,
)
from ._util import latest_models


class AnthropicChatModelPortObjectSpec(ChatModelPortObjectSpec):
    """Spec of an Anthropic Chat Model"""

    def __init__(
        self,
        auth: AnthropicAuthenticationPortObjectSpec,
        model: str,
        temperature: float,
        max_tokens: int,
        n_requests=1,
    ):
        super().__init__(n_requests)
        self._auth = auth
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def auth(self) -> AnthropicAuthenticationPortObjectSpec:
        return self._auth

    def validate_context(self, ctx):
        self.auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "auth": self._auth.serialize(),
            "n_requests": self._n_requests,
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    @classmethod
    def deserialize(cls, data: dict):
        auth = AnthropicAuthenticationPortObjectSpec.deserialize(data["auth"])
        return cls(
            auth=auth,
            model=data["model"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
            n_requests=data.get("n_requests", 1),
        )


class AnthropicChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> AnthropicChatModelPortObjectSpec:
        return super().spec

    def create_model(
        self,
        ctx: knext.ExecutionContext,
        output_format: OutputFormatOptions = OutputFormatOptions.Text,
    ):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=ctx.get_credentials(self.spec.auth.credentials).password,
            base_url=self.spec.auth.base_url,
            model=self.spec.model,
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
        )


anthropic_chat_model_port_type = knext.port_type(
    "Anthropic Chat Model",
    AnthropicChatModelPortObject,
    AnthropicChatModelPortObjectSpec,
)


def _list_models(ctx: knext.ConfigurationContext):
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_model_list(ctx)
    return latest_models


@knext.node(
    name="Anthropic Chat Model Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=anthropic_icon,
    category=anthropic_category,
    keywords=["Anthropic", "GenAI", "Claude", "Sonnet", "Opus", "Haiku"],
)
@knext.input_port(
    "Anthropic Authentication",
    "The authentication for the Anthropic API.",
    anthropic_auth_port_type,
)
@knext.output_port(
    "Anthropic Chat Model",
    "The Anthropic chat model which can be used in the LLM Prompter and Chat Model Prompter.",
    anthropic_chat_model_port_type,
)
class AnthropicChatModelConnector:
    """Connects to a chat model provided by the Anthropic API.

    This node establishes a connection with an Anthropic Chat Model. After successfully authenticating
    using the **Anthropic Authenticator** node, you can select a chat model from a predefined list.

    **Note**: Data sent to the Anthropic API is not used for the training of models by default,
    but can be if the prompts are flagged for Trust & Safety violations. For more information, check the
    [Anthropic documentation](https://privacy.anthropic.com/en/articles/10023580-is-my-data-used-for-model-training).
    """

    model = knext.StringParameter(
        "Model",
        description="""The model to use. The available models are fetched from the Anthropic API if possible.

        Models with the suffix -latest are the latest snapshots of the respective models. For more consistent
        behavior, specific snapshots should be used (e.g. claude-3-7-sonnet-20250219).
        """,
        default_value="claude-3-7-sonnet-latest",
        choices=_list_models,
    )

    temperature = knext.DoubleParameter(
        "Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 1.0.

        Higher values will lead to less deterministic but more creative answers.
        """,
        default_value=1,
        min_value=0,
        max_value=1,
    )

    max_tokens = knext.IntParameter(
        "Max Tokens",
        description="The maximum number of tokens to generate in the response.",
        default_value=1024,
        min_value=1,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: AnthropicAuthenticationPortObjectSpec,
    ) -> AnthropicChatModelPortObjectSpec:
        auth.validate_context(ctx)
        return self.create_spec(auth)

    def create_spec(
        self, auth: AnthropicAuthenticationPortObjectSpec
    ) -> AnthropicChatModelPortObjectSpec:
        return AnthropicChatModelPortObjectSpec(
            auth=auth,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def execute(
        self, ctx: knext.ExecutionContext, auth: AnthropicAuthenticationPortObject
    ) -> AnthropicChatModelPortObject:
        return AnthropicChatModelPortObject(self.create_spec(auth.spec))
