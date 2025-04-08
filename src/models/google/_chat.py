# Nodes:
# - Gemini Chat Model Connector

import knime.extension as knext

from ._utils import (
    google_category,
    gemini_icon,
)
from ._port_types import (
    GenericGeminiConnectionPortObject,
    generic_gemini_connection_port_type,
    GenericGeminiConnectionPortObjectSpec,
    gemini_chat_model_port_type,
    GeminiChatModelPortObjectSpec,
    GeminiChatModelPortObject,
)


def _list_gemini_chat_models(
    ctx: knext.DialogCreationContext,
):
    input_specs = ctx.get_input_specs()
    generic_connection_spec: GenericGeminiConnectionPortObjectSpec = input_specs[0]

    # in case the config dialog is open without any inputs
    if generic_connection_spec is None:
        return [""]

    return [model.name for model in generic_connection_spec.get_chat_model_list(ctx)]


@knext.node(
    "Gemini Chat Model Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=gemini_icon,
    category=google_category,
    keywords=["Google AI", "Vertex AI", "Gemini", "GenAI", "Chat model"],
)
@knext.input_port(
    "Gemini Connection",
    "An authenticated connection to either Vertex AI or Google AI Studio.",
    port_type=generic_gemini_connection_port_type,
)
@knext.output_port(
    "Gemini Chat Model",
    "A Gemini chat model.",
    port_type=gemini_chat_model_port_type,
)
class GeminiChatModelConnector:
    """Connects to Gemini chat models available through either Vertex AI or Google AI Studio.

    This node allows selecting a Gemini chat model using an authenticated connection obtained
    either from the **Vertex AI Connector** node, or from the **Google AI Studio Authenticator** node.
    """

    model_name = knext.StringParameter(
        "Model",
        """Select the Gemini chat model to use.
        The list of available models is fetched using the provided Gemini connection.

        If connection with the API cannot be established, the list is populated with known Gemini chat models
        appropriate for the connection type.
        """,
        choices=_list_gemini_chat_models,
    )

    max_output_tokens = knext.IntParameter(
        "Maximum response tokens",
        """Specify the number of tokens to constrain the model's responses to.

        A token is equivalent to about 4 characters for Gemini models. 100 tokens are about 60-80 English words.
        """,
        min_value=0,
        default_value=4096,
        is_advanced=True,
    )

    temperature = knext.DoubleParameter(
        "Temperature",
        """Specify the temperature for the model's responses.

        Temperature controls the degree of randomness in token selection.
        Lower temperatures are good for prompts that require a more deterministic or less open-ended response,
        while higher temperatures can lead to more diverse or creative results.
        A temperature of 0 is deterministic, meaning that the highest probability response is always selected.
        """,
        min_value=0.0,
        max_value=1.0,
        default_value=0.7,
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        generic_gemini_connection_spec: GenericGeminiConnectionPortObjectSpec,
    ) -> GeminiChatModelPortObjectSpec:
        return self._create_output_spec(generic_gemini_connection_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        generic_gemini_connection: GenericGeminiConnectionPortObject,
    ) -> GeminiChatModelPortObject:
        spec = self._create_output_spec(generic_gemini_connection.spec)

        return GeminiChatModelPortObject(spec=spec)

    def _create_output_spec(
        self, connection_spec: GenericGeminiConnectionPortObjectSpec
    ):
        if not self.model_name:
            raise knext.InvalidParametersError(
                "No model selected. Ensure that you selected a model from the list of available models."
            )

        return GeminiChatModelPortObjectSpec(
            auth_spec=connection_spec,
            model_name=self.model_name,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
