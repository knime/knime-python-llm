# Nodes:
# - Gemini Embedding Model Connector

import knime.extension as knext

from ._utils import (
    google_category,
    gemini_icon,
)
from ._port_types import (
    # inputs:
    GenericGeminiConnectionPortObject,
    generic_gemini_connection_port_type,
    GenericGeminiConnectionPortObjectSpec,
    # outputs:
    gemini_embedding_model_port_type,
    GeminiEmbeddingModelPortObjectSpec,
    GeminiEmbeddingModelPortObject,
)


def _list_gemini_embedding_models(
    ctx: knext.DialogCreationContext,
):
    input_specs = ctx.get_input_specs()
    generic_connection_spec: GenericGeminiConnectionPortObjectSpec = input_specs[0]

    # in case the config dialog is open without any inputs
    # TODO: we return a list with an empty string, because returning an empty
    # list causes the UI of the dialogue setting to be stuck in the "Loading" state.
    # If/when that FE behaviour changes, we should adjust this accordingly.
    if generic_connection_spec is None:
        return [""]

    return generic_connection_spec.get_embedding_model_list(ctx)


@knext.node(
    "Gemini Embedding Model Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=gemini_icon,
    category=google_category,
    keywords=[
        "Google AI",
        "Vertex AI",
        "Gemini",
        "GenAI",
        "Embeddings",
        "Embedding model",
    ],
)
@knext.input_port(
    "Gemini Connection",
    "An authenticated connection to either Vertex AI or Google AI Studio.",
    port_type=generic_gemini_connection_port_type,
)
@knext.output_port(
    "Gemini Embedding Model",
    "A Gemini embedding model.",
    port_type=gemini_embedding_model_port_type,
)
class GeminiEmbeddingModelConnector:
    """Connects to embedding models available through either Vertex AI or Google AI Studio.

    This node allows selecting a Google-published embedding model using an authenticated connection obtained
    either from the **Vertex AI Connector** node, or from the **Google AI Studio Authenticator** node.
    """

    model_name = knext.StringParameter(
        "Model",
        """Select the embedding model to use.
        The list of available models is fetched using the provided Gemini connection.

        If connection with the API cannot be established, the list is populated with known embedding models
        appropriate for the connection type.
        """,
        choices=_list_gemini_embedding_models,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        generic_gemini_connection_spec: GenericGeminiConnectionPortObjectSpec,
    ) -> GeminiEmbeddingModelPortObjectSpec:
        return self._create_output_spec(generic_gemini_connection_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        generic_gemini_connection: GenericGeminiConnectionPortObject,
    ) -> GeminiEmbeddingModelPortObject:
        spec = self._create_output_spec(generic_gemini_connection.spec)

        return GeminiEmbeddingModelPortObject(spec=spec)

    def _create_output_spec(
        self, connection_spec: GenericGeminiConnectionPortObjectSpec
    ):
        if not self.model_name:
            raise knext.InvalidParametersError(
                "No model selected. Ensure that you selected a model from the list of available models."
            )

        return GeminiEmbeddingModelPortObjectSpec(
            connection_spec=connection_spec,
            model_name=self.model_name,
        )
