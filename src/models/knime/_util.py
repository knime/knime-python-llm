import knime.extension as knext
import knime.api.schema as ks
from ._base import (
    hub_connector_icon,
    knime_category,
    list_models_with_descriptions,
    validate_auth_spec,
)
import pandas as pd


@knext.node(
    name="KNIME Hub AI Model Lister",
    node_type=knext.NodeType.SOURCE,
    icon_path=hub_connector_icon,
    category=knime_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "GenAI Gateway", "LLM", "RAG"],
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_table(
    name="KNIME Hub AI Model list",
    description="The list of models, including their name, type and description.",
)
class KnimeHubAIModelLister:
    """
    Lists available models in the GenAI Gateway of the connected KNIME Hub using the
    authentication provided via the input port.

    Use this node to retrieve the available models with their name, type and description.
    """

    chat_models = knext.BoolParameter(
        "Chat Models",
        "List available chat models.",
        False,
    )

    embedding_models = knext.BoolParameter(
        "Embedding Models",
        "List available embedding models.",
        False,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> knext.Schema:
        # raises exception if the hub authenticator has not been executed
        validate_auth_spec(authentication)
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Name"),
                knext.Column(knext.string(), "Type"),
                knext.Column(knext.string(), "Description"),
            ]
        )

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> knext.Table:

        available_models = []

        if self.chat_models:
            available_models.extend(
                list_models_with_descriptions(authentication.spec, "chat")
            )

        if self.embedding_models:
            available_models.extend(
                list_models_with_descriptions(authentication.spec, "embedding")
            )

        # TODO sync with Yannick
        if not available_models:
            available_models = [("", "", "")]
            ctx.set_warning("No models available.")

        models_df = pd.DataFrame(
            available_models, columns=["Name", "Type", "Description"]
        )

        return knext.Table.from_pandas(models_df)
