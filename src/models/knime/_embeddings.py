from langchain_openai import OpenAIEmbeddings
import knime.extension as knext
import knime.api.schema as ks
from knime.extension import ExecutionContext
from ._base import (
    hub_connector_icon,
    knime_category,
    _create_authorization_headers,
    _extract_api_base,
    _list_models_in_dialog,
    _list_models,
)


from ..base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
)


class KnimeHubEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self,
        auth_spec: ks.HubAuthenticationPortObjectSpec,
        model_name: str,
    ) -> None:
        super().__init__()
        self._auth_spec = auth_spec
        self._model_name = model_name

    @property
    def auth_spec(self) -> ks.HubAuthenticationPortObjectSpec:
        return self._auth_spec

    @property
    def model_name(self) -> str:
        return self._model_name

    def serialize(self) -> dict:
        return {
            "auth": self.auth_spec.serialize(),
            "model_name": self.model_name,
        }

    @classmethod
    def deserialize(cls, data: dict, java_callback):
        return cls(
            ks.HubAuthenticationPortObjectSpec.deserialize(data["auth"], java_callback),
            data["model_name"],
        )


class KnimeHubEmbeddingsPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> KnimeHubEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx: ExecutionContext) -> OpenAIEmbeddings:
        auth_spec = self.spec.auth_spec
        return OpenAIEmbeddings(
            model=self.spec.model_name,
            default_headers=_create_authorization_headers(auth_spec),
            openai_api_base=_extract_api_base(auth_spec),
            openai_api_key="placeholder",
            # TODO do we need to switch off tiktoken here to be compatible with non-openai? (Also applies to Chat models)
        )


knime_embeddings_port_type = knext.port_type(
    "KNIME Hub Embeddings",
    KnimeHubEmbeddingsPortObject,
    KnimeHubEmbeddingsPortObjectSpec,
)


@knext.node(
    name="KNIME Hub Embeddings Connector (Labs)",
    node_type=knext.NodeType.SOURCE,
    icon_path=hub_connector_icon,
    category=knime_category,
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_port(
    name="KNIME Hub Embeddings",
    description="An embeddings model that connects to a KNIME hub to embed documents.",
    port_type=knime_embeddings_port_type,
)
class KnimeHubEmbeddingsConnector:
    """Connects to an embeddings model configured in the GenAI gateway of the connected KNIME Hub.

    Connects to an Embeddings Model configured in the GenAI gateway of the connected KNIME Hub using the authentication
    provided via the input port.
    """

    model_name = knext.StringParameter(
        "Model",
        "Select the model to use.",
        choices=_list_models_in_dialog("embedding"),
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> KnimeHubEmbeddingsPortObjectSpec:
        return self._create_spec(authentication)

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> KnimeHubEmbeddingsPortObject:
        available_models = _list_models(authentication.spec, "embedding")
        if self.model_name not in available_models:
            raise knext.InvalidParametersError(
                f"The selected model {self.model_name} is not served by the connected Hub."
            )
        return KnimeHubEmbeddingsPortObject(self._create_spec(authentication.spec))

    def _create_spec(
        self, authentication: ks.HubAuthenticationPortObjectSpec
    ) -> KnimeHubEmbeddingsPortObjectSpec:
        return KnimeHubEmbeddingsPortObjectSpec(
            authentication,
            self.model_name,
        )
