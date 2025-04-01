# Nodes:
# - Vertex AI Connector
# - Google Ai Studio Authenticator

import knime.extension as knext

from models.base import CredentialsSettings

from ._utils import (
    google_category,
    vertex_ai_icon,
    google_ai_studio_icon,
    DEFAULT_VERTEX_AI_LOCATION,
    _vertex_ai_location_choices_provider,
)
from ._port_types import (
    vertex_ai_connection_port_type,
    VertexAiConnectionPortObjectSpec,
    VertexAiConnectionPortObject,
    google_ai_studio_authentication_port_type,
    GoogleAiStudioAuthenticationPortObjectSpec,
    GoogleAiStudioAuthenticationPortObject,
)


@knext.node(
    "Vertex AI Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=vertex_ai_icon,
    category=google_category,
    keywords=["Google", "Vertex AI", "Gemini", "GenAI", "Chat model"],
)
@knext.input_port(
    "Google Credentials",
    "Credentials for a Google Cloud service account with Vertex AI permissions.",
    port_type=knext.PortType.CREDENTIAL,
)
@knext.output_port(
    "Vertex AI Connection",
    "Authenticated connection to Vertex AI.",
    vertex_ai_connection_port_type,
)
class VertexAiConnector:
    """Establishes a connection with Vertex AI via the provided Google Cloud authentication.

    This node allows you to connect with a Vertex AI project in Google Cloud. The Vertex AI
    connection produced by this node can then be used to select chat and embedding models
    from the Gemini family using the **Gemini Chat Model Connector** and **Gemini Embedding Model Connector**
    nodes.

    ---

    **Authentication**: The required Google Cloud authentication can be acquired by configuring the **Google Authenticator** node
    with:

    - the "Service Account" authentication type,
    - a custom scope set to `https://www.googleapis.com/auth/cloud-platform`,
    - the corresponding JSON or P12 key.

    Refer to the Google Cloud documentation for details on how to [create a service account](https://cloud.google.com/iam/docs/service-accounts-create#creating)
    and a corresponding [service account key](https://cloud.google.com/iam/docs/keys-create-delete#creating).
    """

    project_id = knext.StringParameter(
        "Project ID",
        "ID of the Google Cloud project where Vertex AI is enabled.",
    )

    location = knext.StringParameter(
        "Location",
        """
        Google Cloud location for accessing Vertex AI models.

        Refer to the [Google Cloud documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations) to view the list of locations
        available for Generative AI on Vertex AI.
        """,
        default_value=DEFAULT_VERTEX_AI_LOCATION,
        choices=_vertex_ai_location_choices_provider,
    )

    custom_base_api_url = knext.StringParameter(
        "Custom base API URL",
        "If not specified, the default `https://<location>-aiplatform.googleapis.com/v1` base API URL will be used.",
        is_advanced=True,
    )

    should_validate_connection = knext.BoolParameter(
        "Validate connection",
        "Whether to validate the connection to Vertex AI by making an API call during execution.",
        True,
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        google_cloud_credentials_spec: knext.CredentialPortObjectSpec,
    ) -> VertexAiConnectionPortObjectSpec:
        return self.create_spec(google_cloud_credentials_spec, ctx)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        google_cloud_credentials: knext.CredentialPortObjectSpec,
    ) -> VertexAiConnectionPortObject:
        spec = self.create_spec(google_cloud_credentials.spec, ctx)

        if self.should_validate_connection:
            spec.validate_connection()

        return VertexAiConnectionPortObject(spec=spec)

    def create_spec(
        self,
        credential_spec: knext.CredentialPortObjectSpec,
        ctx,
    ) -> VertexAiConnectionPortObjectSpec:
        if not self.project_id:
            raise knext.InvalidParametersError("Project ID cannot be empty.")
        if not self.location:
            raise knext.InvalidParametersError("Location cannot be empty.")
        if self.custom_base_api_url and not self.custom_base_api_url.startswith("http"):
            raise knext.InvalidParametersError(
                "Custom base API URL must be a valid URL."
            )

        return VertexAiConnectionPortObjectSpec(
            google_credentials_spec=credential_spec,
            project_id=self.project_id,
            location=self.location,
            custom_base_api_url=self.custom_base_api_url,
        )


@knext.node(
    name="Google AI Studio Authenticator",
    node_type=knext.NodeType.SOURCE,
    icon_path=google_ai_studio_icon,
    category=google_category,
    keywords=["Google", "AI Studio", "Gemini", "GenAI", "Chat model"],
)
@knext.output_port(
    "Google AI Studio Authentication",
    "Authentication for the Google AI Studio",
    google_ai_studio_authentication_port_type,
)
class GoogleAiStudioAuthenticator:
    """Authenticates with Google AI Studio via API key.

    This node authenticates with Google AI Studio via the provided API key. The Google AI Studio
    authentication produced by this node can then be used to select chat and embedding models
    from the Gemini family using the **Gemini Chat Model Connector** and **Gemini Embedding Model Connector**
    nodes.

    Please refer to the Google AI Studio documentation for details on how to [create an API key](https://ai.google.dev/gemini-api/docs).
    """

    credentials_settings = CredentialsSettings(
        label="Google AI Studio API key",
        description="The credentials containing the Google AI Studio API key in its *password* field (the *username* is ignored).",
    )

    should_validate_api_key = knext.BoolParameter(
        "Validate API key",
        "Whether to validate the provided API key by making a request to list the available models.",
        True,
        is_advanced=True,
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> GoogleAiStudioAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        spec = self.create_spec()
        spec.validate_context(ctx)

        return spec

    def execute(
        self, ctx: knext.ExecutionContext
    ) -> GoogleAiStudioAuthenticationPortObject:
        spec = self.create_spec()

        if self.should_validate_api_key:
            spec.validate_api_key(ctx)

        return GoogleAiStudioAuthenticationPortObject(spec)

    def create_spec(self) -> GoogleAiStudioAuthenticationPortObjectSpec:
        return GoogleAiStudioAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param
        )
