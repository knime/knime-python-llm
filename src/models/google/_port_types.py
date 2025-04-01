# Port types:
# - Vertex AI Connection (currently only Gemini API, then other APIs later)
# - Google AI Studio Authentication (Gemini API)

import knime.extension as knext

from typing import Dict, Literal

from base import AIPortObjectSpec
from ._utils import GEMINI_CHAT_MODELS_FALLBACK, GEMINI_EMBEDDING_MODELS_FALLBACK


# ---------------------------------------------------------------------
# Vertex AI Connection Port Type
# ---------------------------------------------------------------------
class VertexAiConnectionPortObjectSpec(knext.PortObjectSpec):
    def __init__(
        self,
        google_credentials_spec: knext.CredentialPortObjectSpec,
        project_id: str,
        location: str,
        custom_base_api_url: str,
    ):
        self._google_credentials_spec = google_credentials_spec
        self._project_id = project_id
        self._location = location
        self._custom_base_api_url = custom_base_api_url

    @property
    def google_credentials_spec(self) -> knext.CredentialPortObjectSpec:
        return self._google_credentials_spec

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def location(self) -> str:
        return self._location

    @property
    def custom_base_api_url(self) -> str:
        return self._custom_base_api_url

    def serialize(self):
        return {
            "google_credentials_spec": self._google_credentials_spec.serialize(),
            "project_id": self._project_id,
            "location": self._location,
            "custom_base_api_url": self._custom_base_api_url,
        }

    @classmethod
    def deserialize(cls, data: Dict):
        cred_spec = knext.CredentialPortObjectSpec.deserialize(
            data["google_credentials_spec"]
        )
        return cls(
            google_credentials_spec=cred_spec,
            project_id=data["project_id"],
            location=data["location"],
            custom_base_api_url=data["custom_base_api_url"],
        )

    def _construct_google_credentials_from_auth_spec(
        self,
        auth_spec: knext.CredentialPortObjectSpec,
    ):
        def get_refresh_handler(auth_spec: knext.CredentialPortObjectSpec):
            return lambda request, scopes: (
                auth_spec.auth_parameters,
                auth_spec.expires_after,
            )

        token = getattr(auth_spec, "auth_parameters", None)
        expiry = getattr(auth_spec, "expires_after", None)

        if token is None or expiry is None:
            raise knext.InvalidParametersError(
                "Google Cloud credentials are misconfigured. Are you authenticated using a service account?"
            )

        from google.oauth2.credentials import Credentials

        return Credentials(
            token=token,
            expiry=expiry,
            refresh_handler=get_refresh_handler(auth_spec),
        )

    def validate_connection(self):
        credentials = self._construct_google_credentials_from_auth_spec(
            self.google_credentials_spec
        )

        from google.auth.transport.requests import AuthorizedSession
        import requests

        authed_session = AuthorizedSession(credentials)

        base_url = (
            self._custom_base_api_url
            if self._custom_base_api_url
            else f"https://{self.location}-aiplatform.googleapis.com/v1"
        )
        url = f"{base_url}/projects/{self.project_id}/locations/{self.location}"

        try:
            response = authed_session.get(url)
            if response.status_code != 200:
                raise knext.InvalidParametersError(
                    f"Vertex AI connection validation failed with status {response.status_code}: {response.text}"
                )
        except requests.RequestException as e:
            raise knext.InvalidParametersError(
                f"Vertex AI connection validation failed due to a network error: {str(e)}"
            )


class VertexAiConnectionPortObject(knext.PortObject):
    def __init__(self, spec: VertexAiConnectionPortObjectSpec):
        super().__init__(spec=spec)

    @property
    def spec(self) -> VertexAiConnectionPortObjectSpec:
        return self._spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: VertexAiConnectionPortObjectSpec, storage: bytes):
        return cls(spec)


vertex_ai_connection_port_type = knext.port_type(
    "Vertex AI Connection",
    VertexAiConnectionPortObject,
    VertexAiConnectionPortObjectSpec,
)


# ---------------------------------------------------------------------
# Google AI Studio Authentication Port Type
# ---------------------------------------------------------------------
class GoogleAiStudioAuthenticationPortObjectSpec(AIPortObjectSpec):
    def __init__(self, credentials: str) -> None:
        super().__init__()
        self._credentials = credentials

    @property
    def credentials(self) -> str:
        return self._credentials

    def validate_context(self, ctx: knext.ConfigurationContext):
        if self.credentials not in ctx.get_credential_names():
            raise knext.InvalidParametersError(
                f"""The selected credentials '{self.credentials}' holding the Google AI Studio API key do not exist. 
                Make sure that you have selected the correct credentials and that they are still available."""
            )
        api_token = ctx.get_credentials(self.credentials)
        if not api_token.password:
            raise knext.InvalidParametersError(
                f"""The Google AI Studio API key '{self.credentials}' does not exist. Make sure that the node you are using to pass the credentials 
                (e.g. the Credentials Configuration node) is still passing the valid API key as a flow variable to the downstream nodes."""
            )

    def validate_api_key(self, ctx: knext.ExecutionContext):
        try:
            self._fetch_models_from_api(ctx, "chat")
        except Exception as e:
            raise RuntimeError(
                "Could not authenticate with the Google AI Studio API using the provided API key."
            ) from e

    def get_chat_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        try:
            return self._fetch_models_from_api(ctx, "chat")
        except Exception:
            return GEMINI_CHAT_MODELS_FALLBACK

    def get_embedding_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        try:
            return self._fetch_models_from_api(ctx, "embedding")
        except Exception:
            return GEMINI_EMBEDDING_MODELS_FALLBACK

    def _fetch_models_from_api(self, ctx, model_type: Literal["chat", "embedding"]):
        from google import genai

        key = ctx.get_credentials(self.credentials).password
        client = genai.Client(api_key=key)
        models = client.models.list()

        filtered_models = []
        model_action = "generateContent" if model_type == "chat" else "embedContent"
        for m in models:
            if model_action in m.supported_actions:
                filtered_models.append(m)

        if len(filtered_models) > 0:
            return filtered_models

        raise Exception(
            f"The list of available models includes no {model_type} models."
        )

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"])


class GoogleAiStudioAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: GoogleAiStudioAuthenticationPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> GoogleAiStudioAuthenticationPortObjectSpec:
        return super().spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(
        cls, spec: GoogleAiStudioAuthenticationPortObjectSpec, storage: bytes
    ):
        return cls(spec)


google_ai_studio_authentication_port_type = knext.port_type(
    "Google AI Studio Authentication",
    GoogleAiStudioAuthenticationPortObject,
    GoogleAiStudioAuthenticationPortObjectSpec,
)
