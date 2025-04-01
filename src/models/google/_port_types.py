# Port types:
# - Vertex AI Connection (currently only Gemini API, then other APIs later)

import knime.extension as knext

from typing import Dict


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
