# Port types:
# - Vertex AI Connection (currently only Gemini API, then other APIs later)
# - Google AI Studio Authentication (Gemini API)
# - Gemini Chat Model

import knime.extension as knext

from typing import Dict, List, Literal
import logging

from base import AIPortObjectSpec
from models.base import ChatModelPortObject, ChatModelPortObjectSpec
from ._utils import (
    VERTEX_AI_GEMINI_CHAT_MODELS_FALLBACK,
    VERTEX_AI_GEMINI_EMBEDDING_MODELS_FALLBACK,
    GOOGLE_AI_STUDIO_GEMINI_CHAT_MODELS_FALLBACK,
    GOOGLE_AI_STUDIO_GEMINI_EMBEDDING_MODELS_FALLBACK,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Generic Gemini Connection Port Type
# ---------------------------------------------------------------------
class GenericGeminiConnectionPortObjectSpec(AIPortObjectSpec):
    def __init__(self, connection_type: Literal["vertex_ai", "google_ai_studio"]):
        self._connection_type = connection_type

    def get_chat_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        raise NotImplementedError(
            "Subclasses of the generic Gemini connection must implement this method."
        )

    def get_embedding_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        raise NotImplementedError(
            "Subclasses of the generic Gemini connection must implement this method."
        )


class GenericGeminiConnectionPortObject(knext.PortObject):
    """Base class for the Gemini Connection input port of Gemini connector nodes."""


generic_gemini_connection_port_type = knext.port_type(
    "Gemini Connection",
    GenericGeminiConnectionPortObject,
    GenericGeminiConnectionPortObjectSpec,
)


# ---------------------------------------------------------------------
# Vertex AI Connection Port Type
# ---------------------------------------------------------------------
class VertexAiConnectionPortObjectSpec(GenericGeminiConnectionPortObjectSpec):
    def __init__(
        self,
        google_credentials_spec: knext.CredentialPortObjectSpec,
        project_id: str,
        location: str,
        custom_base_api_url: str,
    ):
        super().__init__(connection_type="vertex_ai")
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
    def base_api_url(self) -> str:
        if self._custom_base_api_url:
            return self._custom_base_api_url

        return f"{self.location}-aiplatform.googleapis.com"

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

    def validate_connection(self, ctx: knext.ExecutionContext):
        """
        Called during execution.
        """
        try:
            models = self._fetch_models_from_api("chat")
            if not models:
                ctx.set_warning(
                    "Connection with Vertex AI established, but no models are available"
                    f" for project ID '{self.project_id}' under location '{self.location}'."
                    " A predefined lit of known chat or embedding models will be used downstream."
                    " Try selecting a different location, e.g. `us-central1`."
                )
        except Exception as e:
            raise RuntimeError(
                "Could not establish connection with Vertex AI."
                f" Ensure the project ID exists and the authenticated account has appropriate permissions. Error: {e}"
            )

    def get_chat_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        models = self._fetch_models_from_api("chat")
        if not models:
            LOGGER.info(
                f"No chat models available for specified location '{self.location}'. Using a predefined list of known chat models."
            )
            return VERTEX_AI_GEMINI_CHAT_MODELS_FALLBACK

        return models

    def get_embedding_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        models = self._fetch_models_from_api("embedding")
        if not models:
            LOGGER.info(
                f"No embedding models available for specified location '{self.location}'. Using a predefined list of known embedding models."
            )
            return VERTEX_AI_GEMINI_EMBEDDING_MODELS_FALLBACK

        return models

    def _fetch_models_from_api(
        self,
        model_type: Literal["chat", "embedding"],
    ) -> List[str]:
        """
        Always returns either:
        - a list of matched models from the API endpoint
        - an empty list if no models were matched

        Callers are expected to handle the empty case.
        """

        def matches_model_type(model):
            model_name = model.name.lower()

            def is_chat_model(model_name):
                return any(s in model_name for s in ["gemini", "gemma"])

            def is_embedding_model(model_name):
                return any(s in model_name for s in ["text", "embed"])

            # deprecated models no longer work but still get listed
            def is_deprecated_model(model_name):
                return any(s in model_name for s in ["vision"])

            if model_type == "chat":
                return (
                    is_chat_model(model_name)
                    and not is_embedding_model(model_name)
                    and not is_deprecated_model(model_name)
                )

            if model_type == "embedding":
                return is_embedding_model(model_name)

        credentials = self._construct_google_credentials_from_auth_spec(
            self.google_credentials_spec
        )

        # this gets foundation models published by Google.
        # TODO: add support for custom models deployed in the Model Garden
        from google.genai import Client

        http_options = {"base_url": f"https://{self.base_api_url}"}
        genai_client = Client(
            vertexai=True,
            credentials=credentials,
            project=self.project_id,
            location=self.location,
            http_options=http_options,
        )

        # this is a paged iterator that automatically fetches the next page
        paged_model_list = genai_client.models.list(config={"query_base": True})

        return [m for m in paged_model_list if matches_model_type(m)]


class VertexAiConnectionPortObject(GenericGeminiConnectionPortObject):
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
class GoogleAiStudioAuthenticationPortObjectSpec(GenericGeminiConnectionPortObjectSpec):
    def __init__(self, credentials: str) -> None:
        super().__init__(connection_type="google_ai_studio")
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
        """
        Called during execution.
        """
        try:
            models = self._fetch_models_from_api(ctx, "chat")
            if not models:
                ctx.set_warning(
                    "Connection with Google AI Studio established, but no models are available."
                    " Double-check your Google AI Studio configuration."
                )
        except Exception as e:
            raise RuntimeError(
                f"Could not authenticate with Google AI Studio. Error: {e}"
            )

    def get_chat_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        models = self._fetch_models_from_api(dialog_creation_context, "chat")
        if not models:
            LOGGER.info(
                "No chat models available for the authenticated Google AI Studio user. Using a predefined list of known chat models."
            )
            return GOOGLE_AI_STUDIO_GEMINI_CHAT_MODELS_FALLBACK

        return models

    def get_embedding_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        models = self._fetch_models_from_api(dialog_creation_context, "embedding")
        if not models:
            LOGGER.info(
                "No embedding models available for the authenticated Google AI Studio user. Using a predefined list of known embedding models."
            )
            return GOOGLE_AI_STUDIO_GEMINI_EMBEDDING_MODELS_FALLBACK

        return models

    def _fetch_models_from_api(
        self,
        ctx: knext.ExecutionContext | knext.DialogCreationContext,
        model_type: Literal["chat", "embedding"],
    ) -> List[str]:
        """
        Always returns either:
        - a list of matched models from the API endpoint
        - an empty list if no models were matched

        Callers are expected to handle the empty case.
        """

        def matches_model_type(model):
            if model_type == "chat":
                return "generateContent" in model.supported_actions

            if model_type == "embedding":
                return "embedContent" in model.supported_actions

        from google.genai import Client

        key = ctx.get_credentials(self.credentials).password
        client = Client(api_key=key)

        # this is a paged iterator that automatically fetches the next page
        paged_model_list = client.models.list()

        return [m for m in paged_model_list if matches_model_type(m)]

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"])


class GoogleAiStudioAuthenticationPortObject(GenericGeminiConnectionPortObject):
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


# ---------------------------------------------------------------------
# Gemini Chat Model Port Type
# ---------------------------------------------------------------------
class GeminiChatModelPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(
        self,
        auth_spec: GenericGeminiConnectionPortObjectSpec,
        model_name: str,
        max_output_tokens: int,
        temperature: float,
        n_requests=1,
    ):
        super().__init__(n_requests)
        self._auth_spec = auth_spec
        self._connection_type = auth_spec._connection_type
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    @property
    def auth_spec(self) -> GenericGeminiConnectionPortObjectSpec:
        return self._auth_spec

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    @property
    def temperature(self) -> int:
        return self._temperature

    def serialize(self):
        return {
            "connection_type": self._connection_type,
            "auth_spec_data": self._auth_spec.serialize(),
            "model_name": self._model_name,
            "max_output_tokens": self._max_output_tokens,
            "temperature": self._temperature,
            "n_requests": self._n_requests,
        }

    @classmethod
    def deserialize(cls, data: Dict):
        connection_type = data["connection_type"]
        auth_spec_data = data["auth_spec_data"]

        if connection_type == "vertex_ai":
            auth_spec = VertexAiConnectionPortObjectSpec.deserialize(auth_spec_data)
        elif connection_type == "google_ai_studio":
            auth_spec = GoogleAiStudioAuthenticationPortObjectSpec.deserialize(
                auth_spec_data
            )
        else:
            raise ValueError(
                f"Unknown auth spec type during deserialization: {connection_type}"
            )

        return cls(
            auth_spec=auth_spec,  # _connection_type gets inferred from the auth spec object
            model_name=data["model_name"],
            max_output_tokens=data["max_output_tokens"],
            temperature=data["temperature"],
            n_requests=data.get("n_requests", 1),
        )


class GeminiChatModelPortObject(ChatModelPortObject):
    def __init__(self, spec: GeminiChatModelPortObjectSpec):
        super().__init__(spec=spec)

    @property
    def spec(self) -> GeminiChatModelPortObjectSpec:
        return self._spec

    def create_model(self, ctx):
        auth_spec = self.spec.auth_spec

        if isinstance(auth_spec, VertexAiConnectionPortObjectSpec):
            from langchain_google_vertexai import ChatVertexAI

            google_credentials = auth_spec._construct_google_credentials_from_auth_spec(
                auth_spec.google_credentials_spec
            )

            return ChatVertexAI(
                model_name=self.spec.model_name,
                project=auth_spec.project_id,
                location=auth_spec.location,
                credentials=google_credentials,
                max_tokens=self.spec.max_output_tokens,
                temperature=self.spec.temperature,
                max_retries=2,  # default is 6, instead we just try twice before failing
                base_url=auth_spec.base_api_url,
            )

        if isinstance(auth_spec, GoogleAiStudioAuthenticationPortObjectSpec):
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = ctx.get_credentials(auth_spec.credentials).password
            if not api_key:
                raise ValueError(
                    f"API Key not found in credentials '{auth_spec.credentials}'"
                )

            return ChatGoogleGenerativeAI(
                model=self.spec.model_name,
                google_api_key=api_key,
                max_tokens=self.spec.max_output_tokens,
                temperature=self.spec.temperature,
            )


gemini_chat_model_port_type = knext.port_type(
    "Gemini Chat Model",
    GeminiChatModelPortObject,
    GeminiChatModelPortObjectSpec,
)
