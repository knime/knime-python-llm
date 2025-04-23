# Port types:
# - Vertex AI Connection (currently only Gemini API, then other APIs later)
# - Google AI Studio Authentication (Gemini API)
# - Gemini Chat Model

import knime.extension as knext

from typing import Dict, List, Literal, Optional
import logging

from base import AIPortObjectSpec
from models.base import (
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    OutputFormatOptions,
)
from ._utils import (
    KNOWN_DEPRECATED_MODELS,
    VERTEX_AI_GEMINI_CHAT_MODELS_FALLBACK,
    VERTEX_AI_GEMINI_EMBEDDING_MODELS_FALLBACK,
    GOOGLE_AI_STUDIO_GEMINI_CHAT_MODELS_FALLBACK,
    GOOGLE_AI_STUDIO_GEMINI_EMBEDDING_MODELS_FALLBACK,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Generic Gemini Connection Port Type
# region Generic Authentication
# ---------------------------------------------------------------------
class GenericGeminiConnectionPortObjectSpec(AIPortObjectSpec):
    def __init__(self, connection_type: Literal["vertex_ai", "google_ai_studio"]):
        self._connection_type = connection_type

    def get_chat_model_list(self, dialog_creation_context: knext.DialogCreationContext):
        raise NotImplementedError(
            "Subclasses of the generic Gemini connection must implement this method."
        )

    def get_embedding_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ):
        raise NotImplementedError(
            "Subclasses of the generic Gemini connection must implement this method."
        )

    def _clean_and_sort_models(self, models):
        """
        Given a list of Model objects fetched from the API, we ensure that:
        - their names don't have the "/models" prefix
        - Gemini models are listed first
        - Gemma models are listed second
        - other models are listed third
        """

        def get_group(s):
            if s.startswith("gemini"):
                return 0

            if s.startswith("gemma"):
                return 1

            return 2

        unique_model_names = set([m.name.split("/")[-1] for m in models])

        # we want e.g. gemini-2.5 to appear before gemini-2.0, thus reverse sorting
        sorted_names = sorted(unique_model_names, reverse=True)
        grouped_names = sorted(sorted_names, key=get_group)

        return grouped_names


class GenericGeminiConnectionPortObject(knext.PortObject):
    """Base class for the Gemini Connection input port of Gemini connector nodes."""


generic_gemini_connection_port_type = knext.port_type(
    "Gemini Connection",
    GenericGeminiConnectionPortObject,
    GenericGeminiConnectionPortObjectSpec,
)


# ---------------------------------------------------------------------
# Vertex AI Connection Port Type
# region Vertex Authentication
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
    def custom_base_api_url(self) -> Optional[str]:
        if self._custom_base_api_url:
            return self._custom_base_api_url

        return None

    @property
    def base_api_url(self) -> str:
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
        try:
            genai_client = self._create_genai_client_for_location(self.location)
            genai_client.models.list()
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

        return self._clean_and_sort_models(models)

    def get_embedding_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        models = self._fetch_models_from_api("embedding")
        if not models:
            LOGGER.info(
                f"No embedding models available for specified location '{self.location}'. Using a predefined list of known embedding models."
            )
            return VERTEX_AI_GEMINI_EMBEDDING_MODELS_FALLBACK

        return self._clean_and_sort_models(models)

    def _fetch_models_from_api(
        self,
        model_type: Literal["chat", "embedding"],
    ):
        """
        Always returns either:
        - a list of matched models from the API endpoint
        - an empty list if no models were matched

        Callers are expected to handle the empty case.
        """

        def matches_model_type(model):
            model_name = model.name.split("/")[-1]

            # deprecated models no longer work but still get listed
            is_deprecated_model = model_name in KNOWN_DEPRECATED_MODELS
            is_chat_model = any(s in model_name for s in ["gemini", "gemma"])
            is_embedding_model = any(s in model_name for s in ["text", "embed"])

            if is_deprecated_model:
                return False

            if model_type == "chat":
                return is_chat_model and not is_embedding_model

            if model_type == "embedding":
                return is_embedding_model

            return False

        # 1. we use us-central1 to get the list of all available models
        base_genai_client = self._create_genai_client_for_location("us-central1")

        # 2. filter out deprecated and irrelevant models
        paged_model_list = base_genai_client.models.list(config={"query_base": True})
        filtered_models = [m for m in paged_model_list if matches_model_type(m)]

        # 3. filter out models unavailable to the authenticated user at the specified location
        regional_genai_client = (
            self._create_genai_client_for_location(self.location)
            if self.location != "us-central1"
            else base_genai_client
        )
        available_models = self._check_model_availability(
            regional_genai_client, filtered_models
        )

        return available_models

    def _check_model_availability(self, genai_client, models):
        def is_available(model) -> bool:
            model_name = model.name

            try:
                # free API call: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count#pricing_and_quota
                genai_client.models.count_tokens(model=model_name, contents="check")
                return True
            except Exception:
                return False

        from concurrent.futures import ThreadPoolExecutor, as_completed

        available_models = []

        with ThreadPoolExecutor(max_workers=4) as pool:
            future_to_model = {pool.submit(is_available, m): m for m in models}
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                if future.result():
                    available_models.append(model)

        return available_models

    def _create_genai_client_for_location(self, location: str):
        from google.genai import Client

        creds = self._construct_google_credentials_from_auth_spec(
            self.google_credentials_spec
        )

        base_url = (
            self.custom_base_api_url or f"https://{location}-aiplatform.googleapis.com"
        )
        http_options = {"base_url": base_url}

        return Client(
            vertexai=True,
            project=self.project_id,
            location=location,
            credentials=creds,
            http_options=http_options,
        )


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
# region AI Studio Authentication
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

        return self._clean_and_sort_models(models)

    def get_embedding_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        models = self._fetch_models_from_api(dialog_creation_context, "embedding")
        if not models:
            LOGGER.info(
                "No embedding models available for the authenticated Google AI Studio user. Using a predefined list of known embedding models."
            )
            return GOOGLE_AI_STUDIO_GEMINI_EMBEDDING_MODELS_FALLBACK

        return self._clean_and_sort_models(models)

    def _fetch_models_from_api(
        self,
        ctx: knext.ExecutionContext | knext.DialogCreationContext,
        model_type: Literal["chat", "embedding"],
    ):
        """
        Always returns either:
        - a list of matched models from the API endpoint
        - an empty list if no models were matched

        Callers are expected to handle the empty case.
        """

        def matches_model_type(model):
            # deprecated models no longer work but still get listed
            def is_deprecated_model():
                if model.description is not None:
                    return "deprecated" in model.description.lower()

                return model.name.split("/")[-1] in KNOWN_DEPRECATED_MODELS

            if is_deprecated_model():
                return False

            if model_type == "chat":
                return "generateContent" in model.supported_actions

            if model_type == "embedding":
                return "embedContent" in model.supported_actions

            return False

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
# region Gemini Chat Model
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
        if isinstance(self.auth_spec, VertexAiConnectionPortObjectSpec):
            return f"publishers/google/models/{self._model_name}"

        if isinstance(self.auth_spec, GoogleAiStudioAuthenticationPortObjectSpec):
            return f"models/{self._model_name}"

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

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
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
                base_url=auth_spec.custom_base_api_url or auth_spec.base_api_url,
            )

        if isinstance(auth_spec, GoogleAiStudioAuthenticationPortObjectSpec):
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = ctx.get_credentials(auth_spec.credentials).password

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
