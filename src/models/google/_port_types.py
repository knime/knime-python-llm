# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------


# Port types:
# - Vertex AI Connection (currently only Gemini API, then other APIs later)
# - Google AI Studio Authentication (Gemini API)
# - Gemini Chat Model
# - Gemini Embedding Model

import knime.extension as knext

from typing import Dict, List, Literal, Optional
import logging

from base import AIPortObjectSpec
from models.base import (
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    EmbeddingsPortObject,
    EmbeddingsPortObjectSpec,
    OutputFormatOptions,
)
from ._utils import (
    KNOWN_DEPRECATED_MODELS,
    GEMINI_CHAT_MODELS_FALLBACK,
    VERTEX_AI_GEMINI_EMBEDDING_MODELS_FALLBACK,
    GOOGLE_AI_STUDIO_GEMINI_EMBEDDING_MODELS_FALLBACK,
    GOOGLE_AI_STUDIO_GEMINI_IMAGE_MODELS_FALLBACK
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Generic Gemini Connection Port Type
# region Generic Auth
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
    
    def get_image_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        raise NotImplementedError(
            "Subclasses of the generic Gemini connection must implement this method."
        )
    
    def create_chat_model_name(self, model_name) -> str:
        raise NotImplementedError(
            "Subclasses of the generic Gemini connection must implement this method."
        )

    def create_embedding_model_name(self, model_name) -> str:
        raise NotImplementedError(
            "Subclasses of the generic Gemini connection must implement this method."
        )

    def create_chat_model(
        self,
        execution_ctx: knext.ExecutionContext,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ):
        raise NotImplementedError(
            "Subclasses of the generic Gemini connection must implement this method."
        )

    def create_embedding_model(
        self,
        execution_ctx: knext.ExecutionContext,
        model_name: str,
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
# region Vertex Auth
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

    def create_chat_model_name(self, model_name) -> str:
        return f"publishers/google/models/{model_name}"

    def create_embedding_model_name(self, model_name) -> str:
        return model_name

    def _construct_google_credentials(
        self,
    ):
        def get_refresh_handler(auth_spec: knext.CredentialPortObjectSpec):
            return lambda request, scopes: (
                auth_spec.auth_parameters,
                auth_spec.expires_after,
            )

        token = getattr(self.google_credentials_spec, "auth_parameters", None)
        expiry = getattr(self.google_credentials_spec, "expires_after", None)

        if token is None or expiry is None:
            raise knext.InvalidParametersError(
                "Google Cloud credentials are misconfigured. Are you authenticated using a service account?"
            )

        from google.oauth2.credentials import Credentials

        return Credentials(
            token=token,
            expiry=expiry,
            refresh_handler=get_refresh_handler(self.google_credentials_spec),
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
            return GEMINI_CHAT_MODELS_FALLBACK

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
        us_central_location = "us-central1"
        try:
            base_genai_client = self._create_genai_client_for_location(
                us_central_location
            )
        except Exception as e:
            LOGGER.warning(
                "Could not fetch list of models from Vertex AI."
                " Ensure you are providing valid authentication."
                f" Error: {e}"
            )
            return []

        # 2. filter out deprecated and irrelevant models
        paged_model_list = base_genai_client.models.list(config={"query_base": True})
        filtered_models = [m for m in paged_model_list if matches_model_type(m)]

        # 3. filter out models unavailable to the authenticated user at the specified location
        regional_genai_client = (
            self._create_genai_client_for_location(self.location)
            if self.location != us_central_location
            else base_genai_client
        )
        available_models = self._check_model_availability(
            regional_genai_client, filtered_models, model_type
        )

        return available_models

    def _check_model_availability(self, genai_client, models, model_type):
        def is_available(model) -> bool:
            model_name = model.name

            try:
                # free API calls:
                # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count#pricing_and_quota
                # https://ai.google.dev/api/models#method:-models.get
                if model_type == "chat":
                    genai_client.models.count_tokens(model=model_name, contents="check")
                if model_type == "embedding":
                    genai_client.models.get(model=model_name)

                return True
            except Exception:
                return False

        from concurrent.futures import ThreadPoolExecutor, as_completed

        available_models = []

        # HACK: the node dialog awaits all requests to complete before rendering.
        # We try to decrease that time by parallelising these requests. Ideally, the Python
        # framework would provide support for async "choices" loading
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_to_model = {pool.submit(is_available, m): m for m in models}
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                if future.result():
                    available_models.append(model)

        return available_models

    def _create_genai_client_for_location(self, location: str):
        from google.genai import Client

        creds = self._construct_google_credentials()

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

    def create_chat_model(
        self,
        execution_ctx: knext.ExecutionContext,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ):
        from langchain_google_vertexai import ChatVertexAI

        google_credentials = self._construct_google_credentials()

        return ChatVertexAI(
            model_name=model_name,
            project=self.project_id,
            location=self.location,
            credentials=google_credentials,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=2,  # default is 6, instead we just try twice before failing
            base_url=self.custom_base_api_url or self.base_api_url,
        )

    def create_embedding_model(self, execution_ctx, model_name):
        from langchain_google_vertexai import VertexAIEmbeddings

        google_credentials = self._construct_google_credentials()

        return VertexAIEmbeddings(
            model_name=model_name,
            project=self.project_id,
            location=self.location,
            max_retries=2,  # default is 6, instead we just try twice before failing
            base_url=self.custom_base_api_url or self.base_api_url,
            credentials=google_credentials,
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
# region AI Studio Auth
# ---------------------------------------------------------------------
class GoogleAiStudioAuthenticationPortObjectSpec(GenericGeminiConnectionPortObjectSpec):
    def __init__(self, credentials: str) -> None:
        super().__init__(connection_type="google_ai_studio")
        self._credentials = credentials

    @property
    def credentials(self) -> str:
        return self._credentials

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"])

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

    def create_chat_model_name(self, model_name):
        return f"models/{model_name}"

    def create_embedding_model_name(self, model_name) -> str:
        return f"models/{model_name}"

    def get_chat_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        models = self._fetch_models_from_api(dialog_creation_context, "chat")
        if not models:
            LOGGER.info(
                "No chat models available for the authenticated Google AI Studio user. Using a predefined list of known chat models."
            )
            return GEMINI_CHAT_MODELS_FALLBACK

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

    def get_image_model_list(
        self, dialog_creation_context: knext.DialogCreationContext
    ) -> List[str]:
        models = self._fetch_models_from_api(dialog_creation_context, "image")
        if not models:
            LOGGER.info(
                "No image models available for the authenticated Google AI Studio user. Using a predefined list of known image models."
            )
            return GOOGLE_AI_STUDIO_GEMINI_IMAGE_MODELS_FALLBACK

        return self._clean_and_sort_models(models)

    def _fetch_models_from_api(
        self,
        ctx: knext.ExecutionContext | knext.DialogCreationContext,
        model_type: Literal["chat", "embedding", "image"],
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

            if  model_type == "image":
                return model_type in model.name 

            return False

        from google.genai import Client

        # use a placeholder key to allow the dialog to render even if credentials are not saved
        key = ctx.get_credentials(self.credentials).password or "placeholder"
        client = Client(api_key=key)

        try:
            # this is a paged iterator that automatically fetches the next page
            paged_model_list = client.models.list()
        except Exception as e:
            LOGGER.warning(
                "Could not fetch list of models from Google AI Studio."
                " Ensure you are providing a valid API key for authentication."
                f" Error: {e}"
            )
            return []

        return [m for m in paged_model_list if matches_model_type(m)]

    def create_chat_model(
        self,
        execution_ctx: knext.ExecutionContext,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ):
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = execution_ctx.get_credentials(self.credentials).password

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def create_embedding_model(self, execution_ctx, model_name):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = execution_ctx.get_credentials(self.credentials).password

        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )


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
        connection_spec: GenericGeminiConnectionPortObjectSpec,
        model_name: str,
        max_output_tokens: int,
        temperature: float,
        n_requests: int,
    ):
        super().__init__()
        self._connection_spec = connection_spec
        self._connection_type = connection_spec._connection_type
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._n_requests = n_requests

    @property
    def connection_spec(self) -> GenericGeminiConnectionPortObjectSpec:
        return self._connection_spec

    @property
    def model_name(self) -> str:
        return self.connection_spec.create_chat_model_name(self._model_name)

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    @property
    def temperature(self) -> int:
        return self._temperature

    @property
    def n_requests(self) -> int:
        return self._n_requests
    
    @property
    def supported_output_formats(self) -> list[OutputFormatOptions]:
        return [
            OutputFormatOptions.Text,
            OutputFormatOptions.Structured
        ]

    def serialize(self):
        return {
            "connection_type": self._connection_type,
            "connection_spec_data": self._connection_spec.serialize(),
            "model_name": self._model_name,
            "max_output_tokens": self._max_output_tokens,
            "temperature": self._temperature,
            "n_requests": self._n_requests,
        }

    @classmethod
    def deserialize(cls, data: Dict):
        connection_type = data["connection_type"]
        connection_spec_data = data["connection_spec_data"]

        if connection_type == "vertex_ai":
            connection_spec = VertexAiConnectionPortObjectSpec.deserialize(
                connection_spec_data
            )
        elif connection_type == "google_ai_studio":
            connection_spec = GoogleAiStudioAuthenticationPortObjectSpec.deserialize(
                connection_spec_data
            )
        else:
            raise ValueError(
                f"Unknown connection spec type during deserialization: {connection_type}"
            )

        return cls(
            connection_spec=connection_spec,  # _connection_type gets inferred from the connection spec object
            model_name=data["model_name"],
            max_output_tokens=data["max_output_tokens"],
            temperature=data["temperature"],
            n_requests=data["n_requests"],
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
        return self.spec.connection_spec.create_chat_model(
            execution_ctx=ctx,
            model_name=self.spec.model_name,
            max_tokens=self.spec.max_output_tokens,
            temperature=self.spec.temperature,
        )


gemini_chat_model_port_type = knext.port_type(
    "Gemini Chat Model",
    GeminiChatModelPortObject,
    GeminiChatModelPortObjectSpec,
)


# ---------------------------------------------------------------------
# Gemini Embedding Model Port Type
# region Gemini Embeddings
# ---------------------------------------------------------------------
class GeminiEmbeddingModelPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self,
        connection_spec: GenericGeminiConnectionPortObjectSpec,
        model_name: str,
    ):
        super().__init__()
        self._connection_spec = connection_spec
        self._connection_type = connection_spec._connection_type
        self._model_name = model_name

    @property
    def connection_spec(self) -> GenericGeminiConnectionPortObjectSpec:
        return self._connection_spec

    @property
    def model_name(self) -> str:
        return self.connection_spec.create_embedding_model_name(self._model_name)

    def serialize(self):
        return {
            "connection_type": self._connection_type,
            "connection_spec_data": self._connection_spec.serialize(),
            "model_name": self._model_name,
        }

    @classmethod
    def deserialize(cls, data: Dict):
        connection_type = data["connection_type"]
        connection_spec_data = data["connection_spec_data"]

        if connection_type == "vertex_ai":
            connection_spec = VertexAiConnectionPortObjectSpec.deserialize(
                connection_spec_data
            )
        elif connection_type == "google_ai_studio":
            connection_spec = GoogleAiStudioAuthenticationPortObjectSpec.deserialize(
                connection_spec_data
            )
        else:
            raise ValueError(
                f"Unknown connection spec type during deserialization: {connection_type}"
            )

        return cls(
            connection_spec=connection_spec,  # _connection_type gets inferred from the connection spec object
            model_name=data["model_name"],
        )


class GeminiEmbeddingModelPortObject(EmbeddingsPortObject):
    def __init__(self, spec: GeminiEmbeddingModelPortObjectSpec):
        super().__init__(spec=spec)

    @property
    def spec(self) -> GeminiEmbeddingModelPortObjectSpec:
        return self._spec

    def create_model(self, ctx):
        return self.spec.connection_spec.create_embedding_model(
            execution_ctx=ctx, model_name=self.spec.model_name
        )


gemini_embedding_model_port_type = knext.port_type(
    "Gemini Embedding Model",
    GeminiEmbeddingModelPortObject,
    GeminiEmbeddingModelPortObjectSpec,
)
