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

import knime.extension as knext

from ._util import MISTRAL_CHAT_MODELS_FALLBACK, MISTRAL_EMBEDDING_MODELS_FALLBACK
from ._base import mistral_icon, mistral_category
from ..base import CredentialsSettings, AIPortObjectSpec

_default_mistral_api_base = "https://api.mistral.ai/v1"


class MistralAuthenticationPortObjectSpec(AIPortObjectSpec):
    def __init__(
        self, credentials: str, base_url: str = _default_mistral_api_base
    ) -> None:
        super().__init__()
        self._credentials = credentials
        self._base_url = base_url

    @property
    def credentials(self) -> str:
        return self._credentials

    @property
    def base_url(self) -> str:
        return self._base_url

    def validate_context(self, ctx: knext.ConfigurationContext):
        if self.credentials not in ctx.get_credential_names():
            raise knext.InvalidParametersError(
                f"""The selected credentials '{self.credentials}' holding the Mistral AI API key do not exist.
                Make sure that you have selected the correct credentials and that they are still available."""
            )
        api_token = ctx.get_credentials(self.credentials)
        if not api_token.password:
            raise knext.InvalidParametersError(
                f"""The Mistral AI API key '{self.credentials}' does not exist. Make sure that the node you are using to pass the credentials
                (e.g. the Credentials Configuration node) is still passing the valid API key as a flow variable to the downstream nodes."""
            )
        if not self.base_url:
            raise knext.InvalidParametersError("Please provide a base URL.")

    def validate_api_key(self, ctx: knext.ExecutionContext):
        try:
            self._list_chat_models(ctx)
        except Exception as e:
            raise RuntimeError("Could not authenticate with the Mistral AI API.") from e

    def _list_chat_models(
        self, ctx: knext.ConfigurationContext | knext.ExecutionContext
    ) -> list[str]:
        from openai import Client as OpenAIClient

        api_key = ctx.get_credentials(self.credentials).password
        models = (
            OpenAIClient(api_key=api_key, base_url=self.base_url).models.list().data
        )
        # filter for chat models and de-duplicate ids.
        unique_chat_model_ids: list[str] = []
        seen_ids: set[str] = set()

        for model in models:
            model_id = getattr(model, "id", None)
            if model_id is None:
                continue

            capabilities = getattr(model, "capabilities", None)
            supports_chat = bool(
                isinstance(capabilities, dict) and capabilities.get("completion_chat")
            )

            if supports_chat and model_id not in seen_ids:
                seen_ids.add(model_id)
                unique_chat_model_ids.append(model_id)

        return unique_chat_model_ids or MISTRAL_CHAT_MODELS_FALLBACK

    def get_chat_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        try:
            return self._list_chat_models(ctx)
        except Exception:
            return MISTRAL_CHAT_MODELS_FALLBACK

    def _list_embedding_models(
        self, ctx: knext.ConfigurationContext | knext.ExecutionContext
    ) -> list[str]:
        # The Mistral AI API does not expose an embedding-specific capability flag, so all
        # models are listed. Models are sorted to the front if their ID contains "embed" or
        # if all their capability flags are False (i.e. they are likely not chat models).
        from openai import Client as OpenAIClient

        api_key = ctx.get_credentials(self.credentials).password
        models = (
            OpenAIClient(api_key=api_key, base_url=self.base_url).models.list().data
        )
        likely_embedding: list[str] = []
        other: list[str] = []
        seen_ids: set[str] = set()

        for model in models:
            model_id = getattr(model, "id", None)
            if model_id is None or model_id in seen_ids:
                continue
            seen_ids.add(model_id)

            capabilities = getattr(model, "capabilities", None)
            all_capabilities_false = (
                isinstance(capabilities, dict)
                and capabilities
                and not any(capabilities.values())
            )
            if "embed" in model_id or all_capabilities_false:
                likely_embedding.append(model_id)
            else:
                other.append(model_id)

        return likely_embedding + other

    def get_embedding_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        try:
            return self._list_embedding_models(ctx)
        except Exception:
            return []

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
            "base_url": self._base_url,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"], data.get("base_url", _default_mistral_api_base))


class MistralAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: MistralAuthenticationPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> MistralAuthenticationPortObjectSpec:
        return super().spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: MistralAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


mistral_auth_port_type = knext.port_type(
    "Mistral AI Authentication",
    MistralAuthenticationPortObject,
    MistralAuthenticationPortObjectSpec,
)


@knext.node(
    name="Mistral AI Authenticator",
    node_type=knext.NodeType.SOURCE,
    icon_path=mistral_icon,
    category=mistral_category,
    keywords=["Mistral", "GenAI", "Gen AI", "Generative AI"],
)
@knext.output_port(
    "Mistral AI Authentication",
    "Authentication for the Mistral AI API.",
    mistral_auth_port_type,
)
class MistralAuthenticator:
    """Authenticates with the Mistral AI API via API key.

    Authenticates with the Mistral AI API via API key.
    The *password* field of the selected credentials is used as API key while the username is ignored.

    Refer to [Mistral AI](https://docs.mistral.ai/getting-started/quickstart) for details on
    how to create an API key.
    """

    credentials_settings = CredentialsSettings(
        label="Mistral AI API key",
        description="The credentials containing the Mistral AI API key in its *password* field (the *username* is ignored).",
    )

    base_url = knext.StringParameter(
        "Base URL",
        "The base URL of the Mistral AI API.",
        default_value=_default_mistral_api_base,
        is_advanced=True,
    )

    validate_api_key = knext.BoolParameter(
        "Validate API key",
        "If enabled, the API key is validated during execution by calling the list-models endpoint.",
        default_value=False,
        is_advanced=True,
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> MistralAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        spec = self.create_spec()
        spec.validate_context(ctx)
        return spec

    def execute(self, ctx: knext.ExecutionContext) -> MistralAuthenticationPortObject:
        spec = self.create_spec()
        if self.validate_api_key:
            spec.validate_api_key(ctx)
        return MistralAuthenticationPortObject(spec)

    def create_spec(self) -> MistralAuthenticationPortObjectSpec:
        return MistralAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param, base_url=self.base_url
        )
