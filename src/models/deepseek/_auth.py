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
from ._base import deepseek_icon, deepseek_category
from ..base import CredentialsSettings, AIPortObjectSpec

_default_deepseek_api_base = "https://api.deepseek.com"


class DeepSeekAuthenticationPortObjectSpec(AIPortObjectSpec):
    def __init__(
        self, credentials: str, base_url: str = _default_deepseek_api_base
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
                f"""The selected credentials '{self.credentials}' holding the DeepSeek API key do not exist. 
                Make sure that you have selected the correct credentials and that they are still available."""
            )
        api_token = ctx.get_credentials(self.credentials)
        if not api_token.password:
            raise knext.InvalidParametersError(
                f"""The DeepSeek API key '{self.credentials}' does not exist. Make sure that the node you are using to pass the credentials 
                (e.g. the Credentials Configuration node) is still passing the valid API key as a flow variable to the downstream nodes."""
            )

        if not self.base_url:
            raise knext.InvalidParametersError("Please provide a base URL.")

    def validate_api_key(self, ctx: knext.ExecutionContext):
        try:
            self._get_models_from_api(ctx)
        except Exception as e:
            raise RuntimeError("Could not authenticate with the DeepSeek API.") from e

    def _get_models_from_api(
        self, ctx: knext.ConfigurationContext | knext.ExecutionContext
    ) -> list[str]:
        from openai import Client as OpenAIClient

        key = ctx.get_credentials(self.credentials).password
        base_url = self.base_url
        return [
            model.id
            for model in OpenAIClient(api_key=key, base_url=base_url).models.list().data
        ]

    def get_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        try:
            return self._get_models_from_api(ctx)
        except Exception:
            return ["deepseek-chat", "deepseek-reasoner"]

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
            "base_url": self._base_url,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["credentials"], data.get("base_url", _default_deepseek_api_base)
        )


class DeepSeekAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: DeepSeekAuthenticationPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> DeepSeekAuthenticationPortObjectSpec:
        return super().spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: DeepSeekAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


deepseek_auth_port_type = knext.port_type(
    "DeepSeek Authentication",
    DeepSeekAuthenticationPortObject,
    DeepSeekAuthenticationPortObjectSpec,
)


@knext.node(
    name="DeepSeek Authenticator",
    node_type=knext.NodeType.SOURCE,
    icon_path=deepseek_icon,
    category=deepseek_category,
    keywords=["DeepSeek", "GenAI"],
)
@knext.output_port(
    "DeepSeek Authentication",
    "Authentication for the DeepSeek API",
    deepseek_auth_port_type,
)
class DeepSeekAuthenticator:
    """Authenticates with the DeepSeek API via API key.

    Authenticates with the DeepSeek API via API key.
    The *password* field of the selected credentials is used as API key while the username is ignored.
    You can apply for an API key at [DeepSeek](https://platform.deepseek.com/api_keys).

    **Note**: Data sent to the DeepSeek API is stored and used for the training of future models.
    """

    credentials_settings = CredentialsSettings(
        label="DeepSeek API key",
        description="The credentials containing the DeepSeek API key in its *password* field (the *username* is ignored).",
    )

    base_url = knext.StringParameter(
        "Base URL",
        "The base URL of the DeepSeek API.",
        default_value=_default_deepseek_api_base,
        is_advanced=True,
    )

    validate_api_key = knext.BoolParameter(
        "Validate API key",
        "If set, the API key is validated during execution by fetching the available models.",
        False,
        is_advanced=True,
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> DeepSeekAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        spec = self.create_spec()
        spec.validate_context(ctx)
        return spec

    def execute(self, ctx: knext.ExecutionContext) -> DeepSeekAuthenticationPortObject:
        spec = self.create_spec()
        if self.validate_api_key:
            spec.validate_api_key(ctx)
        return DeepSeekAuthenticationPortObject(spec)

    def create_spec(self) -> DeepSeekAuthenticationPortObjectSpec:
        return DeepSeekAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param, base_url=self.base_url
        )
