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
import knime.api.schema as ks
from knime.extension import ConfigurationContext
from ..base import GeneralSettings, OutputFormatOptions
from ._base import (
    hub_connector_icon,
    knime_category,
    create_authorization_headers,
    extract_api_base,
    create_model_choice_provider,
    list_model_ids,
    is_available_model,
    validate_auth_spec,
)


from ..base import (
    ChatModelPortObjectSpec,
    ChatModelPortObject,
)


class KnimeHubChatModelPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(
        self,
        auth_spec: ks.HubAuthenticationPortObjectSpec,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        n_requests: int,
    ) -> None:
        super().__init__()
        self._auth_spec = auth_spec
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._n_requests = n_requests

    @property
    def auth_spec(self) -> ks.HubAuthenticationPortObjectSpec:
        return self._auth_spec

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def n_requests(self) -> int:
        return self._n_requests

    def serialize(self) -> dict:
        return {
            "auth": self.auth_spec.serialize(),
            "model_name": self.model_name,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "n_requests": self._n_requests,
        }

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def top_p(self) -> float:
        return self._top_p

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            ks.HubAuthenticationPortObjectSpec.deserialize(data["auth"]),
            data["model_name"],
            data["max_tokens"],
            data["temperature"],
            data["top_p"],
            n_requests=data.get("n_requests", 1),
        )

    def validate_context(self, ctx: ConfigurationContext):
        validate_auth_spec(self.auth_spec)


class KnimeHubChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> KnimeHubChatModelPortObjectSpec:
        return super().spec

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        from langchain_openai import ChatOpenAI
        from .._credential_auth import (
            CredentialPortTokenProvider,
            create_http_client,
            create_async_http_client,
        )

        auth_spec = self.spec.auth_spec
        token_provider = CredentialPortTokenProvider(auth_spec)
        return ChatOpenAI(
            model=self.spec.model_name,
            openai_api_base=extract_api_base(auth_spec),
            openai_api_key="placeholder",
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
            http_client=create_http_client(token_provider),
            http_async_client=create_async_http_client(token_provider),
        )


knime_chat_model_port_type = knext.port_type(
    "KNIME Hub Chat Model", KnimeHubChatModelPortObject, KnimeHubChatModelPortObjectSpec
)


class ModelSettings(GeneralSettings):
    max_tokens = knext.IntParameter(
        label="Maximum response length (token)",
        description="""
        The maximum number of tokens to generate.

        This value, plus the token count of your prompt, cannot exceed the model's context length.
        """,
        default_value=200,
        min_value=1,
    )

    # Altered from GeneralSettings because OpenAI has temperatures going up to 2
    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 2.0.

        Higher values will lead to less deterministic answers.

        Try 0.9 for more creative applications, and 0 for ones with a well-defined answer.
        It is generally recommended altering this, or Top-p, but not both.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=2.0,
    )

    n_requests = knext.IntParameter(
        label="Number of concurrent requests",
        description="""Maximum number of concurrent requests a single node (e.g. LLM Prompter (Table)) can make to the GenAI Gateway.
        The more requests a node can make in parallel, the faster it executes. Too many requests might get rate-limited by some
        GenAI providers.
        """,
        default_value=1,
        min_value=1,
        is_advanced=True,
        since_version="5.3.1",
    )


@knext.node(
    name="KNIME Hub LLM Selector",
    node_type=knext.NodeType.SOURCE,
    icon_path=hub_connector_icon,
    category=knime_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "GenAI Gateway", "Chat model", "LLM"],
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_port(
    name="KNIME Hub Large Language Model",
    description="A Large Language Model that connects to the KNIME Hub to make requests.",
    port_type=knime_chat_model_port_type,
)
class KnimeHubChatModelConnector:
    """
    Select an LLM configured in the GenAI Gateway of the connected KNIME Hub.

    Connects to a Large Language Model (LLM) configured in the GenAI Gateway of the connected KNIME Hub using the authentication
    provided via the input port.

    Use this node to generate text, answer questions, summarize content or perform other text-based tasks.
    """

    model_name = knext.StringParameter(
        "Model",
        "Select the ID of the LLM to use. "
        "If set via flow variable, the ID can be obtained from the KNIME Hub AI Model Lister node.",
        choices=create_model_choice_provider("chat"),
    )

    model_settings = ModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> KnimeHubChatModelPortObjectSpec:
        # raises exception if the hub authenticator has not been executed
        validate_auth_spec(authentication)
        if self.model_name == "":
            raise knext.InvalidParametersError("No chat model selected.")
        return self._create_spec(authentication)

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> KnimeHubChatModelPortObject:
        if is_available_model(authentication.spec, self.model_name, "chat") is False:
            raise knext.InvalidParametersError(
                f"The selected model {self.model_name} is not served by the connected Hub."
            )
        return KnimeHubChatModelPortObject(self._create_spec(authentication.spec))

    def _create_spec(
        self, authentication: ks.HubAuthenticationPortObjectSpec
    ) -> KnimeHubChatModelPortObjectSpec:
        return KnimeHubChatModelPortObjectSpec(
            authentication,
            self.model_name,
            max_tokens=self.model_settings.max_tokens,
            temperature=self.model_settings.temperature,
            top_p=self.model_settings.top_p,
            n_requests=self.model_settings.n_requests,
        )
