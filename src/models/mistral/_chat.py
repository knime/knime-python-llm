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

from ._util import MISTRAL_CHAT_MODELS_FALLBACK, MISTRAL_CHAT_DEFAULT
from ..base import (
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    OutputFormatOptions,
)
from ._base import mistral_icon, mistral_category
from ._auth import (
    MistralAuthenticationPortObjectSpec,
    MistralAuthenticationPortObject,
    mistral_auth_port_type,
)


@knext.parameter_group(label="Model Parameters")
class MistralChatModelSettings:
    temperature = knext.DoubleParameter(
        "Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 1.0.

        Higher values produce more random and creative outputs,
        while lower values produce more focused and deterministic outputs.
        Mistral AI recommends values between 0.0 and 0.7.
        """,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
    )

    max_tokens = knext.IntParameter(
        "Max Tokens",
        description="The maximum number of tokens to generate in the response.",
        default_value=4096,
    )

    n_requests = knext.IntParameter(
        label="Number of concurrent requests",
        description="""Maximum number of requests sent to Mistral AI in parallel.

        Increasing this value can improve throughput, but each parallel request also counts toward
        your Mistral AI API usage limits. If this value is set too high, some requests may be rejected
        because rate limits are exceeded, such as the allowed number of requests per second or tokens
        per minute.
        """,
        default_value=1,
        min_value=1,
        is_advanced=True,
    )


class MistralChatModelPortObjectSpec(ChatModelPortObjectSpec):
    """Spec of a Mistral AI Chat Model"""

    def __init__(
        self,
        auth: MistralAuthenticationPortObjectSpec,
        model: str,
        temperature: float,
        max_tokens: int,
        n_requests=1,
    ):
        super().__init__(n_requests)
        self._auth = auth
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def auth(self) -> MistralAuthenticationPortObjectSpec:
        return self._auth

    @property
    def supported_output_formats(self) -> list[OutputFormatOptions]:
        return [
            OutputFormatOptions.Text,
            OutputFormatOptions.Structured,
        ]

    def validate_context(self, ctx):
        self.auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "auth": self._auth.serialize(),
            "n_requests": self._n_requests,
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    @classmethod
    def deserialize(cls, data: dict):
        auth = MistralAuthenticationPortObjectSpec.deserialize(data["auth"])
        return cls(
            auth=auth,
            model=data["model"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
            n_requests=data["n_requests"],
        )


class MistralChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> MistralChatModelPortObjectSpec:
        return super().spec

    def create_model(
        self,
        ctx: knext.ExecutionContext,
        output_format: OutputFormatOptions = OutputFormatOptions.Text,
    ):
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            mistral_api_key=ctx.get_credentials(self.spec.auth.credentials).password,
            endpoint=self.spec.auth.base_url,
            model=self.spec.model,
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
        )


mistral_chat_model_port_type = knext.port_type(
    "Mistral AI Chat Model",
    MistralChatModelPortObject,
    MistralChatModelPortObjectSpec,
)


def _list_models(ctx: knext.ConfigurationContext):
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_chat_model_list(ctx)
    return MISTRAL_CHAT_MODELS_FALLBACK


@knext.node(
    name="Mistral AI LLM Selector",
    node_type=knext.NodeType.SOURCE,
    icon_path=mistral_icon,
    category=mistral_category,
    keywords=["Mistral", "GenAI", "Gen AI", "Generative AI"],
)
@knext.input_port(
    "Mistral AI Authentication",
    "The authentication for the Mistral AI API.",
    mistral_auth_port_type,
)
@knext.output_port(
    "Mistral AI Large Language Model",
    "The Mistral AI large language model which can be used in the LLM Prompter and LLM Chat Prompter nodes.",
    mistral_chat_model_port_type,
)
class MistralChatModelConnector:
    """Select an LLM provided by the Mistral AI API.

    This node establishes a connection with a Large Language Model (LLM) from Mistral AI. After successfully
    authenticating using the **Mistral AI Authenticator** node, you can select a model from those available in
    the Mistral AI API.
    """

    model = knext.StringParameter(
        "Model",
        description="The model to use. The available models are fetched from the Mistral AI API if possible.",
        default_value=MISTRAL_CHAT_DEFAULT,
        choices=_list_models,
    )

    model_settings = MistralChatModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: MistralAuthenticationPortObjectSpec,
    ) -> MistralChatModelPortObjectSpec:
        auth.validate_context(ctx)
        return self.create_spec(auth)

    def create_spec(
        self, auth: MistralAuthenticationPortObjectSpec
    ) -> MistralChatModelPortObjectSpec:
        return MistralChatModelPortObjectSpec(
            auth=auth,
            model=self.model,
            temperature=self.model_settings.temperature,
            max_tokens=self.model_settings.max_tokens,
            n_requests=self.model_settings.n_requests,
        )

    def execute(
        self, ctx: knext.ExecutionContext, auth: MistralAuthenticationPortObject
    ) -> MistralChatModelPortObject:
        return MistralChatModelPortObject(self.create_spec(auth.spec))
