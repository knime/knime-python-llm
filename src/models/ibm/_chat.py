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
from ..base import ChatModelPortObject, ChatModelPortObjectSpec, OutputFormatOptions
from ._base import (
    ibm_watsonx_icon,
    ibm_watsonx_category,
)
from ._auth import (
    IBMwatsonxAuthenticationPortObjectSpec,
    IBMwatsonxAuthenticationPortObject,
    ibm_watsonx_auth_port_type,
)
from ._util import (
    check_model,
    list_chat_models,
    IBMwatsonxChatModelSettings,
)


class IBMwatsonxChatModelPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(
        self,
        auth: IBMwatsonxAuthenticationPortObjectSpec,
        model_id: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        n_requests: int,
        supports_tools: bool,
    ):
        super().__init__()
        self._auth = auth
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._n_requests = n_requests
        self._supports_tools = supports_tools

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def top_p(self) -> float:
        return self._top_p

    @property
    def n_requests(self) -> int:
        return self._n_requests

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    @property
    def auth(self) -> IBMwatsonxAuthenticationPortObjectSpec:
        return self._auth

    def validate_context(self, ctx):
        self.auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "auth": self.auth.serialize(),
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n_requests": self.n_requests,
            "supports_tools": self.supports_tools,
        }

    @classmethod
    def deserialize(cls, data: dict):
        auth = IBMwatsonxAuthenticationPortObjectSpec.deserialize(data["auth"])
        return cls(
            auth=auth,
            model_id=data["model_id"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
            top_p=data["top_p"],
            n_requests=data["n_requests"],
            supports_tools=data["supports_tools"],
        )


class IBMwatsonxChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> IBMwatsonxChatModelPortObjectSpec:
        return super().spec

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        from langchain_ibm import ChatWatsonx

        # Retrieve the name-id map for projects or spaces
        # The map is used to convert the selected name to the corresponding ID
        project_id, space_id = self.spec.auth.get_project_or_space_ids(ctx)

        try:
            chat_model = ChatWatsonx(
                apikey=self.spec.auth.get_api_key(ctx),
                url=self.spec.auth.base_url,
                model_id=self.spec.model_id,
                project_id=project_id,
                space_id=space_id,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
                top_p=self.spec.top_p,
                n_requests=self.spec.n_requests,
            )
        except Exception:
            raise knext.InvalidParametersError(
                "Failed to create the chat model. If you selected a space, make sure that "
                "the space has a valid runtime service instance. You can check this at IBM watsonx.ai Studio "
                "under Manage tab in your space."
            )
        return chat_model


ibm_watsonx_chat_model_port_type = knext.port_type(
    "IBM watsonx.ai Chat Model",
    IBMwatsonxChatModelPortObject,
    IBMwatsonxChatModelPortObjectSpec,
)


@knext.node(
    name="IBM watsonx.ai Chat Model Selector",
    node_type=knext.NodeType.SOURCE,
    icon_path=ibm_watsonx_icon,
    category=ibm_watsonx_category,
    keywords=["IBM", "watsonx", "GenAI"],
)
@knext.input_port(
    "IBM watsonx.ai Authentication",
    "The authentication for the IBM watsonx.ai API.",
    ibm_watsonx_auth_port_type,
)
@knext.output_port(
    "IBM watsonx.ai Chat Model",
    "The IBM watsonx.ai chat model which can be used in the LLM Prompter and Chat Model Prompter.",
    ibm_watsonx_chat_model_port_type,
)
class IBMwatsonxChatModelConnector:
    """Select a chat model provided by the IBM watsonx.ai API.

    In order to use IBM watsonx.ai models, you'll need to create an IBM watsonx.ai account and obtain an API key.
    After successfully authenticating using the **IBM watsonx.ai Authenticator** node, you can select a chat
    model from a predefined list.

    Refer to the [IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-ibm.html)
    for more information on available chat models. At the moment, only the chat models from foundation models are supported.
    Refer to [Choosing a model](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-choose.html?context=wx&locale=en&audience=wdp)
    page for more information on chat models that support tool calls.

    **Note**: If you want to use a space, make sure that the space has a valid runtime service instance. You can check this at
    [IBM watsonx.ai Studio](https://dataplatform.cloud.ibm.com/login) under Manage tab in your space.
    """

    model_id = knext.StringParameter(
        "Model",
        description="The model to use for the chat completion.",
        default_value="",
        choices=list_chat_models,
    )

    model_settings = IBMwatsonxChatModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: IBMwatsonxAuthenticationPortObjectSpec,
    ) -> IBMwatsonxChatModelPortObjectSpec:
        # Check if a model is selected
        check_model(self.model_id)

        auth.validate_context(ctx)

        # assume model supports tools as we cannot check it here without making a request
        return self.create_spec(auth, model_supports_tools=True)

    def create_spec(
        self, auth: IBMwatsonxAuthenticationPortObjectSpec, model_supports_tools: bool
    ) -> IBMwatsonxChatModelPortObjectSpec:
        return IBMwatsonxChatModelPortObjectSpec(
            auth,
            self.model_id,
            self.model_settings.temperature,
            self.model_settings.max_tokens,
            self.model_settings.top_p,
            self.model_settings.n_requests,
            model_supports_tools,
        )

    def execute(
        self, ctx: knext.ExecutionContext, auth: IBMwatsonxAuthenticationPortObject
    ) -> IBMwatsonxChatModelPortObject:
        # Check if the model is still available
        if self.model_id not in auth.spec.get_chat_model_list(ctx):
            raise knext.InvalidParametersError(
                f"The chat model {self.model_id} is not available."
            )
        model_supports_tools = auth.spec.model_supports_tools(self.model_id, ctx)
        return IBMwatsonxChatModelPortObject(
            self.create_spec(auth.spec, model_supports_tools)
        )
