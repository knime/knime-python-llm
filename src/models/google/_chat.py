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


# Nodes:
# - Gemini LLM Selector

import knime.extension as knext

from ._utils import (
    google_category,
    gemini_icon,
)
from ._port_types import (
    GenericGeminiConnectionPortObject,
    generic_gemini_connection_port_type,
    GenericGeminiConnectionPortObjectSpec,
    gemini_chat_model_port_type,
    GeminiChatModelPortObjectSpec,
    GeminiChatModelPortObject,
)


def _list_gemini_chat_models(
    ctx: knext.DialogCreationContext,
):
    input_specs = ctx.get_input_specs()
    generic_connection_spec: GenericGeminiConnectionPortObjectSpec = input_specs[0]

    # in case the config dialog is open without any inputs
    # TODO: we return a list with an empty string, because returning an empty
    # list causes the UI of the dialogue setting to be stuck in the "Loading" state.
    # If/when that FE behaviour changes, we should adjust this accordingly.
    if generic_connection_spec is None:
        return [""]

    return generic_connection_spec.get_chat_model_list(ctx)


@knext.node(
    "Gemini LLM Selector",
    node_type=knext.NodeType.SOURCE,
    icon_path=gemini_icon,
    category=google_category,
    keywords=["Google AI", "Vertex AI", "Gemini", "GenAI", "Chat model"],
)
@knext.input_port(
    "Gemini Connection",
    "An authenticated connection to either Vertex AI or Google AI Studio.",
    port_type=generic_gemini_connection_port_type,
)
@knext.output_port(
    "Gemini Large Language Model",
    "A Gemini large language model.",
    port_type=gemini_chat_model_port_type,
)
class GeminiChatModelConnector:
    """Select a Gemini language model available through either Vertex AI or Google AI Studio.

    This node allows selecting a Gemini language model using an authenticated connection obtained
    either from the **Vertex AI Connector** node, or from the **Google AI Studio Authenticator** node.
    """

    model_name = knext.StringParameter(
        "Model",
        """Select the Gemini language model to use.
        The list of available models is fetched using the provided Gemini connection.

        If connection with the API cannot be established, the list is populated with known Gemini models
        appropriate for the connection type.
        """,
        choices=_list_gemini_chat_models,
    )

    max_output_tokens = knext.IntParameter(
        "Maximum response tokens",
        """Specify the number of tokens to constrain the model's responses to.

        A token is equivalent to about 4 characters for Gemini models. 100 tokens are about 60-80 English words.
        """,
        min_value=0,
        default_value=4096,
    )

    temperature = knext.DoubleParameter(
        "Temperature",
        """Specify the temperature for the model's responses.

        Temperature controls the degree of randomness in token selection.
        Lower temperatures are good for prompts that require a more deterministic or less open-ended response,
        while higher temperatures can lead to more diverse or creative results.
        A temperature of 0 is deterministic, meaning that the highest probability response is always selected.
        """,
        min_value=0.0,
        max_value=1.0,
        default_value=0.7,
    )

    n_requests = knext.IntParameter(
        "Number of concurrent requests",
        """Specify the number of maximum requests that can be made concurrently.

        Increasing this number is particularly useful in conjunction with the **LLM Prompter (Table)** node,
        as it operates in a row-based manner (e.g. 10 concurrent requests would result
        in a batch of 10 rows being processed concurrently).
        """,
        10,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        generic_gemini_connection_spec: GenericGeminiConnectionPortObjectSpec,
    ) -> GeminiChatModelPortObjectSpec:
        return self._create_output_spec(generic_gemini_connection_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        generic_gemini_connection: GenericGeminiConnectionPortObject,
    ) -> GeminiChatModelPortObject:
        spec = self._create_output_spec(generic_gemini_connection.spec)

        return GeminiChatModelPortObject(spec=spec)

    def _create_output_spec(
        self, connection_spec: GenericGeminiConnectionPortObjectSpec
    ):
        if not self.model_name:
            raise knext.InvalidParametersError(
                "No model selected. Ensure that you selected a model from the list of available models."
            )

        return GeminiChatModelPortObjectSpec(
            connection_spec=connection_spec,
            model_name=self.model_name,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            n_requests=self.n_requests,
        )
