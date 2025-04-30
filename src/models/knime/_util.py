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
import util
from ._base import (
    hub_connector_icon,
    knime_category,
    list_models_with_descriptions,
    validate_auth_spec,
)
import pyarrow as pa


@knext.parameter_group("Model types")
class ModelTypeSettings:
    chat_models = knext.BoolParameter(
        "Chat models",
        "List available chat models.",
        True,
    )

    embedding_models = knext.BoolParameter(
        "Embedding models",
        "List available embedding models.",
        True,
    )


@knext.node(
    name="KNIME Hub AI Model Lister",
    node_type=knext.NodeType.SOURCE,
    icon_path=hub_connector_icon,
    category=knime_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "GenAI Gateway", "LLM", "RAG"],
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_table(
    name="KNIME Hub AI Model list",
    description="The list of models, including their name, type and description.",
)
class KnimeHubAIModelLister:
    """
    Lists available models in the GenAI Gateway of the connected KNIME Hub.

    Lists available models in the GenAI Gateway of the connected KNIME Hub using the
    authentication provided via the input port.

    Use this node to retrieve the available models with their name, type and description.
    """

    model_types = ModelTypeSettings()

    # Name, KNIME type, and PyArrow type of the columns to output
    column_list = [
        util.OutputColumn("Name", knext.string(), pa.string()),
        util.OutputColumn("Type", knext.string(), pa.string()),
        util.OutputColumn("Description", knext.string(), pa.string()),
    ]

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> knext.Schema:
        # raises exception if the hub authenticator has not been executed
        validate_auth_spec(authentication)

        knime_columns = [column.to_knime_column() for column in self.column_list]

        return knext.Schema.from_columns(knime_columns)

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> knext.Table:
        import pandas as pd

        available_models = []

        if self.model_types.chat_models:
            available_models.extend(
                list_models_with_descriptions(authentication.spec, "chat")
            )

        if self.model_types.embedding_models:
            available_models.extend(
                list_models_with_descriptions(authentication.spec, "embedding")
            )

        if not available_models:
            return self._create_empty_table()

        models_df = pd.DataFrame(
            available_models, columns=["Name", "Type", "Description"]
        )

        return knext.Table.from_pandas(models_df)

    def _create_empty_table(self) -> knext.Table:
        """Constructs an empty KNIME Table with the correct output columns."""

        return util.create_empty_table(None, self.column_list)
