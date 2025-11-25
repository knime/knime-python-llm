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
    list_models_with_scope_info,
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


@knext.parameter_group("Output columns", since_version="5.8.0")
class OutputColumnSettings:
    """Controls which metadata columns are included in the output table.
    By default, all columns are included.
    """

    scope_id = knext.BoolParameter(
        "Scope ID",
        "Include the model's scope identifier.",
        default_value=lambda v: v >= knext.Version(5, 9, 0),
    )

    scope_name = knext.BoolParameter(
        "Scope Name",
        "Include the model's scope name (e.g., 'Global', team name).",
        default_value=lambda v: v >= knext.Version(5, 9, 0),
    )

    id = knext.BoolParameter(
        "ID",
        "Include the model's unique identifier.",
        default_value=lambda v: v >= knext.Version(5, 8, 0),
    )

    platform_id = knext.BoolParameter(
        "Platform ID",
        "Include the model's platform identifier.",
        default_value=lambda v: v >= knext.Version(5, 10, 0),
    )

    platform_name = knext.BoolParameter(
        "Platform Name",
        "Include the model's platform name (e.g., OpenAI, Anthropic).",
        default_value=lambda v: v >= knext.Version(5, 10, 0),
    )

    name = knext.BoolParameter(
        "Name",
        "Include the model's display name.",
        default_value=True,
    )


    type = knext.BoolParameter(
        "Type",
        "Include the model's type (e.g., chat, embedding).",
        default_value=True,
    )

    description = knext.BoolParameter(
        "Description",
        "Include the model's description if available.",
        default_value=True,
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

    # Full set of potential output columns (order defines default order)
    # Scope columns first, then ID, Platform ID, Platform Name, Name, Type, Description
    full_column_list = [
        util.OutputColumn("Scope ID", knext.string(), pa.string()),
        util.OutputColumn("Scope Name", knext.string(), pa.string()),
        util.OutputColumn("ID", knext.string(), pa.string()),
        util.OutputColumn("Platform ID", knext.string(), pa.string()),
        util.OutputColumn("Platform Name", knext.string(), pa.string()),
        util.OutputColumn("Name", knext.string(), pa.string()),
        util.OutputColumn("Type", knext.string(), pa.string()),
        util.OutputColumn("Description", knext.string(), pa.string()),
    ]

    output_columns = OutputColumnSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> knext.Schema:
        # raises exception if the hub authenticator has not been executed
        validate_auth_spec(authentication)
        selected_columns = self._selected_columns()
        knime_columns = [column.to_knime_column() for column in selected_columns]
        return knext.Schema.from_columns(knime_columns)

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> knext.Table:
        import pandas as pd

        # Get models with scope information
        models_with_scope = list_models_with_scope_info(authentication.spec, mode=None)

        modes = set()

        if self.model_types.chat_models:
            modes.add("chat")
        if self.model_types.embedding_models:
            modes.add("embedding")

        models_with_scope = [
            (model, scope_id, scope_name)
            for model, scope_id, scope_name in models_with_scope
            if model.mode in modes
        ]

        # Sort by mode, then scope name, then model name
        models_with_scope.sort(key=lambda x: (x[0].mode, x[2] or "", x[0].name))

        if not models_with_scope:
            return self._create_empty_table()

        df = pd.DataFrame(
            [self._to_tuple(model, scope_id, scope_name) for model, scope_id, scope_name in models_with_scope],
            columns=["Scope ID", "Scope Name", "ID", "Platform ID", "Platform Name", "Name", "Type", "Description"],
        )

        # Reduce to selected columns (preserving order of full_column_list)
        selected_names = [col.default_name for col in self._selected_columns()]
        df = df[selected_names]
        return knext.Table.from_pandas(df)

    def _to_tuple(self, model, scope_id: str, scope_name: str) -> tuple:
        platform_name = _get_platform_title(model.platform)
        return (scope_id, scope_name, model.id, model.platform, platform_name, model.name, model.mode, model.description)

    def _create_empty_table(self) -> knext.Table:
        """Constructs an empty KNIME Table with the correct output columns."""

        return util.create_empty_table(None, self._selected_columns())

    # Helper methods -----------------------------------------------------
    def _selected_columns(self):
        """Return list of OutputColumn objects in order based on user selection."""
        oc = self.output_columns
        selection_flags = {
            "Scope ID": oc.scope_id,
            "Scope Name": oc.scope_name,
            "ID": oc.id,
            "Platform ID": oc.platform_id,
            "Platform Name": oc.platform_name,
            "Name": oc.name,
            "Type": oc.type,
            "Description": oc.description,
        }
        return [c for c in self.full_column_list if selection_flags[c.default_name]]

def _get_platform_title(platform: str | None) -> str:
    """
    Convert platform identifier to user-friendly title.
    Taken from "knime-hub-webapp/jsonforms/ai-models/constants.ts"
    to make it consistent with the frontend.
    """
    if platform is None:
        return ""
    
    PLATFORM_TITLES = {
        "openai": "OpenAI",
        "azure": "Azure OpenAI",
        "bedrock": "Amazon Bedrock",
        "aistudio": "Google AI Studio",
        "anthropic": "Anthropic",
        "huggingface": "Hugging Face Hub",
        "openaiCompatible": "OpenAI API-compatible Providers",
    }
    
    return PLATFORM_TITLES.get(platform, platform)
