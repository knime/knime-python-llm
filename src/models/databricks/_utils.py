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


from typing import Literal
import knime.extension.nodes as kn
import knime.extension as knext
from ..base import model_category

databricks_workspace_port_type_id = (
    "org.knime.bigdata.databricks.workspace.port.DatabricksWorkspacePortObject"
)


def get_user_agent() -> str:
    from knime.extension.parameter import _extension_version

    return f"KNIME/{_extension_version}"


def get_user_agent_header() -> dict[str, str]:
    return {
        "User-Agent": get_user_agent(),
    }


def workspace_port_type_available() -> bool:
    return kn.has_port_type_for_id(databricks_workspace_port_type_id)


def get_workspace_port_type():
    return kn.get_port_type_for_id(databricks_workspace_port_type_id)


def get_base_url(databricks_workspace_spec):
    from urllib.parse import urljoin

    return urljoin(databricks_workspace_spec.workspace_url, "serving-endpoints")


def get_api_key(databricks_workspace_spec):
    return databricks_workspace_spec.auth_parameters


def check_workspace_available(databricks_workspace_spec):
    try:
        databricks_workspace_spec.auth_parameters
    except ValueError:
        raise knext.InvalidParametersError(
            "Databricks Workspace is not available. Re-execute the connector node."
        )


def get_models(
    databricks_workspace_spec, model_type: Literal["chat", "embeddings"]
) -> list[str]:
    import requests
    from urllib.parse import urljoin

    base_url = get_base_url(databricks_workspace_spec)
    api_key = get_api_key(databricks_workspace_spec)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": get_user_agent(),
    }
    serving_endpoints_url = urljoin(base_url, "api/2.0/serving-endpoints")
    res = requests.get(serving_endpoints_url, headers=headers)
    res.raise_for_status()
    models = res.json()["endpoints"]
    task = "llm/v1/" + model_type
    return [model["name"] for model in models if model["task"] == task]


def get_model_choices_provider(model_type: Literal["chat", "embeddings"]):
    def get_model_choices(ctx: knext.DialogCreationContext):
        if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
            try:
                check_workspace_available(auth_spec)
            except knext.InvalidParametersError:
                return []

            return get_models(auth_spec, model_type)
        return []

    return get_model_choices


# looks nicer
databricks_icon = "icons/databricks/Databricks-embeddings-connector.png"

databricks_category = knext.category(
    path=model_category,
    level_id="databricks",
    name="Databricks",
    description="Contains nodes to connect models hosted on Databricks.",
    icon=databricks_icon,
)
