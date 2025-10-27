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
from ..base import model_category
import knime.api.schema as ks
from urllib.parse import urlparse, urlunparse
from typing import Callable

hub_connector_icon = "icons/knime/Hub_AI_connector.png"

knime_category = knext.category(
    path=model_category,
    level_id="knime",
    name="KNIME",
    description="Models that connect to the KNIME Hub.",
    icon=hub_connector_icon,
)


def create_authorization_headers(
    auth_spec: ks.HubAuthenticationPortObjectSpec,
) -> dict[str, str]:
    return {
        "Authorization": f"{auth_spec.auth_schema} {auth_spec.auth_parameters}",
    }


def validate_auth_spec(auth_spec: ks.HubAuthenticationPortObjectSpec) -> None:
    if auth_spec.hub_url is None:
        raise knext.InvalidParametersError(
            "KNIME Hub connection not available. Please re-execute the node."
        )


def extract_api_base(auth_spec: ks.HubAuthenticationPortObjectSpec) -> str:
    try:
        validate_auth_spec(auth_spec)
    except knext.InvalidParametersError as ex:
        # ValueError does not add the exception type to the error message in the dialog
        raise ValueError(str(ex))
    hub_url = auth_spec.hub_url
    parsed_url = urlparse(hub_url)
    # drop params, query and fragment
    ai_proxy_url = (parsed_url.scheme, parsed_url.netloc, "ai-gateway/v1", "", "", "")
    return urlunparse(ai_proxy_url)


def create_model_choice_provider(
    mode: str,
) -> Callable[[knext.DialogCreationContext], list[str]]:
    def model_choices_provider(ctx: knext.DialogCreationContext) -> list[str]:
        model_list = []
        if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
            model_list = list_model_choices(auth_spec, mode)
            model_list.sort(key=lambda c: c.label)
        return model_list

    return model_choices_provider


def list_model_choices(auth_spec, mode: str | None = None) -> list[knext.StringParameter.Choice]:
    return [knext.StringParameter.Choice(model.id, model.name, model.description) for model in list_models(auth_spec, mode)]

def is_available_model(auth_spec, model_id: str, mode: str) -> bool:
    models = list_models(auth_spec, mode)
    allowed_ids = set()
    for model in models:
        allowed_ids.add(model.id)
        # models in hub:global scope can be referenced by name for backwards compatibility
        if model.scope_id == "hub:global":
            allowed_ids.add(model.name)
    return model_id in allowed_ids

def list_model_ids(auth_spec, mode: str) -> list[str]:
    return [model.id for model in list_models(auth_spec, mode)]

def list_models(auth_spec, mode: str | None = None):
    import requests
    from ._models import ModelsResponse

    api_base = extract_api_base(auth_spec)
    model_info = api_base + "/management/models"
    if mode is not None:
        model_info = model_info + "?mode=" + mode
    response = requests.get(
        url=model_info, headers=create_authorization_headers(auth_spec)
    )
    if response.status_code == 404:
        raise ValueError(
            "The GenAI gateway is not reachable. Is it activated in the connected KNIME Hub?"
        )
    response.raise_for_status()
    models_resp = ModelsResponse.model_validate(response.json())
    return models_resp.models


def list_models_with_descriptions(auth_spec, mode: str) -> list[tuple[str, str, str]]:
    return [
        (data.name, data.mode, data.description or None)
        for data in list_models(auth_spec, mode)
    ]
