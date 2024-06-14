import knime.extension as knext
from ..base import model_category
import knime.api.schema as ks
from urllib.parse import urlparse, urlunparse
import requests
from typing import Callable

hub_connector_icon = "icons/Hub_AI_connector.png"

knime_category = knext.category(
    path=model_category,
    level_id="knime",
    name="KNIME",
    description="Models that connect to the KNIME Hub.",
    icon=hub_connector_icon,
)


def _create_authorization_headers(
    auth_spec: ks.HubAuthenticationPortObjectSpec,
) -> dict[str, str]:
    return {
        "Authorization": f"{auth_spec.auth_schema} {auth_spec.auth_parameters}",
    }


def _extract_api_base(auth_spec: ks.HubAuthenticationPortObjectSpec) -> str:
    hub_url = auth_spec.hub_url
    parsed_url = urlparse(hub_url)
    # drop params, query and fragment
    ai_proxy_url = (parsed_url.scheme, parsed_url.netloc, "ai-gateway", "", "", "")
    return urlunparse(ai_proxy_url)


def _list_models_in_dialog(
    mode: str,
) -> Callable[[knext.DialogCreationContext], list[str]]:
    def list_models(ctx: knext.DialogCreationContext) -> list[str]:
        if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
            return _list_models(auth_spec, mode)
        return [""]

    return list_models


def _list_models(auth_spec, mode: str) -> list[str]:
    model_list = [model_data["name"] for model_data in _get_model_data(auth_spec, mode)]
    if len(model_list) == 0:
        model_list.append("")
    else:
        model_list.sort()
        return model_list


def _get_model_data(auth_spec, mode: str):
    api_base = _extract_api_base(auth_spec)
    model_info = api_base + "/management/models?mode=" + mode
    response = requests.get(
        url=model_info, headers=_create_authorization_headers(auth_spec)
    )
    if response.status_code == 404:
        raise ValueError(
            "The AI proxy is not reachable. Is it activated in the connected KNIME Hub?"
        )
    response.raise_for_status()
    return response.json()["models"]
