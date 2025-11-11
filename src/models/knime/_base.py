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
import logging

logger = logging.getLogger(__name__)

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


def _extract_hub_service_url(
    auth_spec: ks.HubAuthenticationPortObjectSpec, path: str
) -> str:
    """
    Extract a normalized service URL from the hub URL.
    
    Args:
        auth_spec: Hub authentication specification
        path: Service path (e.g., "ai-gateway/v1", "accounts")
    
    Returns:
        Normalized URL with scheme, netloc, and path
    """
    try:
        validate_auth_spec(auth_spec)
    except knext.InvalidParametersError as ex:
        # ValueError does not add the exception type to the error message in the dialog
        raise ValueError(str(ex))
    hub_url = auth_spec.hub_url
    parsed_url = urlparse(hub_url)
    # drop params, query and fragment
    service_url = (parsed_url.scheme, parsed_url.netloc, path, "", "", "")
    return urlunparse(service_url)


def extract_api_base(auth_spec: ks.HubAuthenticationPortObjectSpec) -> str:
    return _extract_hub_service_url(auth_spec, "ai-gateway/v1")


def _extract_accounts_base(auth_spec: ks.HubAuthenticationPortObjectSpec) -> str:
    """Extract the base URL for the accounts service from the auth spec."""
    return _extract_hub_service_url(auth_spec, "accounts")


def _get_user_team_names(
    auth_spec: ks.HubAuthenticationPortObjectSpec,
) -> dict[str, str]:
    """
    Build a cache mapping team IDs to team names from the user's identity.
    Makes a single API call to get all teams the user is a member of.
    
    Returns:
        Dictionary mapping team ID to team name.
    """
    import requests
    from ._models import AccountIdentityResponse
    
    try:
        accounts_base = _extract_accounts_base(auth_spec)
        identity_url = f"{accounts_base}/identity"
        
        response = requests.get(
            url=identity_url,
            headers=create_authorization_headers(auth_spec)
        )
        response.raise_for_status()
        
        identity = AccountIdentityResponse.model_validate(response.json())
        return {team.id: team.name for team in identity.teams}
    except Exception as ex:
        logger.warning(
            "Failed to fetch team names from identity endpoint: %s. "
            "Team scopes will be displayed with IDs instead of names.",
            ex
        )
        return {}


def _get_team_name_by_id(
    auth_spec: ks.HubAuthenticationPortObjectSpec,
    team_id: str
) -> str | None:
    """
    Fetch team name directly by team ID.
    Returns team name if successful, None otherwise.
    """
    import requests
    from ._models import TeamInfo
    
    try:
        accounts_base = _extract_accounts_base(auth_spec)
        team_url = f"{accounts_base}/{team_id}"
        
        response = requests.get(
            url=team_url,
            headers={
                **create_authorization_headers(auth_spec),
                "Prefer": "representation=minimal"
            }
        )
        response.raise_for_status()
        
        team = TeamInfo.model_validate(response.json())
        return team.name
    except Exception as ex:
        logger.warning(
            "Failed to fetch team name for team ID '%s': %s. "
            "Team scope will be displayed with ID instead of name.",
            team_id,
            ex
        )
        return None


def _fetch_team_names_concurrently(
    auth_spec: ks.HubAuthenticationPortObjectSpec,
    team_ids: list[str]
) -> dict[str, str]:
    """
    Fetch multiple team names concurrently.
    
    Args:
        auth_spec: Hub authentication specification
        team_ids: List of team IDs to fetch
    
    Returns:
        Dictionary mapping team IDs to team names (only successful fetches)
    """
    import concurrent.futures
    
    team_names = {}
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(team_ids))) as executor:
        # Submit all fetch tasks
        future_to_team_id = {
            executor.submit(_get_team_name_by_id, auth_spec, team_id): team_id
            for team_id in team_ids
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_team_id):
            team_id = future_to_team_id[future]
            try:
                team_name = future.result()
                if team_name is not None:
                    team_names[team_id] = team_name
            except Exception as ex:
                logger.warning(
                    "Exception while fetching team name for '%s': %s",
                    team_id,
                    ex
                )
    
    return team_names


def _resolve_scope_display_name_with_cache(
    scope_id: str | None,
    team_name_cache: dict[str, str]
) -> str | None:
    """
    Resolve a scope ID to a human-readable display name using a pre-built cache.
    
    Args:
        auth_spec: Hub authentication specification
        scope_id: Scope identifier to resolve
        team_name_cache: Pre-built mapping of team IDs to team names
    
    Returns:
        - "Global" for hub:global scope or None
        - Team name for account:team:* scopes (from cache)
        - None for unknown/unsupported scope types or if team name cannot be resolved
    """
    if scope_id is None or scope_id == "hub:global":
        return "Global"
    
    if scope_id.startswith("account:team:"):
        # Return cached team name or None if not found (omit scope prefix)
        return team_name_cache.get(scope_id)
    
    # For unknown scope types (e.g., future user scopes), return None
    return None


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


def _build_complete_team_name_cache(
    auth_spec: ks.HubAuthenticationPortObjectSpec,
    models: list,
) -> dict[str, str]:
    """
    Build a complete cache of team names for all models.
    
    Args:
        auth_spec: Hub authentication specification
        models: List of ModelInfo objects
    
    Returns:
        Dictionary mapping team IDs to team names
    """
    # Build a mapping of team IDs to team names from identity (one API call)
    team_name_cache = _get_user_team_names(auth_spec)
    
    # Identify team IDs that are missing from the cache
    missing_team_ids = set()
    for model in models:
        if (
            model.scope_id
            and model.scope_id.startswith("account:team:")
            and model.scope_id not in team_name_cache
        ):
            missing_team_ids.add(model.scope_id)
    
    # Fetch missing team names concurrently
    if missing_team_ids:
        missing_team_names = _fetch_team_names_concurrently(
            auth_spec, list(missing_team_ids)
        )
        team_name_cache.update(missing_team_names)
    
    return team_name_cache


def list_model_choices(auth_spec, mode: str | None = None) -> list[knext.StringParameter.Choice]:
    models = list_models(auth_spec, mode)
    team_name_cache = _build_complete_team_name_cache(auth_spec, models)
    
    # Build choices with resolved scope names
    choices = []
    for model in models:
        # Resolve scope to display name using the cache
        scope_display = _resolve_scope_display_name_with_cache(
            model.scope_id, team_name_cache
        )
        
        # Format label with scope prefix
        if scope_display is not None:
            label = f"{scope_display} » {model.name}"
            # Description: <scope name> » <model name>: <model description>
            description = f"{scope_display} » {model.name}: {model.description}" if model.description else f"{scope_display} » {model.name}"
        else:
            label = model.name
            # Description: <model name>: <model description>
            description = f"{model.name}: {model.description}" if model.description else model.name
        
        choices.append(knext.StringParameter.Choice(model.id, label, description))
    
    # Sort by display label (case-insensitive)
    choices.sort(key=lambda c: c.label.lower())
    
    return choices

def is_available_model(auth_spec, model_id: str, mode: str) -> bool:
    models = list_models(auth_spec, mode)
    allowed_ids = set()
    for model in models:
        allowed_ids.add(model.id)
        # models in hub:global scope can be referenced by name for backwards compatibility
        # None scope_id indicates legacy gateway where all models were hub:global
        if model.scope_id == "hub:global" or model.scope_id is None:
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


def list_models_with_scope_info(auth_spec, mode: str | None = None):
    """
    List models with resolved scope information.
    
    Returns:
        List of tuples containing (model, scope_id, scope_name) where:
        - model: ModelInfo object
        - scope_id: The scope ID (or "hub:global" for legacy None values)
        - scope_name: Resolved human-readable scope name (or None if not resolved)
    """
    models = list_models(auth_spec, mode)
    team_name_cache = _build_complete_team_name_cache(auth_spec, models)
    
    # Build result with resolved scope information
    result = []
    for model in models:
        # Normalize None scope_id to "hub:global" for legacy compatibility
        scope_id = model.scope_id if model.scope_id is not None else "hub:global"
        
        # Resolve scope name (None if it can't be resolved)
        scope_name = _resolve_scope_display_name_with_cache(
            model.scope_id, team_name_cache
        )
        
        result.append((model, scope_id, scope_name))
    
    return result
