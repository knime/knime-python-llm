import logging
from urllib.parse import urlparse, urlunparse, quote

import knime.extension as knext
import knime.api.schema as ks
import requests
import pandas as pd

import util
from models.knime._base import (
    create_authorization_headers,
    validate_auth_spec,
)

tool_icon = "icons/tool.png"
LOGGER = logging.getLogger(__name__)


def workflow_service_choices_provider(ctx: knext.DialogCreationContext) -> list[str]:
    specs = ctx.get_input_specs()
    if specs and (auth_spec := specs[0]):
        data = get_workflow_service_data(auth_spec)
        return get_names(data)
    return []


@knext.parameter_group(label="Workflow Service Selection")
class WorkflowService:
    workflow_service = knext.StringParameter(
        label="Workflow service",
        description="Workflow service to be converted to a tool.",
        choices=workflow_service_choices_provider,
    )

    tool_name = knext.StringParameter(
        label="Tool name",
        description="Name of the tool. If not defined, the workflow service name will be used.",
    )

    tool_description = knext.StringParameter(
        label="Tool description",
        description="Description of the tool.",
    )


@knext.node(
    name="Workflow Service to Tool",
    node_type=knext.NodeType.SOURCE,
    icon_path=tool_icon,
    category=util.main_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "GenAI Gateway"],
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_table(name="Tool", description="Tool description.")
class WorkflowServiceToToolDescription:
    workflow_service_list = knext.ParameterArray(
        label="Workflow Service Selection",
        description="Workflow service that will be converted to a tool.",
        since_version="5.5.0",
        parameters=WorkflowService(),
        button_text="Add workflow service",
        array_title="Workflow service",
    )

    # TODO handle if no service is selected
    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> ks.HubAuthenticationPortObjectSpec:
        validate_auth_spec(authentication)
        return knext.Schema.from_columns([knext.Column(knext.logical(dict), "tools")])

    def execute(
        self,
        ctx: knext.ExecutionContext,
        authentication: knext.PortObject,
    ):
        if not self.workflow_service_list:
            raise knext.InvalidParametersError("No workflow service selected.")

        rows = []
        for wf in self.workflow_service_list:
            openapi_details = get_deployment_openapi_details(
                authentication.spec, wf.workflow_service
            )

            tool_name = wf.workflow_service
            if wf.tool_name:
                tool_name = wf.tool_name

            tool_object = openapi_to_tool_schema(
                tool_name,
                wf.tool_description,
                openapi_details.get("input_args", {}),
            )
            rows.append({"tools": tool_object})

        df = pd.DataFrame(rows, columns=["tools"])
        return knext.Table.from_pandas(df)


def openapi_to_tool_schema(
    tool_name: str, tool_description: str, input_args: dict
) -> dict:
    new_properties = {}
    if isinstance(input_args, dict):
        properties = input_args.get("properties", input_args)
        LOGGER.warn(f"Properties: {properties}")
        for _, field_value in list(properties.items()):
            example = field_value.get("example", {})
            table_spec = example.get("table-spec")
            if table_spec is not None:
                description = field_value.get("description", "")
                for spec in table_spec:
                    for key, type_str in spec.items():
                        prop_name = key.lower()
                        new_properties[prop_name] = {
                            "title": key,
                            "type": type_str,
                        }
    return {
        "title": tool_name,
        "type": "object",
        "description": tool_description if tool_description else description,
        "properties": new_properties,
    }


def get_workflow_service_data(auth_spec: ks.HubAuthenticationPortObjectSpec):
    try:
        validate_auth_spec(auth_spec)
    except knext.InvalidParametersError as ex:
        raise ValueError(str(ex))

    hub_url = auth_spec.hub_url
    parsed_url = urlparse(hub_url)

    account_id_url = urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            "/accounts/accounts/identity",
            "",
            "",
            "",
        )
    )

    team_deployments_url = urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            "/execution/deployments/{scope}/rest",
            "",
            "",
            "",
        )
    )

    shared_deployments_url = urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            "/execution/deployments/shared",
            "",
            "",
            "",
        )
    )

    def fetch_data(url):
        response = requests.get(url, headers=create_authorization_headers(auth_spec))
        response.raise_for_status()
        return response.json()

    account_data = fetch_data(account_id_url)
    teams = account_data.get("teams", [])
    if not teams:
        LOGGER.warning("No teams found in the response.")
        return {}

    teams_by_id = {team["id"]: team for team in teams}

    def fetch_deployments_for_team(team_id):
        url = team_deployments_url.format(scope=team_id)
        data = fetch_data(url)
        return data.get("deployments", [])

    def fetch_shared_deployments():
        data = fetch_data(shared_deployments_url)
        return [
            deployment
            for deployment in data.get("deployments", [])
            if deployment.get("type") == "rest"
        ]

    deployments_by_team = {
        team_id: fetch_deployments_for_team(team_id) for team_id in teams_by_id.keys()
    }
    deployments_by_team["shared"] = fetch_shared_deployments()

    return deployments_by_team


def get_names(deployments_by_team) -> list[str]:
    return [
        service.get("name")
        for services in deployments_by_team.values()
        for service in services
        if service.get("name")
    ]


def get_deployment_openapi_details(
    auth_spec: ks.HubAuthenticationPortObjectSpec, deployment_name: str
):
    try:
        validate_auth_spec(auth_spec)
    except knext.InvalidParametersError as ex:
        raise ValueError(str(ex))

    hub_url = auth_spec.hub_url
    parsed_url = urlparse(hub_url)

    def fetch_openapi_details(deployment_id):
        encoded_id = quote(deployment_id, safe="")
        url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                f"/execution/deployments/{encoded_id}/open-api",
                "",
                "",
                "",
            )
        )
        response = requests.get(url, headers=create_authorization_headers(auth_spec))
        response.raise_for_status()
        return response.json()

    deployment_data = get_workflow_service_data(auth_spec)
    for deployments in deployment_data.values():
        for deployment in deployments:
            if deployment.get("name") == deployment_name:
                deployment_id = deployment.get("id")
                if not deployment_id:
                    LOGGER.warning(
                        f"Deployment ID not found for deployment: {deployment}"
                    )
                    continue
                try:
                    openapi_spec = fetch_openapi_details(deployment_id)
                    input_parameters_schema = (
                        openapi_spec.get("components", {})
                        .get("schemas", {})
                        .get("InputParameters")
                    )
                    output_values_schema = (
                        openapi_spec.get("components", {})
                        .get("schemas", {})
                        .get("OutputValues")
                    )
                    return {
                        "name": deployment.get("name"),
                        "openapi_spec": openapi_spec,
                        "input_args": input_parameters_schema,
                        "output_values": output_values_schema,
                    }
                except requests.exceptions.RequestException as e:
                    LOGGER.warning(
                        f"Failed to fetch details for deployment {deployment_id}: {e}"
                    )
                    continue

    raise ValueError(f"Deployment with name '{deployment_name}' not found.")
