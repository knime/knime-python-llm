import knime.extension.nodes as kn
import knime.extension as knext
from ..base import model_category

databricks_workspace_port_type_id = (
    "org.knime.bigdata.databricks.workspace.port.DatabricksWorkspacePortObject"
)


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


databricks_icon = "icons/databricks.png"

databricks_category = knext.category(
    path=model_category,
    level_id="databricks",
    name="Databricks",
    description="Contains nodes to connect models hosted on Databricks.",
    icon=databricks_icon,
)
