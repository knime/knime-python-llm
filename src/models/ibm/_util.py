import knime.extension as knext
from dataclasses import dataclass

_default_ibm_api_base = "https://eu-de.ml.cloud.ibm.com"


@dataclass
class ProjectOrSpace:
    """
    Project or space name and type.
    """

    name: str
    type: str

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            name=data["name"],
            type=data["type"],
        )


class ProjectOrSpaceSelection(knext.EnumParameterOptions):
    PROJECT = (
        "Project",
        "watsonx.ai Studio project to run the selected model.",
    )
    SPACE = (
        "Space",
        "watsonx.ai Studio space to run the selected model.",
    )


@knext.parameter_group(label="Connection configuration")
class IBMwatsonxConnectionSettings:

    project_or_space = knext.EnumParameter(
        "Project or space",
        """
        To provide context for chat completion or to use an embedding model, you must select a project or space.
        You can manage your projects and spaces in [watsonx.ai Studio](https://dataplatform.cloud.ibm.com/wx/home).
        """,
        default_value=ProjectOrSpaceSelection.PROJECT.name,
        enum=ProjectOrSpaceSelection,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    project_name = knext.StringParameter(
        "Project name",
        "Watson Studio project name to run the selected model.",
        default_value="",
    ).rule(
        knext.OneOf(project_or_space, [ProjectOrSpaceSelection.PROJECT.name]),
        knext.Effect.SHOW,
    )

    space_name = knext.StringParameter(
        "Space name",
        "Watson Studio space name to run the selected model",
        default_value="",
    ).rule(
        knext.OneOf(project_or_space, [ProjectOrSpaceSelection.SPACE.name]),
        knext.Effect.SHOW,
    )

    base_url = knext.StringParameter(
        "Base URL",
        """
        The base URL for the IBM watsonx API.
        Refer to the [IBM watsonx API documentation](https://cloud.ibm.com/apidocs/watsonx-ai#endpoint-url)
        to view the list of available base URLs.
        """,
        default_value=_default_ibm_api_base,
        is_advanced=True,
    )

    validate_api_key = knext.BoolParameter(
        "Validate API key",
        "If set, the API key is validated during execution by fetching the available models.",
        True,
        is_advanced=True,
    )


def list_models(ctx: knext.ConfigurationContext):
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_model_list(ctx)
    return []


def check_model(model_id: str) -> None:
    if not model_id:
        raise knext.InvalidParametersError("Select a chat model.")


def get_project_or_space(
    connection_settings: IBMwatsonxConnectionSettings,
) -> ProjectOrSpace:
    """
    Get the project or space name and type.
    """

    type = connection_settings.project_or_space

    type_to_name_mapping = {
        ProjectOrSpaceSelection.PROJECT.name: connection_settings.project_name,
        ProjectOrSpaceSelection.SPACE.name: connection_settings.space_name,
    }

    name = type_to_name_mapping.get(type)

    return ProjectOrSpace(
        name=name,
        type=type,
    )
