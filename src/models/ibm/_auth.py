import knime.extension as knext
from ._base import ibm_watsonx_icon, ibm_watsonx_category
from ..base import CredentialsSettings, AIPortObjectSpec
from ._util import (
    _default_ibm_api_base,
    IBMwatsonxConnectionSettings,
    ProjectOrSpaceSelection,
    get_project_or_space,
    ProjectOrSpace,
)
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials


class IBMwatsonxAuthenticationPortObjectSpec(AIPortObjectSpec):
    def __init__(
        self,
        credentials: str,
        project_or_space: ProjectOrSpace,
        base_url: str = _default_ibm_api_base,
    ) -> None:
        super().__init__()
        self._credentials = credentials
        self._project_or_space = project_or_space
        self._base_url = base_url

    @property
    def credentials(self) -> str:
        return self._credentials

    @property
    def project_or_space(self) -> ProjectOrSpace:
        return self._project_or_space

    @property
    def base_url(self) -> str:
        return self._base_url

    def validate_context(self, ctx: knext.ConfigurationContext):
        if self.credentials not in ctx.get_credential_names():
            raise knext.InvalidParametersError(
                f"""The selected credentials '{self.credentials}' holding the IBM watsonx.ai API key do not exist. 
                Make sure that you have selected the correct credentials and that they are still available."""
            )
        api_token = ctx.get_credentials(self.credentials)
        if not api_token.password:
            raise knext.InvalidParametersError(
                f"""The IBM watsonx.ai API key '{self.credentials}' does not exist. Make sure that the node you are using to pass the credentials 
                (e.g. the Credentials Configuration node) is still passing the valid API key as a flow variable to the downstream nodes."""
            )

        if not self.base_url:
            raise knext.InvalidParametersError("Please provide a base URL.")

        if not self.project_or_space.name:
            raise knext.InvalidParametersError(
                "Please provide a project or space name."
            )

    def validate_api_key(self, ctx: knext.ExecutionContext):
        try:
            self._get_models_from_api(ctx)
        except Exception as e:
            raise RuntimeError(
                "Could not authenticate with the IBM watsonx.ai API."
            ) from e

    def _get_models_from_api(
        self, ctx: knext.ConfigurationContext | knext.ExecutionContext
    ) -> list[str]:

        api_key = ctx.get_credentials(self.credentials).password
        base_url = self.base_url

        credentials = Credentials(
            url=base_url,
            api_key=api_key,
        )

        api_client = APIClient(credentials)
        return api_client

    def _get_models_from_api(
        self,
        ctx: knext.ConfigurationContext | knext.ExecutionContext,
    ) -> list[str]:
        """
        Get the list of available chat models from the IBM watsonx.ai API.
        """

        api_client = self._get_api_client(ctx)

        chat_models = api_client.foundation_models.get_chat_model_specs()

        return [model["model_id"] for model in chat_models["resources"]]

    def get_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        return self._get_models_from_api(ctx)

    def _map_names_to_ids(
        self,
        ctx: knext.ConfigurationContext | knext.ExecutionContext,
        name,
        type,
    ) -> dict[str, str]:
        """
        Get the ID for the given name from the list of available projects or spaces.
        The list of available projects or spaces is retrieved from the IBM watsonx.ai API.
        """

        api_client = self._get_api_client(ctx)

        items = (
            api_client.projects.list()
            if type == ProjectOrSpaceSelection.PROJECT.name
            else api_client.spaces.list()
        )

        # build a name-to-ID mapping
        name_to_id_map = dict(zip(items["NAME"], items["ID"]))

        # validate and retrieve the target ID
        try:
            return name_to_id_map[name]
        except KeyError:
            raise knext.InvalidParametersError(
                f"'{name}' was not found in the list of available projects or spaces. "
                "Please make sure the name is correct."
            )

    def get_map_names_to_ids(
        self,
        ctx: knext.ConfigurationContext | knext.ExecutionContext,
        name: str,
        type: str,
    ) -> dict[str, str]:
        return self._map_names_to_ids(ctx, name, type)

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
            "project_or_space": self._project_or_space.serialize(),
            "base_url": self._base_url,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["credentials"],
            ProjectOrSpace.deserialize(data["project_or_space"]),
            data.get("base_url", _default_ibm_api_base),
        )


class IBMwatsonxAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: IBMwatsonxAuthenticationPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> IBMwatsonxAuthenticationPortObjectSpec:
        return super().spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: IBMwatsonxAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


ibm_watsonx_auth_port_type = knext.port_type(
    "IBM watsonx.ai Authentication",
    IBMwatsonxAuthenticationPortObject,
    IBMwatsonxAuthenticationPortObjectSpec,
)


@knext.node(
    name="IBM watsonx.ai Authenticator",
    node_type=knext.NodeType.SOURCE,
    icon_path=ibm_watsonx_icon,
    category=ibm_watsonx_category,
    keywords=["IBM", "watsonx", "GenAI"],
)
@knext.output_port(
    "IBM watsonx.ai API Authentication",
    "Authentication for the IBM watsonx.ai API.",
    ibm_watsonx_auth_port_type,
)
class IBMwatsonxAuthenticator:
    """Authenticates with IBM watsonx via API key.

    This nodes authenticates with IBM watsonx.ai via the provided API key. The resulting
    authenticated connection can then be used to select chat and embedding models available
    in your IBM watsonx.ai project using the **IBM watsonx.ai Chat Model Connector** and
    **IBM watsonx.ai Embedding Model Connector** nodes.

    Refer to [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) for details on
    how to create an API key.

    ---

    The API key must be provided to this node as the password of a `credentials` flow variable from the **Credentials
    Configuration** node.
    """

    credentials_settings = CredentialsSettings(
        label="IBM watsonx.ai API key",
        description="The credentials containing the IBM watsonx.ai API key in its *password* field (the *username* is ignored).",
    )

    connection_settings = IBMwatsonxConnectionSettings()

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> IBMwatsonxAuthenticationPortObjectSpec:

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        spec = self.create_spec(ctx)
        spec.validate_context(ctx)

        return spec

    def execute(
        self, ctx: knext.ExecutionContext
    ) -> IBMwatsonxAuthenticationPortObject:
        spec = self.create_spec(ctx)
        if self.connection_settings.validate_api_key:
            spec.validate_api_key(ctx)
        return IBMwatsonxAuthenticationPortObject(spec)

    def create_spec(self, ctx) -> IBMwatsonxAuthenticationPortObjectSpec:
        project_or_space = get_project_or_space(self.connection_settings)

        return IBMwatsonxAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param,
            base_url=self.connection_settings.base_url,
            project_or_space=project_or_space,
        )
