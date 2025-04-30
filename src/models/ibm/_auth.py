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
from ._base import ibm_watsonx_icon, ibm_watsonx_category
from ..base import CredentialsSettings, AIPortObjectSpec
from ._util import (
    _default_ibm_api_base,
    IBMwatsonxConnectionSettings,
    ProjectOrSpaceSelection,
    get_project_or_space,
    ProjectOrSpace,
)


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

    def get_api_key(self, ctx: knext.ExecutionContext) -> str:
        api_key = ctx.get_credentials(self.credentials).password
        return api_key

    def validate_context(self, ctx: knext.ConfigurationContext):
        if self.credentials not in ctx.get_credential_names():
            raise knext.InvalidParametersError(
                f"""The selected credentials '{self.credentials}' holding the IBM watsonx.ai API key do not exist. 
                Make sure that you have selected the correct credentials and that they are still available."""
            )
        api_key = ctx.get_credentials(self.credentials)
        if not api_key.password:
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
            self.get_chat_model_list(ctx)
        except Exception as e:
            raise RuntimeError(
                "Could not authenticate with the IBM watsonx.ai API."
            ) from e

    def model_supports_tools(
        self, model: str, ctx: knext.ConfigurationContext | knext.ExecutionContext
    ) -> bool:
        """
        Check if the model supports tools.
        This is a placeholder implementation and should be overridden by subclasses.
        """

        api_client = self._get_api_client(ctx)

        chat_models = (
            api_client.foundation_models.get_chat_function_calling_model_specs()
        )

        return any(
            chat_model["model_id"] == model for chat_model in chat_models["resources"]
        )

    def _get_api_client(self, ctx: knext.ConfigurationContext | knext.ExecutionContext):
        from ibm_watsonx_ai import APIClient
        from ibm_watsonx_ai import Credentials

        api_key = ctx.get_credentials(self.credentials).password
        base_url = self.base_url

        credentials = Credentials(
            url=base_url,
            api_key=api_key,
        )

        api_client = APIClient(credentials)
        return api_client

    def get_chat_model_list(
        self,
        ctx: knext.ConfigurationContext | knext.ExecutionContext,
    ) -> list[str]:
        """
        Get the list of available chat models from the IBM watsonx.ai API.
        """

        api_client = self._get_api_client(ctx)

        chat_models = api_client.foundation_models.get_chat_model_specs()

        return [model["model_id"] for model in chat_models["resources"]]

    def get_embedding_model_list(
        self,
        ctx: knext.ConfigurationContext | knext.ExecutionContext,
    ) -> list[str]:
        """
        Get the list of available embedding models from the IBM watsonx.ai API.
        """

        api_client = self._get_api_client(ctx)

        embedding_models = api_client.foundation_models.get_embeddings_model_specs()

        return [model["model_id"] for model in embedding_models["resources"]]

    def _map_names_to_ids(
        self,
        ctx: knext.ConfigurationContext | knext.ExecutionContext,
    ) -> str:
        """
        Get the ID for the given name from the list of available projects or spaces.
        The list of available projects or spaces is retrieved from the IBM watsonx.ai API.
        """

        api_client = self._get_api_client(ctx)

        name = self.project_or_space.name
        type = self.project_or_space.type

        items = (
            api_client.projects.list()
            if type == ProjectOrSpaceSelection.PROJECT.name
            else api_client.spaces.list()
        )

        for item_name, item_id in zip(items["NAME"], items["ID"]):
            if item_name == name:
                # found the name, return the ID
                return item_id
        # if the name is not found, raise an error
        raise knext.InvalidParametersError(
            f"'{name}' was not found in the list of available projects or spaces. "
            "Please make sure the name is correct."
        )

    def get_project_or_space_ids(self, ctx) -> tuple[str | None, str | None]:
        """
        Returns a tuple of (project_id, space_id) based on the selected type and name.
        If the type is PROJECT, project_id is set to id and space_id is None.
        If the type is SPACE, space_id is set to id and project_id is None.
        """

        project_or_space = self.project_or_space
        type = project_or_space.type

        id = self._map_names_to_ids(ctx)

        project_id = id if type == ProjectOrSpaceSelection.PROJECT.name else None
        space_id = id if type == ProjectOrSpaceSelection.SPACE.name else None

        return project_id, space_id

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
    "IBM watsonx.ai Authentication",
    "Authentication for the IBM watsonx.ai API.",
    ibm_watsonx_auth_port_type,
)
class IBMwatsonxAuthenticator:
    """Authenticates with IBM watsonx.ai via API key.

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
