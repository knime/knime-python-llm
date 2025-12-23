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


# KNIME / own imports
import knime.extension as knext
import knime.extension.nodes as kn
from .base import model_category, CredentialsSettings, OutputFormatOptions

from models.openai import (
    OpenAIGeneralSettings,
    OpenAIAuthenticationPortObjectSpec,
    OpenAIAuthenticationPortObject,
    OpenAIModelPortObjectSpec,
    OpenAILLMPortObjectSpec,
    OpenAILLMPortObject,
    OpenAIChatModelPortObjectSpec,
    OpenAIChatModelPortObject,
    OpenAIEmbeddingsPortObjectSpec,
    OpenAIEmbeddingsPortObject,
)

# Other imports
import logging
from typing import Optional


azure_icon = "icons/azure/azure_logo.png"
azure_openai_category = knext.category(
    path=model_category,
    level_id="azure",
    name="Azure OpenAI",
    description="Contains nodes for connecting to Azure OpenAI.",
    icon=azure_icon,
)

LOGGER = logging.getLogger(__name__)

credentials_port_type_id = "org.knime.credentials.base.CredentialPortObject"

# == Port Objects ==


class AzureOpenAIAuthenticationPortObjectSpec(OpenAIAuthenticationPortObjectSpec):
    def __init__(
        self,
        credentials: str,
        base_url: str,
        api_version: str,
        api_type: str,
        credential_spec: Optional[knext.CredentialPortObjectSpec] = None,
    ) -> None:
        super().__init__(credentials, base_url, credential_spec)
        self._api_version = api_version
        self._api_type = api_type

    @property
    def api_version(self) -> str:
        return self._api_version

    @property
    def api_type(self) -> str:
        return self._api_type

    def serialize(self) -> dict:
        return {
            **super().serialize(),
            "api_version": self._api_version,
            "api_type": self._api_type,
        }

    @classmethod
    def deserialize(cls, data: dict):
        base_url = data["base_url"] if "base_url" in data else data["api_base"]
        serialized_credential_spec = data.get("credential_spec")
        credential_spec = (
            None
            if serialized_credential_spec is None
            else knext.CredentialPortObjectSpec.deserialize(serialized_credential_spec)
        )
        return cls(
            data["credentials"],
            base_url,
            data["api_version"],
            data["api_type"],
            credential_spec,
        )


class AzureOpenAIAuthenticationPortObject(OpenAIAuthenticationPortObject):
    def __init__(self, spec: AzureOpenAIAuthenticationPortObjectSpec) -> None:
        super().__init__(spec)

    @classmethod
    def deserialize(cls, spec: AzureOpenAIAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


azure_openai_authentication_port_type = knext.port_type(
    "Azure OpenAI Authentication",
    AzureOpenAIAuthenticationPortObject,
    AzureOpenAIAuthenticationPortObjectSpec,
)


class AzureOpenAIModelPortObjectSpec(OpenAIModelPortObjectSpec):
    def __init__(
        self, azure_auth_spec: AzureOpenAIAuthenticationPortObjectSpec
    ) -> None:
        self._credentials = azure_auth_spec

    @property
    def api_version(self) -> str:
        return self._credentials._api_version

    @property
    def api_type(self) -> str:
        return self._credentials._api_type

    @classmethod
    def deserialize_credentials_spec(
        cls, data: dict
    ) -> AzureOpenAIAuthenticationPortObjectSpec:
        return AzureOpenAIAuthenticationPortObjectSpec.deserialize(data)


class AzureOpenAILLMPortObjectSpec(
    AzureOpenAIModelPortObjectSpec, OpenAILLMPortObjectSpec
):
    def __init__(
        self,
        credentials: AzureOpenAIAuthenticationPortObjectSpec,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        n_requests: int,
        is_reasoning_model: bool,
    ) -> None:
        super().__init__(credentials)
        self._model = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._seed = seed
        self._n_requests = n_requests
        self._is_reasoning_model = is_reasoning_model

    @property
    def is_reasoning_model(self) -> bool:
        return self._is_reasoning_model

    def serialize(self) -> dict:
        data = super().serialize()
        data["is_reasoning_model"] = self._is_reasoning_model
        return data

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            AzureOpenAIAuthenticationPortObjectSpec.deserialize(data),
            data["model"],
            data["temperature"],
            data["top_p"],
            data["max_tokens"],
            data.get("seed", 0),
            data.get("n_requests", 1),
            data.get("is_reasoning_model", False),
        )


def _create_instruct_model(
    po_instance: "AzureOpenAILLMPortObject",
    ctx: knext.ExecutionContext,
):
    from langchain_openai import AzureOpenAI

    return AzureOpenAI(
        openai_api_key=po_instance.spec.auth_spec.get_api_key(ctx),
        api_version=po_instance.spec.api_version,
        azure_endpoint=po_instance.spec.base_url,
        openai_api_type=po_instance.spec.api_type,
        deployment_name=po_instance.spec.model,
        temperature=po_instance.spec.temperature,
        top_p=po_instance.spec.top_p,
        max_tokens=po_instance.spec.max_tokens,
        seed=po_instance.spec.seed,
    )


def _create_model(
    po_instance: "AzureOpenAIChatModelPortObject",
    ctx: knext.ExecutionContext,
    output_format: OutputFormatOptions = OutputFormatOptions.Text,
):
    from ._azure import _AzureChatOpenAI

    model_kwargs = {}

    # Map max_tokens to max_completion_tokens
    if po_instance.spec.is_reasoning_model:
        model_kwargs["max_completion_tokens"] = po_instance.spec.max_tokens

    # Ignore top_p for reasoning models as it's an unsupported parameter
    if not po_instance.spec.is_reasoning_model:
        model_kwargs["top_p"] = po_instance.spec.top_p

    if output_format == OutputFormatOptions.JSON:
        model_kwargs["response_format"] = {"type": "json_object"}

    return _AzureChatOpenAI(
        openai_api_key=po_instance.spec.auth_spec.get_api_key(ctx),
        openai_api_version=po_instance.spec.api_version,
        azure_endpoint=po_instance.spec.base_url,
        openai_api_type=po_instance.spec.api_type,
        deployment_name=po_instance.spec.model,
        temperature=1.0
        if po_instance.spec.is_reasoning_model
        else po_instance.spec.temperature,
        # For reasoning models, we exclude 'max_tokens' from the request by setting it to None.
        # We also avoid removing the parameter entirely, as older models still require it to be present.
        max_tokens=None
        if po_instance.spec.is_reasoning_model
        else po_instance.spec.max_tokens,
        model_kwargs=model_kwargs,
        seed=po_instance.spec.seed,
    )


class AzureOpenAILLMPortObject(OpenAILLMPortObject):
    def __init__(self, spec: AzureOpenAILLMPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> AzureOpenAILLMPortObjectSpec:
        return super().spec

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        return _create_instruct_model(self, ctx)


azure_openai_llm_port_type = knext.port_type(
    "Azure OpenAI LLM", AzureOpenAILLMPortObject, AzureOpenAILLMPortObjectSpec
)


class AzureOpenAIChatModelPortObjectSpec(
    AzureOpenAILLMPortObjectSpec, OpenAIChatModelPortObjectSpec
):
    """Spec of an Azure OpenAI chat model."""

    @property
    def is_instruct_model(self):
        return False


class AzureOpenAIChatModelPortObject(OpenAIChatModelPortObject):
    @property
    def spec(self) -> AzureOpenAIChatModelPortObjectSpec:
        return super().spec

    def create_model(
        self,
        ctx: knext.ExecutionContext,
        output_format: OutputFormatOptions = OutputFormatOptions.Text,
    ):
        if self.spec.is_instruct_model:
            return _create_instruct_model(self, ctx)

        return _create_model(self, ctx, output_format)


azure_openai_chat_port_type = knext.port_type(
    "Azure OpenAI Chat Model",
    AzureOpenAIChatModelPortObject,
    AzureOpenAIChatModelPortObjectSpec,
)


class AzureOpenAIEmbeddingsPortObjectSpec(
    AzureOpenAIModelPortObjectSpec, OpenAIEmbeddingsPortObjectSpec
):
    def __init__(
        self,
        credentials: AzureOpenAIAuthenticationPortObjectSpec,
        model_name,
    ) -> None:
        super().__init__(credentials)
        self._model = model_name

    def serialize(self) -> dict:
        return {
            **self._credentials.serialize(),
            "model": self._model,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            AzureOpenAIAuthenticationPortObjectSpec.deserialize(data),
            data["model"],
        )


class AzureOpenAIEmbeddingsPortObject(OpenAIEmbeddingsPortObject):
    def __init__(self, spec: AzureOpenAIEmbeddingsPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> AzureOpenAIEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx: knext.ExecutionContext):
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            openai_api_key=self.spec.auth_spec.get_api_key(ctx),
            azure_endpoint=self.spec.base_url,
            api_version=self.spec.api_version,
            openai_api_type=self.spec.api_type,
            deployment=self.spec.model,
            chunk_size=16,  # Azure only supports 16 docs per request
        )


azure_openai_embeddings_port_type = knext.port_type(
    "Azure OpenAI Embedding Model",
    AzureOpenAIEmbeddingsPortObject,
    AzureOpenAIEmbeddingsPortObjectSpec,
)


@knext.parameter_group(label="Azure Connection")
class AzureSettings:
    api_base = knext.StringParameter(
        label="Azure Resource Endpoint",
        description="""The Azure Resource Endpoint address can be found in the 'Keys and Endpoints' 
        section of the [Azure Portal](https://portal.azure.com/).""",
        default_value="",
    )

    api_version = knext.StringParameter(
        label="Azure API Version",
        description="""Available API versions can be found at the [Azure OpenAI API preview lifecycle](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation).
        Note that the latest API versions could support more functionality, such as function calling.""",
        default_value="2023-07-01-preview",
    )


@knext.parameter_group(label="Azure Deployment")
class AzureDeploymentSettings:
    deployment_name = knext.StringParameter(
        label="Deployment name",
        description="""The name of the deployed model to use. Find the deployed models on the [Azure AI Studio](https://oai.azure.com).""",
        default_value="",
    )

class AzureLLMDeploymentSettings(AzureDeploymentSettings):
    deployment_name = AzureDeploymentSettings.deployment_name
    is_reasoning_model = knext.BoolParameter(
        label="The model is a reasoning model",
        description="""
        Reasoning models use fixed settings for certain parameters: "temperature=1.0" and "top_p=1.0". These models do not support custom values for either of these parameters.
        Additionally, reasoning models do not support the standard 'Maximum response length (tokens)' setting unless the "The model is a reasoning model" option is enabled. Once this option is selected, 
        the model will be able to use the maximum token parameter as intended.
        For more information and to verify whether a model is categorized as a reasoning model (e.g. o3, o4-mini and gpt-5), please refer to the [OpenAI Docs](https://platform.openai.com/docs/guides/reasoning?api-mode=responses).
        """,
        default_value=False,
        since_version="5.5.2",
    )


credentials_port_type = kn.get_port_type_for_id(credentials_port_type_id)


def _has_auth_port(ctx: knext.DialogCreationContext) -> bool:
    specs = ctx.get_input_specs()
    return len(specs) == 1 and specs[0] is not None


# region Authenticator
@knext.node(
    "Azure OpenAI Authenticator",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI"],
)
@knext.input_port(
    "Credentials",
    "Credentials for connecting to the Azure OpenAI API.",
    port_type=credentials_port_type,
    optional=True,
)
@knext.output_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
class AzureOpenAIAuthenticator:
    """
    Authenticates the Azure OpenAI API key against the the Cognitive Services account.

    This node provides the authentication for all Azure OpenAI models.
    It allows you to select the credentials that contain a valid Azure OpenAI API key in their *password* field (the *username* is ignored).

    Credentials can be set on the workflow level (right-click the workflow in the KNIME Explorer and click "Workflow Credentials") or created inside the workflow e.g. with the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and fed into this node via flow variable.

    Alternatively, you can access an Azure OpenAI API via OAuth authentication by connecting a credential
    port object to the dynamic input port. For OAuth authentication, the `Azure OpenAI` scope is required.
    The scope can be defined either in the Microsoft Authenticator or the secret store.

    To find your Azure OpenAI API key, navigate to your Azure OpenAI Resource on the [Azure Portal](https://portal.azure.com/) and copy one of the keys from
    'Resource Management - Keys and Endpoints'.

    The Azure Resource Endpoint URL can also be found on the [Azure Portal](https://portal.azure.com/) under 'Resource Management - Keys and Endpoints'.

    For the correct API version, refer to the [Azure OpenAI API preview lifecycle](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation).
    """

    credentials_settings = CredentialsSettings(
        label="Azure OpenAI API Key",
        description="""
        The credentials containing the OpenAI API key in its *password* field (the *username* is ignored).
        Only shown if the optional credential port is not connected.
        """,
    ).rule(knext.DialogContextCondition(_has_auth_port), knext.Effect.HIDE)

    azure_connection = AzureSettings()

    verify_settings = knext.BoolParameter(
        "Verify settings",
        "Whether to verify the settings by calling the *list models* endpoint.",
        True,
        since_version="5.2.1",
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        credential: Optional[knext.CredentialPortObjectSpec],
    ) -> AzureOpenAIAuthenticationPortObjectSpec:
        if credential is None:
            if not ctx.get_credential_names():
                raise knext.InvalidParametersError("Credentials not provided.")

            if not self.credentials_settings.credentials_param:
                raise knext.InvalidParametersError("Credentials not selected.")

        if not self.azure_connection.api_base:
            raise knext.InvalidParametersError("API endpoint not provided.")

        if not self.azure_connection.api_version:
            raise knext.InvalidParametersError("API version not provided.")

        spec = self.create_spec(credential)
        spec.validate_context(ctx)
        return spec

    def execute(
        self, ctx: knext.ExecutionContext, credential: Optional[knext.PortObject]
    ) -> AzureOpenAIAuthenticationPortObject:
        if self.verify_settings:
            self._verify_settings(ctx, None if credential is None else credential.spec)

        return AzureOpenAIAuthenticationPortObject(
            self.create_spec(None if credential is None else credential.spec)
        )

    def _verify_settings(
        self,
        ctx: knext.ExecutionContext,
        credential_spec: Optional[knext.CredentialPortObjectSpec],
    ):
        import openai

        # Get API key from credential spec or credential settings
        if credential_spec is not None:
            try:
                api_key = credential_spec.auth_parameters
            except ValueError:
                raise knext.InvalidParametersError(
                    """The Azure OpenAI API key is not available. Make sure that the node you are using to pass the
                    credential port object is still passing the valid API key to the downstream nodes."""
                )
        else:
            api_key = ctx.get_credentials(
                self.credentials_settings.credentials_param,
            ).password

        try:
            openai.AzureOpenAI(
                api_key=api_key,
                api_version=self.azure_connection.api_version,
                azure_endpoint=self.azure_connection.api_base,
            ).models.list()

        except openai.APIConnectionError:
            raise knext.InvalidParametersError(
                f"Invalid Azure endpoint provided: '{self.azure_connection.api_base}'"
            )
        except openai.NotFoundError:
            raise knext.InvalidParametersError(
                """API resource not found. Please ensure you are using a valid address and API version.
                The address can be found in the 'Keys and Endpoints' section of the Azure Portal at https://portal.azure.com/.
                For the correct API version, refer to the API Version Deprecation Guide at https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation.
                """
            )
        except openai.AuthenticationError:
            raise knext.InvalidParametersError("Invalid API key provided.")

    def create_spec(
        self,
        credential: Optional[knext.CredentialPortObjectSpec],
    ) -> AzureOpenAIAuthenticationPortObjectSpec:
        return AzureOpenAIAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param,
            self.azure_connection.api_base,
            self.azure_connection.api_version,
            "azure",
            credential,
        )


# region Instruct Selector
@knext.node(
    "Azure OpenAI LLM Connector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Large Language Model",
        "OpenAI",
        "Azure",
    ],
    is_deprecated=True,
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI LLM",
    "Configured Azure OpenAI LLM connection.",
    azure_openai_llm_port_type,
)
class AzureOpenAILLMConnector:
    """
    Select an LLM from Azure OpenAI.

    This node establishes a connection with an Azure OpenAI Large Language Model (LLM).
    After successfully authenticating using the **Azure OpenAI Authenticator** node, enter the deployment name of
    the model you want to use. You can find the models on the [Azure AI Studio](https://oai.azure.com) at
    'Management - Deployments'. Note that only models compatible with Azure OpenAI's Completions API will work with this node.

    **Note**: See the **Azure OpenAI LLM Selector** node for LLMs optimized for chat-specific usecases.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    deployment = AzureLLMDeploymentSettings()
    model_settings = OpenAIGeneralSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        azure_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
    ) -> AzureOpenAILLMPortObjectSpec:
        if not hasattr(azure_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use OpenAI Model Connectors")

        if not self.deployment.deployment_name:
            raise knext.InvalidParametersError("No deployment name provided")

        azure_auth_spec.validate_context(ctx)

        return self.create_spec(azure_auth_spec, self.deployment.deployment_name)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        azure_auth_port: AzureOpenAIAuthenticationPortObject,
    ) -> AzureOpenAILLMPortObject:
        # We know that the API key is correct from the authenticator, but a call to
        # AzureOpenAI() does not verify the deployment_name unless it is prompted
        # Could be done with an AD Bearer token

        return AzureOpenAILLMPortObject(
            self.create_spec(azure_auth_port.spec, self.deployment.deployment_name)
        )

    def create_spec(
        self,
        azure_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
        deployment_name: str,
    ) -> AzureOpenAILLMPortObjectSpec:
        return AzureOpenAILLMPortObjectSpec(
            azure_auth_spec,
            deployment_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.seed,
            self.model_settings.n_requests,
        )


# region LLM Selector
@knext.node(
    "Azure OpenAI LLM Selector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI", "Azure"],
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI Large Language Model",
    "Configured Azure OpenAI Large Language Model.",
    azure_openai_chat_port_type,
)
class AzureOpenAIChatModelConnector:
    """
    Select an LLM from Azure OpenAI.

    This node establishes a connection with a Large Language Model (LLM) deployed on Azure OpenAI.
    After successfully authenticating using the **Azure OpenAI Authenticator** node, enter the deployment name of
    the model you want to use. You can find the models on the [Azure AI Studio](https://oai.azure.com) at
    'Management - Deployments'.

    **Note**: Chat models are LLMs that have been fine-tuned for chat-based usecases. As such, these models can also be
    used in other applications as well.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    deployment = AzureLLMDeploymentSettings()
    model_settings = OpenAIGeneralSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        azure_openai_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
    ) -> AzureOpenAIChatModelPortObjectSpec:
        if not hasattr(azure_openai_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use OpenAI Model Connectors")

        if not self.deployment.deployment_name:
            raise knext.InvalidParametersError("No model name provided.")

        azure_openai_auth_spec.validate_context(ctx)

        return self.create_spec(azure_openai_auth_spec, self.deployment.deployment_name)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        azure_openai_auth_port: AzureOpenAIAuthenticationPortObject,
    ) -> AzureOpenAIChatModelPortObject:
        # We know that the API key is correct from the authenticator, but a call to
        # AzureChatOpenAI() does not verify the deployment_name until it is prompted

        return AzureOpenAIChatModelPortObject(
            self.create_spec(
                azure_openai_auth_port.spec, self.deployment.deployment_name
            )
        )

    def create_spec(
        self,
        azure_openai_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
        deployment_name: str,
    ) -> AzureOpenAIChatModelPortObjectSpec:
        return AzureOpenAIChatModelPortObjectSpec(
            azure_openai_auth_spec,
            deployment_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.seed,
            self.model_settings.n_requests,
            self.deployment.is_reasoning_model,
        )


# region Embedding Selector
@knext.node(
    "Azure OpenAI Embedding Model Selector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "OpenAI",
        "Azure",
        "RAG",
        "Retrieval Assisted Generation",
    ],
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI Embedding Model",
    "Configured Azure OpenAI Embedding Model connection.",
    azure_openai_embeddings_port_type,
)
class AzureOpenAIEmbeddingsConnector:
    """
    Select an Azure OpenAI embedding model.

    This node establishes a connection with an Azure OpenAI Embedding Model. After successfully authenticating
    using the **Azure OpenAI Authenticator** node, you need to provide the name of a deployed embedding model
    found on the [Azure AI Studio](https://oai.azure.com).

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    deployment = AzureDeploymentSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        azure_openai_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
    ) -> AzureOpenAIEmbeddingsPortObjectSpec:
        if not hasattr(azure_openai_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use OpenAI Model Connectors")

        if not self.deployment.deployment_name:
            raise knext.InvalidParametersError("No deployment name provided")

        azure_openai_auth_spec.validate_context(ctx)

        return self.create_spec(azure_openai_auth_spec, self.deployment.deployment_name)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        azure_openai_auth_port: AzureOpenAIAuthenticationPortObject,
    ) -> AzureOpenAIEmbeddingsPortObject:
        # We know that the API key is correct from the authenticator, but a call to
        # OpenAIEmeddings() does not verify the deployment_name unless it is prompted

        return AzureOpenAIEmbeddingsPortObject(
            self.create_spec(
                azure_openai_auth_port.spec, self.deployment.deployment_name
            )
        )

    def create_spec(
        self,
        azure_openai_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
        deployment_name: str,
    ) -> AzureOpenAIEmbeddingsPortObjectSpec:
        return AzureOpenAIEmbeddingsPortObjectSpec(
            azure_openai_auth_spec, deployment_name
        )
