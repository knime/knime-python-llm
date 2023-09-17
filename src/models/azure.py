# KNIME / own imports
import knime.extension as knext
from .base import model_category, CredentialsSettings

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

# Langchain imports
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


# Other imports
import openai
from socket import gaierror

azure_icon = "icons/azure_logo.png"
azure_openai_category = knext.category(
    path=model_category,
    level_id="azure",
    name="Azure OpenAI",
    description="Contains nodes for connecting to Azure OpenAI.",
    icon=azure_icon,
)

# This logger is necessary
import logging

LOGGER = logging.getLogger(__name__)

# == Port Objects ==


class AzureOpenAIAuthenticationPortObjectSpec(OpenAIAuthenticationPortObjectSpec):
    def __init__(
        self,
        credentials: str,
        api_base: str,
        api_version: str,
        api_type: str,
    ) -> None:
        super().__init__(credentials)
        self._api_base = api_base
        self._api_version = api_version
        self._api_type = api_type

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def api_version(self) -> str:
        return self._api_version

    @property
    def api_type(self) -> str:
        return self._api_type

    def serialize(self) -> dict:
        return {
            **super().serialize(),
            "api_base": self._api_base,
            "api_version": self._api_version,
            "api_type": self._api_type,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["credentials"], data["api_base"], data["api_version"], data["api_type"]
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
    def __init__(self, azure_auth_spec=AzureOpenAIAuthenticationPortObjectSpec) -> None:
        self._credentials = azure_auth_spec

    @property
    def api_base(self) -> str:
        return self._credentials._api_base

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
        model_name,
        temperature,
        top_p,
        max_tokens,
        n,
    ) -> None:
        super().__init__(credentials)
        self._model = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._n = n

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            AzureOpenAIAuthenticationPortObjectSpec.deserialize(data),
            data["model"],
            data["temperature"],
            data["top_p"],
            data["max_tokens"],
            data["n"],
        )


class AzureOpenAILLMPortObject(OpenAILLMPortObject):
    def __init__(self, spec: AzureOpenAILLMPortObjectSpec):
        super().__init__(spec)

    def create_model(self, ctx):

        return AzureOpenAI(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            openai_api_version=self.spec.api_version,
            openai_api_base=self.spec.api_base,
            openai_api_type=self.spec.api_type,
            deployment_name=self.spec.model,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            max_tokens=self.spec.max_tokens,
            n=self.spec.n,
        )


azure_openai_llm_port_type = knext.port_type(
    "Azure OpenAI LLM", AzureOpenAILLMPortObject, AzureOpenAILLMPortObjectSpec
)


class AzureOpenAIChatModelPortObjectSpec(
    AzureOpenAIModelPortObjectSpec, OpenAIChatModelPortObjectSpec
):
    def __init__(
        self,
        credentials: AzureOpenAIAuthenticationPortObjectSpec,
        model_name,
        temperature,
        top_p,
        max_tokens,
        n,
    ) -> None:
        super().__init__(credentials)
        self._model = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._n = n

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            AzureOpenAIAuthenticationPortObjectSpec.deserialize(data),
            data["model"],
            data["temperature"],
            data["top_p"],
            data["max_tokens"],
            data["n"],
        )


class AzureOpenAIChatModelPortObject(OpenAIChatModelPortObject):
    def __init__(self, spec: AzureOpenAIChatModelPortObjectSpec):
        super().__init__(spec)

    def create_model(self, ctx):

        model_kwargs = {"top_p": self.spec.top_p}

        return AzureChatOpenAI(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            openai_api_version=self.spec.api_version,
            openai_api_base=self.spec.api_base,
            openai_api_type=self.spec.api_type,
            deployment_name=self.spec.model,
            temperature=self.spec.temperature,
            model_kwargs=model_kwargs,
            max_tokens=self.spec.max_tokens,
            n=self.spec.n,
        )


azure_openai_chat_port_type = knext.port_type(
    "Azure OpenAI Chat Model",
    AzureOpenAIChatModelPortObject,
    AzureOpenAIChatModelPortObjectSpec,
)


class AzureOpenAIEmbeddingsPortObjectSpec(
    AzureOpenAIModelPortObjectSpec, OpenAIEmbeddingsPortObjectSpec
):
    def __init__(
        self, credentials: AzureOpenAIAuthenticationPortObjectSpec, model_name
    ) -> None:
        super().__init__(credentials)
        self._model = model_name

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            AzureOpenAIAuthenticationPortObjectSpec.deserialize(data),
            data["model"],
        )


class AzureOpenAIEmbeddingsPortObject(OpenAIEmbeddingsPortObject):
    def __init__(self, spec: AzureOpenAIEmbeddingsPortObjectSpec):
        super().__init__(spec)

    def create_model(self, ctx):
        return OpenAIEmbeddings(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            openai_api_base=self.spec.api_base,
            openai_api_version=self.spec.api_version,
            openai_api_type=self.spec.api_type,
            deployment=self.spec.model,
            chunk_size=16,  # Azure only supports 16 docs per request
        )


azure_openai_embeddings_port_type = knext.port_type(
    "Azure OpenAI Embeddings Model",
    AzureOpenAIEmbeddingsPortObject,
    AzureOpenAIEmbeddingsPortObjectSpec,
)


@knext.parameter_group(label="Azure Connection")
class AzureSettings:
    api_base = knext.StringParameter(
        label="Azure Resource Endpoint",
        description="""The Azure OpenAI Resource Endpoint e.g. https://<myResource>.openai.azure.com/ which can be
        found on the [Azure Portal](https://portal.azure.com/)
        """,
        default_value="",
    )

    api_version = knext.StringParameter(
        label="Azure API Version",
        description="""The API version you want to use. Note that the latest API versions could support more functionality such
        as function calling. Find the available API versions here:
        [API versions](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#completions)""",
        default_value="2023-07-01-preview",
    )


@knext.parameter_group(label="Azure Deployment")
class AzureDeploymentSettings:
    deployment_name = knext.StringParameter(
        label="Deployment name",
        description="""The name of the deployed model to use. Find the deployed models on the [Azure AI Studio](https://oai.azure.com).""",
        default_value="",
    )


# == Nodes ==


@knext.node(
    "Azure OpenAI Authenticator",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
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
    Credentials can be set on the workflow level or created inside the workflow e.g. with the 
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and fed into this node via flow variable.

    To find your Azure OpenAI API key, navigate to your Azure OpenAI Resource on the [Azure Portal](https://portal.azure.com/) and copy one of the keys and the endpoint from
    'Resource Management - Keys and Endpoints'.

    [Available API versions](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#completions)""",

    credentials_settings = CredentialsSettings(
        label="Azure OpenAI API Key",
        description="""
        The credentials containing the OpenAI API key in its *password* field (the *username* is ignored).
        """,
    )

    azure_connection = AzureSettings()

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> AzureOpenAIAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        if not self.azure_connection.api_base:
            raise knext.InvalidParametersError("API endpoint not provided.")

        spec = self.create_spec()
        spec.validate_context(ctx)
        return spec

    def execute(
        self, ctx: knext.ExecutionContext
    ) -> AzureOpenAIAuthenticationPortObject:
        try:
            openai.Model.list(
                api_key=ctx.get_credentials(
                    self.credentials_settings.credentials_param
                ).password,
                api_version=self.azure_connection.api_version,
                api_base=self.azure_connection.api_base,
                api_type="azure",
            )

        except gaierror:
            raise knext.InvalidParametersError("API endpoint not valid.")
        except openai.error.InvalidRequestError:
            raise knext.InvalidParametersError(
                "Wrong resource endpoint or API version provided."
            )
        except openai.error.AuthenticationError:
            raise knext.InvalidParametersError("Access denied, wrong API key.")

        return AzureOpenAIAuthenticationPortObject(self.create_spec())

    def create_spec(self) -> AzureOpenAIAuthenticationPortObjectSpec:
        return AzureOpenAIAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param,
            self.azure_connection.api_base,
            self.azure_connection.api_version,
            "azure",
        )


@knext.node(
    "Azure OpenAI LLM Connector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI LLM",
    "Configured OpenAI LLM connection.",
    azure_openai_llm_port_type,
)
class AzureOpenAILLMConnector:
    """
    Connects to an Azure OpenAI Large Language Model.

    This node establishes a connection with an Azure OpenAI Large Language Model (LLM).
    After successfully authenticating using the **Azure OpenAI Authenticator node**, enter the deployment name of
    the model you want to use. You can find the models on the [Azure AI Studio](https://oai.azure.com) at
    'Management - Deployments'. Note that only models compatible with Azure OpenAI's Completions API will work with this node.

    If you a looking for gpt-3.5-turbo (the model behind ChatGPT) or gpt-4, check out the **Azure OpenAI Chat Model Connector** node.
    """

    deployment = AzureDeploymentSettings()
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

        LOGGER.info(f"Selected model: {deployment_name}")

        return AzureOpenAILLMPortObjectSpec(
            azure_auth_spec,
            deployment_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.n,
        )


@knext.node(
    "Azure OpenAI Chat Model Connector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI Chat Model",
    "Configured Azure OpenAI Chat Model connection.",
    azure_openai_chat_port_type,
)
class AzureOpenAIChatModelConnector:
    """
    Connects to an Azure OpenAI Chat Model.

    This node establishes a connection with an Azure OpenAI Chat Model.
    After successfully authenticating using the **Azure OpenAI Authenticator node**, enter the deployment name of
    the model you want to use. You can find the models on the [Azure AI Studio](https://oai.azure.com) at
    'Management - Deployments'.

    Note that chat models can also be used as LLMs because they are actually a subcategory of LLMs that are optimized
    for chat-like applications.
    """

    deployment = AzureDeploymentSettings()
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

        LOGGER.info(f"Selected model: {deployment_name}")

        return AzureOpenAIChatModelPortObjectSpec(
            azure_openai_auth_spec,
            deployment_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.n,
        )


@knext.node(
    "Azure OpenAI Embeddings Connector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI Embeddings Model",
    "Configured Azure OpenAI Embeddings Model connection.",
    azure_openai_embeddings_port_type,
)
class AzureOpenAIEmbeddingsConnector:
    """
    Connects to an Azure OpenAI Embeddings Model.

    This node establishes a connection with an Azure OpenAI Embeddings Model. After successfully authenticating
    using the **Azure OpenAI Authenticator node**, you need to provide the name of a deployed embeddings model
    found on the [Azure AI Studio](https://oai.azure.com).
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

        LOGGER.info(f"Selected model: {deployment_name}")

        return AzureOpenAIEmbeddingsPortObjectSpec(
            azure_openai_auth_spec, deployment_name
        )
