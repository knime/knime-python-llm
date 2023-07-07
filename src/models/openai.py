# KNIME / own imports
import knime.extension as knext
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    model_category,
    GeneralSettings,
    CredentialsSettings,
)

# Langchain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# Other imports
import re
import openai

openai_icon = "icons/openai.png"
openai_category = knext.category(
    path=model_category,
    level_id="openai",
    name="OpenAI",
    description="",
    icon=openai_icon,
)

# This logger is necessary
import logging

LOGGER = logging.getLogger(__name__)


# == SETTINGS ==

# @knext.parameter_group(label="Model Settings") -- Imported
class OpenAIGeneralSettings(GeneralSettings):
    max_tokens = knext.IntParameter(
        label="Max tokens",
        description="""
        The maximum number of tokens to generate.

        The token count of your prompt plus 
        max_tokens cannot exceed the model's context length.
        """,
        default_value=50,
        min_value=1,
    )

    n = knext.IntParameter(
        label="Completions generation",
        description="""
        How many chat completion choices to generate for each input message.
        This parameter generates many completions and
        can quickly consume your token quota. 
        """,
        default_value=1,
        min_value=1,
        is_advanced=True,
    )

    max_tokens = knext.IntParameter(
        label="Max tokens",
        description="""
        The maximum number of tokens to generate in the completion.

        The token count of your prompt plus 
        max_tokens cannot exceed the model's context length.

        Max token limits per models can be found [here](https://platform.openai.com/docs/models/overview).
        """,
        default_value=50,
        min_value=0,
    )


def get_model_list(ctx: knext.DialogCreationContext):
    def get_key(specs):
        auth_spec = specs[0] if specs else None
        if (
            auth_spec is not None
            and auth_spec.credentials in ctx.get_credential_names()
        ):
            return ctx.get_credentials(auth_spec.credentials).password
        return None

    model_list = ["unselected"]

    specs = ctx.get_input_specs()
    key = get_key(specs)

    if not key:
        return model_list
    openai.api_key = key

    for model in openai.Model.list()["data"]:
        # If model has '-<number>' string or is not owned_by openai
        if re.search(r"-\d", model["id"]) or "openai" not in model["owned_by"]:
            model_list.append(model["id"])

    model_list.sort()
    return model_list


def _create_specific_model_name(api_name:str ) -> knext.StringParameter:
    return knext.StringParameter(
        label="Specific Model ID",
        description=f"""Select from a list of all available OpenAI models.
            The model chosen has to be compatible with OpenAI's {api_name} API.
            """,
        choices=lambda c: get_model_list(c),
        default_value="unselected",
        is_advanced=True,
    )


@knext.parameter_group(label="OpenAI Model Selection")
class LLMLoaderInputSettings:
    class OpenAIModelCompletionsOptions(knext.EnumParameterOptions):
        Ada = (
            "text-ada-001",
            "Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.",
        )
        Babbage = (
            "text-babbage-001",
            "Capable of straightforward tasks, very fast, and lower cost.",
        )
        Curie = (
            "text-curie-001",
            "Very capable, but faster and lower cost than Davinci.",
        )
        DaVinci2 = (
            "text-davinci-002",
            "Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models.",
        )
        DaVinci3 = (
            "text-davinci-003",
            "Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models.",
        )

    default_model_name = knext.EnumParameter(
        "Model ID",
        "Select the OpenAI model ID to be used.",
        OpenAIModelCompletionsOptions.Ada.name,
        OpenAIModelCompletionsOptions,
    )

    specific_model_name = _create_specific_model_name("Completions")


@knext.parameter_group(label="OpenAI Chat Model Selection")
class ChatModelLoaderInputSettings:
    class OpenAIModelCompletionsOptions(knext.EnumParameterOptions):
        Turbo = (
            "gpt-3.5-turbo",
            """Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003.""",
        )
        Turbo_16k = (
            "gpt-3.5-turbo-16k",
            "Same capabilities as the standard gpt-3.5-turbo model but with 4 times the context.",
        )
        GPT_4 = (
            "gpt-4",
            """More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat.""",
        )
        GPT_4_32k = (
            "gpt-4-32k",
            """Same capabilities as the base gpt-4 mode but with 4x the context length.""",
        )

    model_name = knext.EnumParameter(
        "Model ID",
        "Select the chat-optimized OpenAI model ID to be used.",
        OpenAIModelCompletionsOptions.Turbo.name,
        OpenAIModelCompletionsOptions,
    )

    specific_model_name = _create_specific_model_name("Chat")


@knext.parameter_group(label="OpenAI Embeddings Selection")
class EmbeddingsLoaderInputSettings:
    class OpenAIEmbeddingsOptions(knext.EnumParameterOptions):
        Ada1 = (
            "text-search-ada-doc-001",
            "Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.",
        )
        Ada2 = (
            "text-embedding-ada-002",
            "Capable of straightforward tasks, very fast, and lower cost.",
        )

    model_name = knext.EnumParameter(
        "Model ID",
        "Select the Ada text embedding model to be used.",
        OpenAIEmbeddingsOptions.Ada2.name,
        OpenAIEmbeddingsOptions,
    )

    specific_model_name = _create_specific_model_name("Embeddings")


# == Port Objects ==


class OpenAIAuthenticationPortObjectSpec(knext.PortObjectSpec):
    def __init__(self, credentials) -> None:
        super().__init__()
        self._credentials = credentials

    @property
    def credentials(self):
        return self._credentials

    def serialize(self) -> dict:
        return {"credentials": self._credentials}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"])


class OpenAIAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: OpenAIAuthenticationPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: OpenAIAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


openai_authentication_port_type = knext.port_type(
    "OpenAI Authentication",
    OpenAIAuthenticationPortObject,
    OpenAIAuthenticationPortObjectSpec,
)


class OpenAILLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self, credentials, model_name, temperature, top_p, max_tokens, n
    ) -> None:
        super().__init__()
        self._credentials = credentials
        self._model = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._n = n

    @property
    def credentials(self):
        return self._credentials

    @property
    def model(self):
        return self._model

    @property
    def temperature(self):
        return self._temperature

    @property
    def top_p(self):
        return self._top_p

    @property
    def max_tokens(self):
        return self._max_tokens

    @property
    def n(self):
        return self._n

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
            "model": self._model,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "n": self._n,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["credentials"],
            data["model"],
            data["temperature"],
            data["top_p"],
            data["max_tokens"],
            data["n"],
        )


class OpenAILLMPortObject(LLMPortObject):
    def __init__(self, spec: OpenAILLMPortObjectSpec):
        super().__init__(spec)

    def create_model(self, ctx):
        return OpenAI(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            model=self.spec.model,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            max_tokens=self.spec.max_tokens,
            n=self.spec.n,
        )


openai_llm_port_type = knext.port_type(
    "OpenAI LLM", OpenAILLMPortObject, OpenAILLMPortObjectSpec
)


class OpenAIChatModelPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(
        self, credentials, model_name, temperature, top_p, max_tokens, n
    ) -> None:
        super().__init__()
        self._credentials = credentials
        self._model = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._n = n

    @property
    def credentials(self):
        return self._credentials

    @property
    def model(self):
        return self._model

    @property
    def temperature(self):
        return self._temperature

    @property
    def top_p(self):
        return self._top_p

    @property
    def max_tokens(self):
        return self._max_tokens

    @property
    def n(self):
        return self._n

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
            "model": self._model,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "n": self._n,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["credentials"],
            data["model"],
            data["temperature"],
            data["top_p"],
            data["max_tokens"],
            data["n"],
        )


class OpenAIChatModelPortObject(ChatModelPortObject):
    def __init__(self, spec: OpenAIChatModelPortObjectSpec):
        super().__init__(spec)

    def create_model(self, ctx):
        return ChatOpenAI(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            model=self.spec.model,
        )


openai_chat_port_type = knext.port_type(
    "OpenAI Chat Model", OpenAIChatModelPortObject, OpenAIChatModelPortObjectSpec
)


class OpenAIEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(self, credentials, model_name) -> None:
        super().__init__()
        self._credentials = credentials
        self._model = model_name

    @property
    def credentials(self):
        return self._credentials

    @property
    def model(self):
        return self._model

    def serialize(self) -> dict:
        return {"credentials": self._credentials, "model": self._model}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"], data["model"])


class OpenAIEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(self, spec: EmbeddingsPortObjectSpec):
        super().__init__(spec)

    def create_model(self, ctx):
        return OpenAIEmbeddings(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            model=self.spec.model,
        )


openai_embeddings_port_type = knext.port_type(
    "OpenAI Embeddings Model",
    OpenAIEmbeddingsPortObject,
    OpenAIEmbeddingsPortObjectSpec,
)


# == Nodes ==


@knext.node(
    "OpenAI Authenticator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.output_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
class OpenAIAuthenticator:
    """
    Authenticates the OpenAI API key.

    This node provides the authentication for all OpenAI models.
    It allows you to select the credentials that contain a valid OpenAI API key in their *password* field (the *username* is ignored).
    Credentials can be set on the workflow level or created inside the workflow e.g. with the [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and fed into this node via flow variable.

    When this node is run, it validates the OpenAI key by sending a request to the https://api.openai.com/v1/models endpoint.
    This request does not take up any tokens.

    If you don't have an OpenAI API key yet, you can generate one at
    [OpenAI](https://platform.openai.com/account/api-keys).

    """

    credentials_settings = CredentialsSettings(
        label="OpenAI API Key",
        description="""
        The credentials containing the OpenAI API key in its *password* field (the *username* is ignored).
        """,
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> OpenAIAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> OpenAIAuthenticationPortObject:
        try:
            openai.api_key = ctx.get_credentials(
                self.credentials_settings.credentials_param
            ).password

            openai.Model.list()
        except:
            raise knext.InvalidParametersError("API key is not valid.")

        return OpenAIAuthenticationPortObject(self.create_spec())

    def create_spec(self):
        return OpenAIAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param
        )


# TODO: Check proxy settings and add them to configuration
# TODO: Generate prompts as configuration dialog as seen on langchain llm.generate(["Tell me a joke", "Tell me a poem"]*15)
@knext.node(
    "OpenAI LLM Connector",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI LLM",
    "Configured OpenAI LLM connection.",
    openai_llm_port_type,
)
class OpenAILLMConnector:
    """
    Connects to an OpenAI Large Language Model.

    This node establishes a connection with an OpenAI Large Language Model (LLM).
    After successfully authenticating using the **OpenAI Authenticator node**, you can select an LLM from a predefined list
    or explore advanced options to get a list of all models available for your API key (including fine-tunes).
    Note that only models compatible with OpenAI's Completions API will work with this node (unfortunately this information is not available programmatically).

    If you a looking for gpt-3.5-turbo (the model behind ChatGPT) or gpt-4, check out the **OpenAI Chat Model Connector** node.   
    """

    input_settings = LLMLoaderInputSettings()
    model_settings = OpenAIGeneralSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        openai_auth_spec: OpenAIAuthenticationPortObjectSpec,
    ) -> OpenAILLMPortObjectSpec:
        return self.create_spec(openai_auth_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        openai_auth_port: OpenAIAuthenticationPortObject,
    ) -> OpenAILLMPortObject:
        return OpenAILLMPortObject(self.create_spec(openai_auth_port.spec))

    def create_spec(
        self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec
    ) -> OpenAILLMPortObjectSpec:
        if self.input_settings.specific_model_name != "unselected":
            model_name = self.input_settings.specific_model_name
        else:
            model_name = self.input_settings.OpenAIModelCompletionsOptions[
                self.input_settings.default_model_name
            ].label

        LOGGER.info(f"Connecting to {model_name}...")

        return OpenAILLMPortObjectSpec(
            openai_auth_spec.credentials,
            model_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.n,
        )


@knext.node(
    "OpenAI Chat Model Connector",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI Chat Model",
    "Configured OpenAI Chat Model connection.",
    openai_chat_port_type,
)
class OpenAIChatModelConnector:
    """
    Connects to an OpenAI Chat Model.

    This node establishes a connection with an OpenAI Chat Model. After successfully authenticating
    using the **OpenAI Authenticator node**, you can select a chat model from a predefined list.

    If OpenAI releases a new model that is not among the listed models, you can also select from a list
    of all available OpenAI models but you have to ensure that selected model is compatible with the OpenAI Chat API.

    Note that chat models can also be used as LLMs because they are actually a subcategory of LLMs that are optimized
    for chat-like applications.
    """

    input_settings = ChatModelLoaderInputSettings()
    model_settings = OpenAIGeneralSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        openai_auth_spec: OpenAIAuthenticationPortObjectSpec,
    ) -> OpenAIChatModelPortObjectSpec:
        return self.create_spec(openai_auth_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        openai_auth_port: OpenAIAuthenticationPortObject,
    ) -> OpenAIChatModelPortObject:
        return OpenAIChatModelPortObject(self.create_spec(openai_auth_port.spec))

    def create_spec(
        self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec
    ) -> OpenAIChatModelPortObjectSpec:
        if self.input_settings.specific_model_name != "unselected":
            model_name = self.input_settings.specific_model_name
        else:
            model_name = self.input_settings.OpenAIModelCompletionsOptions[
                self.input_settings.model_name
            ].label

        LOGGER.info(f"Connecting to {model_name}...")

        return OpenAIChatModelPortObjectSpec(
            openai_auth_spec.credentials,
            model_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.n,
        )


@knext.node(
    "OpenAI Embeddings Connector",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI Embeddings Model",
    "Configured OpenAI Embeddings Model connection.",
    openai_embeddings_port_type,
)
class OpenAIEmbeddingsConnector:
    """
    Connects to an OpenAI Embedding Model.

    This node establishes a connection with an OpenAI Embeddings Model. After successfully authenticating
    using the **OpenAI Authenticator node**, you can select an embedding model from a predefined list.

    If OpenAI releases a new embeddings model that is not contained in the predefined list, you can select it from
    the list in the advanced settings which contains all OpenAI models available for your OpenAI API key.
    """

    input_settings = EmbeddingsLoaderInputSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        openai_auth_spec: OpenAIAuthenticationPortObjectSpec,
    ) -> OpenAIEmbeddingsPortObjectSpec:
        return self.create_spec(openai_auth_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        openai_auth_port: OpenAIAuthenticationPortObject,
    ) -> OpenAIEmbeddingsPortObject:
        return OpenAIEmbeddingsPortObject(self.create_spec(openai_auth_port.spec))

    def create_spec(
        self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec
    ) -> OpenAIEmbeddingsPortObjectSpec:
        if self.input_settings.specific_model_name != "unselected":
            model_name = self.input_settings.specific_model_name
        else:
            model_name = self.input_settings.OpenAIEmbeddingsOptions[
                self.input_settings.model_name
            ].label

        LOGGER.info(f"Connecting to {model_name}...")

        return OpenAIEmbeddingsPortObjectSpec(openai_auth_spec.credentials, model_name)
