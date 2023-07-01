# TODO: Node idea: Re implement the Model List Retriever for better usability (for customers with data apps e.g. to select the model there)

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
)

# Langchain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# Other imports
import re
import openai

# TODO: Get someone to do new icons
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


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing the API key",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


@knext.parameter_group(label="Model Settings")
class GeneralSettings:

    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        What sampling temperature to use, between 0 and 1. 
        Higher values like 0.8 will make the output more random, 
        while lower values like 0.2 will make it more focused and deterministic.
        
        We generally recommend altering this or top_p but not both.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
    )

    top_p = knext.DoubleParameter(
        label="top_p",
        description="""
        An alternative to sampling with temperature, 
        called nucleus sampling, where the model considers 
        the results of the tokens with top_p probability mass. 
        So 0.1 means only the tokens comprising the top 10% 
        probability mass are considered.

        We generally recommend altering this or temperature but not both.
        """,
        default_value=0.1,
        min_value=0.01,
        max_value=1.0,
    )

    max_tokens = knext.IntParameter(
        label="Max tokens",
        description="""
        The maximum number of tokens to generate in the completion.

        The token count of your prompt plus 
        max_tokens cannot exceed the model's context length.
        """,
        default_value=512,
        min_value=1,
    )

    n = knext.IntParameter(
        label="n",
        description="""
        How many chat completion choices to generate for each input message.
        This parameter generates many completions and
        can quickly consume your token quota. 
        """,
        default_value=1,
        min_value=1,
    )


def get_model_list(ctx: knext.DialogCreationContext):
    for spec in ctx.get_input_specs():
        if isinstance(spec, OpenAIAuthenticationPortObjectSpec):
            auth_spec = spec

    if not auth_spec:
        raise ValueError("No OpenAI Authentication provided")

    openai.api_key = ctx.get_credentials(auth_spec.credentials).password

    model_list = ["unselected"]
    for model in openai.Model.list()["data"]:
        # If model has '-<number>' string or is not owned_by openai
        if re.search(r"-\d", model["id"]) or "openai" not in model["owned_by"]:
            model_list.append(model["id"])

    model_list.sort()
    return model_list


specific_model_name = knext.StringParameter(
    label="Specific Model ID",
    description="""Select from a list of all available OpenAI models.
        The model chosen has to match the type that the node wants to instantiate.
        """,
    choices=lambda c: get_model_list(c),
    default_value="unselected",
    is_advanced=True,
)


@knext.parameter_group(label="OpenAI LLM Settings")
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
        "GPT-3 (Ada, Babbage, Curie) and GPT-3.5 (Davinci) models",
        OpenAIModelCompletionsOptions.Ada.name,
        OpenAIModelCompletionsOptions,
    )

    specific_model_name = specific_model_name


@knext.parameter_group(label="OpenAI Chat Model Settings")
class ChatModelLoaderInputSettings:
    class OpenAIModelCompletionsOptions(knext.EnumParameterOptions):
        Turbo = (
            "gpt-3.5-turbo",
            "Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. Will be updated with our latest model iteration 2 weeks after it is released",
        )
        Turbo_16k = (
            "gpt-3.5-turbo-16k",
            "Same capabilities as the standard gpt-3.5-turbo model but with 4 times the context.",
        )
        GPT_4 = (
            "gpt-4",
            "More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with our latest model iteration 2 weeks after it is released.",
        )
        GPT_4_32k = (
            "gpt-4-32k",
            "Same capabilities as the base gpt-4 mode but with 4x the context length. Will be updated with our latest model iteration.",
        )

    model_name = knext.EnumParameter(
        "Model name",
        "GPT-3.5 turbo, GPT-3.5 turbo 16k, GPT-4, GPT-4 32k",
        OpenAIModelCompletionsOptions.Turbo.name,
        OpenAIModelCompletionsOptions,
    )

    specific_model_name = specific_model_name


@knext.parameter_group(label="OpenAI Embeddings Configuration")
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
        "Embeddings model name",
        "Ada text embedding models",
        OpenAIEmbeddingsOptions.Ada1.name,
        OpenAIEmbeddingsOptions,
    )

    specific_model_name = specific_model_name


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


# TODO: Better node description text
@knext.node(
    "OpenAI Authenticator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.output_port(
    "OpenAI Authentication",
    "Successful authentication to OpenAI.",
    openai_authentication_port_type,
)
class OpenAIAuthenticator:
    """
    Authenticates the OpenAI API Key.

    This node makes a call to openai.Model.list() to check
    whether the provided API key is valid.
    """

    credentials_settings = CredentialsSettings()

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> OpenAIAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            raise ValueError("No credentials provided to node.")

        if not self.credentials_settings.credentials_param:
            raise ValueError("No credentials selected.")

        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> OpenAIAuthenticationPortObject:
        try:
            openai.api_key = ctx.get_credentials(
                self.credentials_settings.credentials_param
            ).password

            openai.Model.list()
        except:
            raise ValueError("Wrong API key provided")

        return OpenAIAuthenticationPortObject(self.create_spec())

    def create_spec(self):
        return OpenAIAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param
        )


# TODO: Better node description text
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
    "Successfull authentication to OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI LLM",
    "Connection to a specific LLM from OpenAI.",
    openai_llm_port_type,
)
class OpenAILLMConnector:
    """
    Connects to an OpenAI Large Language Model.

    Given the successfull authentication through the OpenAI Authenticator Node,
    choose one of the available LLMs from either a predefined list or through
    the advanced options from all available models that are available
    for the given connection.
    """

    input_settings = LLMLoaderInputSettings()

    model_settings = GeneralSettings()

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

        LOGGER.info(f"Connecting to {model_name}")

        return OpenAILLMPortObjectSpec(
            openai_auth_spec.credentials,
            model_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.n,
        )


# TODO: Better node description text
@knext.node(
    "OpenAI Chat Model Connector",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Authentication",
    "Successfull authentication to OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI Chat Model",
    "Connection to a specific Chat Model from OpenAI.",
    openai_chat_port_type,
)
class OpenAIChatModelConnector:
    """
    Connect to an OpenAI Chat Model

    Given the successfull authentication through the OpenAI Authenticator Node,
    choose one of the available chat models from either a predefined list or through
    the advanced options from all available models that are available
    for the given connection.
    """

    input_settings = ChatModelLoaderInputSettings()
    model_settings = GeneralSettings()

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

        LOGGER.info(f"Connecting to {model_name}")

        return OpenAIChatModelPortObjectSpec(
            openai_auth_spec.credentials,
            model_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.n,
        )


# TODO: Better node description text
@knext.node(
    "OpenAI Embeddings Connector",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Authentication",
    "Successfull authentication to OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI Embeddings Model",
    "Connection to a specific Embeddings Model from OpenAI.",
    openai_embeddings_port_type,
)
class OpenAIEmbeddingsConnector:
    """
    Connect to an OpenAI Embedding Model

    Given the successfull authentication through the OpenAI Authenticator Node,
    choose one of the available embedding models from either a predefined list or through
    the advanced options from all available models that are available
    for the given connection.
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

        LOGGER.info(f"Connecting to {model_name}")

        return OpenAIEmbeddingsPortObjectSpec(openai_auth_spec.credentials, model_name)
