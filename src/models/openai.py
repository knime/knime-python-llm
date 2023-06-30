# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods


from typing import Dict
import knime.extension as knext

from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,

    ChatModelPortObjectSpec,
    ChatModelPortObject,
    chat_model_port_type,

    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,

    model_category
)
import re
import openai

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

openai_icon = "icons/openai.png"
openai_category = knext.category(
    path=model_category,
    level_id="openai",
    name="OpenAI",
    description="",
    icon=openai_icon,
)

import logging

LOGGER = logging.getLogger(__name__)

@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing OpenAI API key",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )



# TODO: Retrieve these Settings from OpenAI with OpenAi Connector

def get_model_list(ctx: knext.DialogCreationContext):

    #TODO: Ask if we need to filter for OpenAIAuthenticationPortObjectSpec
    spec = ctx.get_input_specs()[0]
    credentials = spec.credentials

    openai.api_key = ctx.get_credentials(credentials).password
    model_list = ["unselected"]
    for model in  openai.Model.list()["data"]:
            if re.search(r'-\d', model["id"]) or "openai" not in model["owned_by"]:
                model_list.append(model["id"])
    return model_list

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

    model_name = knext.EnumParameter(
        "Model ID",
        "GPT-3 (Ada, Babbage, Curie) and GPT-3.5 (Davinci) models",
        OpenAIModelCompletionsOptions.Ada.name,
        OpenAIModelCompletionsOptions,
    )

    custom_model_name = knext.StringParameter(
        label="Specific Model ID",
        description="TODO",
        choices=lambda c: get_model_list(c),
        default_value="unselected",
        is_advanced=True,
    )


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

    custom_model_name = knext.StringParameter(
        label="Specific Model ID",
        description="TODO.",
        choices=lambda c: get_model_list(c),
        default_value="unselected",
        is_advanced=True,
    )


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

    custom_model_name = knext.StringParameter(
        label="Specific Model ID",
        description="TODO",
        choices=lambda c: get_model_list(c),
        default_value="unselected",
        is_advanced=True,
    )




############## DONE


class OpenAIAuthenticationPortObjectSpec(knext.PortObjectSpec):
    def __init__(self, credentials) -> None:
        super().__init__()
        self._credentials = credentials

    @property
    def credentials(self):
        return self._credentials

    def serialize(self):
        return {
            "credentials": self._credentials
        }
    
    @classmethod
    def deserialize(cls, data: Dict):
        return cls(data["credentials"])
    
class OpenAIAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: OpenAIAuthenticationPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self):
        return b""
    
    @classmethod
    def deserialize(cls, spec: OpenAIAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)

openai_authentication_port_type = knext.port_type("OpenAI Authentication", OpenAIAuthenticationPortObject, OpenAIAuthenticationPortObjectSpec)

@knext.node(
    "OpenAI Authenticator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.output_port(
    "OpenAI Authentication", 
    "All available OpenAI models by ID.",
    openai_authentication_port_type
)
class OpenAIAuthenticator:
    credentials_settings = CredentialsSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> OpenAIAuthenticationPortObjectSpec:

        return OpenAIAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param
        )

    def execute(self, ctx: knext.ExecutionContext) -> OpenAIAuthenticationPortObject:

        try:
            openai.api_key = ctx.get_credentials(
                self.credentials_settings.credentials_param
            ).password

            openai.Model.list()
        except:
            raise ValueError("Wrong API key provided")

        return OpenAIAuthenticationPortObject(
            OpenAIAuthenticationPortObjectSpec(
                self.credentials_settings.credentials_param
            )
        )


class OpenAILLMPortObjectSpec(LLMPortObjectSpec):
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

class OpenAILLMPortObject(LLMPortObject):
    def __init__(self, spec: OpenAILLMPortObjectSpec):
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""
    
    @classmethod
    def deserialize(cls, spec, data):
        return cls(spec)

    def create_model(self, ctx):
        return OpenAI(
            openai_api_key=ctx.get_credentials(
                self.spec.credentials
            ).password,
            model=self.spec.model,
        )

openai_llm_port_type = knext.port_type("OpenAI LLM", OpenAILLMPortObject, OpenAILLMPortObjectSpec)

class OpenAIChatModelPortObjectSpec(ChatModelPortObjectSpec):
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

class OpenAIChatModelPortObject(ChatModelPortObject):

    def __init__(self, spec: OpenAIChatModelPortObjectSpec):
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""
    
    @classmethod
    def deserialize(cls, spec, data):
        return cls(spec)
    
    def create_model(self, ctx):
        return ChatOpenAI(
            openai_api_key=ctx.get_credentials(
                self.spec.credentials
            ).password,
            model=self.spec.model,
        )

openai_chat_port_type = knext.port_type("OpenAI Chat Model", OpenAIChatModelPortObject, OpenAIChatModelPortObjectSpec)

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

    def serialize(self) -> bytes:
        return b""
    
    @classmethod
    def deserialize(cls, spec, data):
        return cls(spec)
    
    def create_model(self, ctx):
        return OpenAIEmbeddings(
            openai_api_key=ctx.get_credentials(
                self.spec.credentials
            ).password,
            model=self.spec.model,
        )

openai_embeddings_port_type = knext.port_type("OpenAI Embeddings Model", OpenAIEmbeddingsPortObject, OpenAIEmbeddingsPortObjectSpec)

@knext.node(
    "OpenAI LLM Configurator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Authentication", 
    "Credentials to OpenAI.",
    openai_authentication_port_type
)
@knext.output_port(
    "OpenAI LLM Configuration",
    "A large language model configuration for OpenAI.",
    openai_llm_port_type,
)
class OpenAILLMConfigurator:

    input_settings = LLMLoaderInputSettings()

    def configure(
            self, 
            ctx: knext.ConfigurationContext,
            openai_auth_port_spec: OpenAIAuthenticationPortObjectSpec
        ) -> OpenAILLMPortObjectSpec:
        
        return self.create_spec(openai_auth_port_spec)

    def execute(
            self, 
            ctx: knext.ExecutionContext,
            openai_auth_port: OpenAIAuthenticationPortObject
        ) -> OpenAILLMPortObject:

        return OpenAILLMPortObject(
            self.create_spec(openai_auth_port.spec)
        )

    def create_spec(self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec) -> OpenAILLMPortObjectSpec:
        credential_params = openai_auth_spec.credentials
        
        if self.input_settings.custom_model_name != "unselected":
            model_name = self.input_settings.custom_model_name
        else:
            model_name = self.input_settings.OpenAIModelCompletionsOptions[
                self.input_settings.model_name
            ].label

        return OpenAILLMPortObjectSpec(credential_params, model_name)
    


@knext.node(
    "OpenAI ChatModel Configurator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Authentication", 
    "All available OpenAI models by ID.",
    openai_authentication_port_type
)
@knext.output_port(
    "OpenAI Chat Model Configuration",
    "A chat model configuration for OpenAI.",
    openai_chat_port_type,
)
class OpenAIChatModelConfigurator:
    input_settings = ChatModelLoaderInputSettings()

    def configure(self, ctx: knext.ConfigurationContext, openai_auth_port_spec: OpenAIAuthenticationPortObjectSpec) -> OpenAIChatModelPortObjectSpec:
        return self.create_spec(openai_auth_port_spec)

    def execute(self, ctx: knext.ExecutionContext, openai_auth_port: OpenAIAuthenticationPortObject) -> OpenAIChatModelPortObject:

        return OpenAIChatModelPortObject(self.create_spec(openai_auth_port.spec))

    def create_spec(self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec) -> OpenAIChatModelPortObjectSpec:
        credential_params = openai_auth_spec.credentials
        
        if self.input_settings.custom_model_name != "unselected":
            model_name = self.input_settings.custom_model_name
        else:
            model_name = self.input_settings.OpenAIModelCompletionsOptions[
                self.input_settings.model_name
            ].label

        return OpenAIChatModelPortObjectSpec(credential_params, model_name)


@knext.node(
    "OpenAI Embeddings Configurator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Authentication", 
    "All available OpenAI models by ID.",
    openai_authentication_port_type
)
@knext.output_port(
    "OpenAI Embeddings Configuration",
    "An embeddings model configuration for OpenAI.",
    openai_embeddings_port_type,
)
class OpenAIEmbeddingsConfigurator:
    input_settings = EmbeddingsLoaderInputSettings()

    def configure(self, ctx: knext.ConfigurationContext, openai_auth_port_spec: OpenAIAuthenticationPortObjectSpec) -> OpenAIEmbeddingsPortObjectSpec:
        return self.create_spec(openai_auth_port_spec)

    def execute(self, ctx: knext.ExecutionContext, openai_auth_port: OpenAIAuthenticationPortObject) -> OpenAIEmbeddingsPortObject:

        return OpenAIEmbeddingsPortObject(
            self.create_spec(openai_auth_port.spec)
        )

    def create_spec(self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec) -> OpenAIEmbeddingsPortObjectSpec:
        credential_params = openai_auth_spec.credentials
        
        if self.input_settings.custom_model_name != "unselected":
            model_name = self.input_settings.custom_model_name
        else:
            model_name = self.input_settings.OpenAIEmbeddingsOptions[
                self.input_settings.model_name
            ].label

        return OpenAIEmbeddingsPortObjectSpec(credential_params, model_name)


