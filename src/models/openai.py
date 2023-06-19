import knime.extension as knext
from .base import (
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    ModelPortObjectSpecContent,
    ModelPortObjectContent,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
)
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

openai_icon = ""
openai_category = knext.category(
    path="/community",
    level_id="llm",
    name="Model Loaders",
    description="All nodes related to models",
    icon="",
)


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing OpenAI API key",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


class OpenAILLMPortObjectSpecContent(ModelPortObjectSpecContent):
    def __init__(self, credentials, model_name) -> None:
        super().__init__()
        self._credentials = credentials
        self._model_name = model_name

    def serialize(self) -> dict:
        return {"credentials": self._credentials, "model": self._model_name}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"], data["model"])


LLMPortObjectSpec.register_content_type(OpenAILLMPortObjectSpecContent)


class OpenAILLMPortObjectContent(ModelPortObjectContent):
    def create_model(self, ctx):
        return OpenAI(
            openai_api_key=ctx.get_credentials(
                self.spec.serialize()["credentials"]
            ).password,
            model=self.spec.serialize()["model"],
        )


LLMPortObject.register_content_type(OpenAILLMPortObjectContent)


class OpenAIChatModelPortObjectSpecContent(ModelPortObjectSpecContent):
    def __init__(self, credentials, model_name) -> None:
        super().__init__()
        self._credentials = credentials
        self._model_name = model_name

    def serialize(self) -> dict:
        return {"credentials": self._credentials, "model": self._model_name}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"], data["model"])


ChatModelPortObjectSpec.register_content_type(OpenAIChatModelPortObjectSpecContent)


class OpenAIChatModelPortObjectContent(ModelPortObjectContent):
    def create_model(self, ctx):
        return OpenAI(
            openai_api_key=ctx.get_credentials(
                self.spec.serialize()["credentials"]
            ).password,
            model=self.spec.serialize()["model"],
        )


ChatModelPortObject.register_content_type(OpenAILLMPortObjectContent)


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
        "Model name",
        "GPT-3 (Ada, Babbage, Curie) and GPT-3.5 (Davinci) models",
        OpenAIModelCompletionsOptions.Ada.name,
        OpenAIModelCompletionsOptions,
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


@knext.node(
    "OpenAI LLM Configurator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.output_port(
    "OpenAI LLM Configuration",
    "A large language model configuration for OpenAI.",
    llm_port_type,
)
class OpenAILLMConfigurator:
    input_settings = LLMLoaderInputSettings()
    credentials_settings = CredentialsSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> LLMPortObjectSpec:
        return LLMPortObjectSpec(self.create_spec_content())

    def execute(self, ctx: knext.ExecutionContext) -> LLMPortObject:
        spec_content = self.create_spec_content()

        return LLMPortObject(
            spec=LLMPortObjectSpec(spec_content),
            content=OpenAILLMPortObjectContent(spec_content),
        )

    def create_spec_content(self):
        credential_params = self.credentials_settings.credentials_param
        model_name = self.input_settings.OpenAIModelCompletionsOptions[
            self.input_settings.model_name
        ].label

        return OpenAILLMPortObjectSpecContent(credential_params, model_name)


@knext.node(
    "OpenAI ChatModel Configurator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.output_port(
    "OpenAI LLM Configuration",
    "A chat model configuration for OpenAI.",
    llm_port_type,
)
class OpenAIChatModelConfigurator:
    input_settings = ChatModelLoaderInputSettings()
    credentials_settings = CredentialsSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> LLMPortObjectSpec:
        return LLMPortObjectSpec(self.create_spec_content())

    def execute(self, ctx: knext.ExecutionContext) -> LLMPortObject:
        spec_content = self.create_spec_content()

        return LLMPortObject(
            spec=LLMPortObjectSpec(spec_content),
            content=OpenAILLMPortObjectContent(spec_content),
        )

    def create_spec_content(self):
        credential_params = self.credentials_settings.credentials_param
        model_name = self.input_settings.OpenAIModelCompletionsOptions[
            self.input_settings.model_name
        ].label

        return OpenAILLMPortObjectSpecContent(credential_params, model_name)


class OpenAIEmbeddingsPortObjectSpecContent(ModelPortObjectSpecContent):
    def __init__(self, credentials, model_name) -> None:
        super().__init__()
        self._credentials = credentials
        self._model_name = model_name

    def serialize(self) -> dict:
        return {"credentials": self._credentials, "model": self._model_name}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"], data["model"])


EmbeddingsPortObjectSpec.register_content_type(OpenAIEmbeddingsPortObjectSpecContent)


class OpenAIEmbeddingsPortObjectContent(ModelPortObjectContent):
    def create_model(self, ctx):
        return OpenAIEmbeddings(
            openai_api_key=ctx.get_credentials(
                self.spec.serialize()["credentials"]
            ).password,
            model=self.spec.serialize()["model"],
        )


EmbeddingsPortObject.register_content_type(OpenAIEmbeddingsPortObjectContent)


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


@knext.node(
    "OpenAI Embeddings Configurator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.output_port(
    "OpenAI Embeddings Configuration",
    "An embeddings model configuration for OpenAI.",
    embeddings_port_type,
)
class OpenAIEmbeddingsConfigurator:
    input_settings = EmbeddingsLoaderInputSettings()
    credentials_settings = CredentialsSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> EmbeddingsPortObjectSpec:
        return EmbeddingsPortObjectSpec(self.create_spec_content())

    def execute(self, ctx: knext.ExecutionContext) -> EmbeddingsPortObject:
        spec_content = self.create_spec_content()

        return EmbeddingsPortObject(
            spec=EmbeddingsPortObjectSpec(spec_content),
            content=OpenAIEmbeddingsPortObjectContent(spec_content),
        )

    def create_spec_content(self):
        credential_params = self.credentials_settings.credentials_param
        model_name = self.input_settings.OpenAIEmbeddingsOptions[
            self.input_settings.model_name
        ].label

        return OpenAIEmbeddingsPortObjectSpecContent(credential_params, model_name)
