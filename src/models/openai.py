import knime.extension as knext
from .base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
    LLMPortObjectSpec,
    LLMPortObjectSpecContent,
    LLMPortObject,
    LLMPortObjectContent,
    llm_port_type,
)
from langchain.llms import OpenAI
import openai

import logging

LOGGER = logging.getLogger(__name__)

openai_icon = "./icons/openai.png"
openai_category = knext.category(
    path="/community",
    level_id="top",
    name="Model Loaders",
    description="All nodes related to models",
    icon="icons/ml.svg",
)

class OpenAILLMPortObjectSpecContent(LLMPortObjectSpecContent):

    def __init__(self, credentials, model_name) -> None:
        super().__init__()
        self._credentials = credentials
        self._model_name = model_name

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
            "model": self._model_name
        }
    
    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"],data["model"])

    
LLMPortObjectSpec.register_content_type(OpenAILLMPortObjectSpecContent)

class OpenAILLMPortObjectContent(LLMPortObjectContent):

    def create_llm(self, ctx):
        return OpenAI(
            openai_api_key=ctx.get_credentials(self.spec.serialize()["credentials"]).password, 
            model=self.spec.serialize()["model"]
        )
    
LLMPortObject.register_content_type(OpenAILLMPortObjectContent)

@knext.parameter_group(label="OpenAI LLM Settings")
class LLMLoaderInputSettings:
    class ModelOptions(knext.EnumParameterOptions):
        Ada = (
            "text-ada-001",
            "Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.",
        )
        DaVinci = (
            "text-davinci-003",
            "Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models.",
        )
        Babbage = (
            "text-babbage-001",
            "Capable of straightforward tasks, very fast, and lower cost.",
        )
        Curie = (
            "text-curie-001",
            "Very capable, but faster and lower cost than Davinci.",
        )
        GPT_3_5_Turbo = (
            "gpt-3.5-turbo",
            "Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003.",
        )

    model_name = knext.EnumParameter(
        "Model name",
        "GPT-3 (ada, babbage, curie) and GPT-3.5 (davinci) models",
        ModelOptions.Ada.name,
        ModelOptions,
    )


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing OpenAI API key",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


@knext.node(
    "OpenAI Embeddings Loader",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
)
@knext.output_port(
    "OpenAI Embeddings", "An embeddings model from OpenAI.", embeddings_port_type
)
class OpenAIEmbeddingsLoader:
    credentials_settings = CredentialsSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> EmbeddingsPortObjectSpec:
        return EmbeddingsPortObjectSpec(None, None)

    def execute(self, ctx: knext.ExecutionContext) -> EmbeddingsPortObject:
        credential_params = self.credentials_settings.credentials_param
        model_name = "text-embedding-ada-002"

        return EmbeddingsPortObject(
            EmbeddingsPortObjectSpec(credential_params, model_name)
        )


@knext.node(
    "OpenAI LLM Loader", knext.NodeType.SOURCE, openai_icon, category=openai_category
)
@knext.output_port("OpenAI LLM", "A large language model from OpenAI.", llm_port_type)
class OpenAILLMLoader:
    input_settings = LLMLoaderInputSettings()
    credentials_settings = CredentialsSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> LLMPortObjectSpec:
        return LLMPortObjectSpec(self.create_spec_content())

    def execute(self, ctx: knext.ExecutionContext) -> LLMPortObject:

        spec_content = self.create_spec_content()

        return LLMPortObject(
            spec=LLMPortObjectSpec(spec_content),
            content=OpenAILLMPortObjectContent(spec_content)
        )

    def create_spec_content(self):
        credentials_params = self.credentials_settings.credentials_param
        model_name = self.input_settings.ModelOptions[
            self.input_settings.model_name
        ].label

        return OpenAILLMPortObjectSpecContent(credentials_params, model_name)