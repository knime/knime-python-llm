import knime.extension as knext
from .base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_port_type,
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
)
from langchain.llms import OpenAI
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
        return LLMPortObjectSpec(None, None)

    def execute(self, ctx: knext.ExecutionContext) -> LLMPortObject:
        credentials_params = self.credentials_settings.credentials_param
        model_name = self.input_settings.ModelOptions[
            self.input_settings.model_name
        ].label

        return LLMPortObject(LLMPortObjectSpec(credentials_params, model_name))


@knext.node(
    "OpenAI LLM Prompter", knext.NodeType.SOURCE, openai_icon, category=openai_category
)
@knext.input_port("OpenAI LLM", "A large language model from OpenAI.", llm_port_type)
@knext.output_port("OpenAI LLM", "A large language model from OpenAI.", llm_port_type)
class OpenAILLMPrompter:
    prompt = knext.StringParameter("Prompt", "The prompt that is being asked", "")

    def configure(
        self, ctx: knext.ConfigurationContext, spec: LLMPortObjectSpec
    ) -> LLMPortObjectSpec:
        return spec

    def execute(
        self, ctx: knext.ExecutionContext, llm_port: LLMPortObject
    ) -> LLMPortObject:
        llm = OpenAI(
            model_name=llm_port.spec.model_name,
            openai_api_key=ctx.get_credentials(llm_port.spec.credentials).password,
        )

        LOGGER.info(llm(self.prompt))

        return llm_port
