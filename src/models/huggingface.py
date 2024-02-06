# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods

# KNIME / own imports
import knime.extension as knext
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    model_category,
    GeneralSettings,
    CredentialsSettings,
    LLMChatModelAdapter,
)
from base import AIPortObjectSpec

# Langchain imports
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings

# Other imports
import huggingface_hub

huggingface_icon = "icons/huggingface.png"
huggingface_category = knext.category(
    path=model_category,
    level_id="hugging",
    name="Hugging Face",
    description="",
    icon=huggingface_icon,
)

huggingface_hub_category = knext.category(
    path=huggingface_category,
    level_id="hub",
    name="Hub",
    description="Contains nodes that connect to Hugging Face Hub.",
    icon=huggingface_icon,
)

# == SETTINGS ==


# @knext.parameter_group(label="Model Settings") -- Imported
class HuggingFaceModelSettings(GeneralSettings):
    top_k = knext.IntParameter(
        label="Top k",
        description="The number of top-k tokens to consider when generating text.",
        default_value=1,
        min_value=0,
        is_advanced=True,
    )

    typical_p = knext.DoubleParameter(
        label="Typical p",
        description="The typical probability threshold for generating text.",
        default_value=0.95,
        max_value=1.0,
        min_value=0.1,
        is_advanced=True,
    )

    repetition_penalty = knext.DoubleParameter(
        label="Repetition penalty",
        description="The repetition penalty to use when generating text.",
        default_value=1.0,
        min_value=0.0,
        max_value=100.0,
        is_advanced=True,
    )

    max_new_tokens = knext.IntParameter(
        label="Max new tokens",
        description="""
        The maximum number of tokens to generate in the completion.

        The token count of your prompt plus *max new tokens* cannot exceed the model's context length.
        """,
        default_value=50,
        min_value=0,
    )


@knext.parameter_group(label="Hugging Face TextGen Inference Server Settings")
class HuggingFaceTextGenInferenceInputSettings:
    server_url = knext.StringParameter(
        label="Inference Server URL",
        description="The URL of the inference server to use, e.g. `http://localhost:8010/`.",
        default_value="",
    )


class HFHubTask(knext.EnumParameterOptions):
    TEXT_GENERATION = (
        "text-generation",
        """A popular variant of Text Generation where the model predicts the next word given a sequence of words.
        GPT-based models, such as GPT-3, are commonly used for this task. [Available text generation models](https://huggingface.co/models?pipeline_tag=text-generation)
        """,
    )
    TEXT2TEXT_GENERATION = (
        "text2text-generation",
        "Task used for mapping between pairs of texts, such as translation from one language to another. [Available text-to-text generation models](https://huggingface.co/models?pipeline_tag=text2text-generation)",
    )
    SUMMARIZATION = (
        "summarization",
        "Task used for generating text summaries. [Available summarization models](https://huggingface.co/models?pipeline_tag=summarization)",
    )


def _create_repo_id_parameter():
    return knext.StringParameter(
        label="Repo ID",
        description="""The model name to be used, in the format `<organization_name>/<model_name>`. For example, `Writer/camel-5b-hf`.
                    You can find available models at the [Hugging Face Models repository](https://huggingface.co/models).""",
        default_value="",
    )


@knext.parameter_group(label="Hugging Face Hub Settings")
class HuggingFaceHubSettings:
    repo_id = _create_repo_id_parameter()

    task = knext.EnumParameter(
        "Model Task",
        """
        Task type for a given model.

        Please ensure that the model capabilities align with the chosen task.
        """,
        HFHubTask.TEXT_GENERATION.name,
        HFHubTask,
    )


@knext.parameter_group(label="Prompt Templates")
class HFPromptTemplateSettings:
    system_prompt_template = knext.MultilineStringParameter(
        "System Prompt Template",
        """ Model specific system prompt template. Defaults to "%1".
        Refer to the Hugging Face Hub model card for information on the correct prompt template.""",
        default_value="%1",
    )

    prompt_template = knext.MultilineStringParameter(
        "Prompt Template",
        """ Model specific prompt template. Defaults to "%1". 
        Refer to the Hugging Face Hub model card for information on the correct prompt template.""",
        default_value="%1",
    )


# == Port Objects ==


class HuggingFaceTextGenInfLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self,
        inference_server_url,
        max_new_tokens,
        top_k,
        top_p,
        typical_p,
        temperature,
        repetition_penalty,
    ) -> None:
        super().__init__()
        self._inference_server_url = inference_server_url
        self._max_new_tokens = max_new_tokens
        self._top_k = top_k
        self._top_p = top_p
        self._typical_p = typical_p
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty

    @property
    def inference_server_url(self):
        return self._inference_server_url

    @property
    def max_new_tokens(self):
        return self._max_new_tokens

    @property
    def top_k(self):
        return self._top_k

    @property
    def top_p(self):
        return self._top_p

    @property
    def typical_p(self):
        return self._typical_p

    @property
    def temperature(self):
        return self._temperature

    @property
    def repetition_penalty(self):
        return self._repetition_penalty

    def serialize(self) -> dict:
        return {
            "inference_server_url": self.inference_server_url,
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["inference_server_url"],
            data["max_new_tokens"],
            data["top_k"],
            data["top_p"],
            data["typical_p"],
            data["temperature"],
            data["repetition_penalty"],
        )


class HuggingFaceTextGenInfLLMPortObject(LLMPortObject):
    def __init__(self, spec: HuggingFaceTextGenInfLLMPortObjectSpec) -> None:
        super().__init__(spec)

    def create_model(self, ctx):
        return HuggingFaceTextGenInference(
            inference_server_url=self.spec.inference_server_url,
            max_new_tokens=self.spec.max_new_tokens,
            top_k=self.spec.top_k,
            top_p=self.spec.top_p,
            typical_p=self.spec.typical_p,
            temperature=self.spec.temperature,
            repetition_penalty=self.spec.repetition_penalty,
        )


huggingface_textGenInference_llm_port_type = knext.port_type(
    "Hugging Face TGI LLM",
    HuggingFaceTextGenInfLLMPortObject,
    HuggingFaceTextGenInfLLMPortObjectSpec,
)


class HFTGIChatModelPortObjectSpec(
    HuggingFaceTextGenInfLLMPortObjectSpec, ChatModelPortObjectSpec
):
    def __init__(
        self,
        llm_spec: HuggingFaceTextGenInfLLMPortObjectSpec,
        system_prompt_template: str,
        prompt_template: str,
    ) -> None:
        super().__init__(
            llm_spec.inference_server_url,
            llm_spec.max_new_tokens,
            llm_spec.top_k,
            llm_spec.top_p,
            llm_spec.typical_p,
            llm_spec.temperature,
            llm_spec.repetition_penalty,
        )
        self._system_prompt_template = system_prompt_template
        self._prompt_template = prompt_template

    @property
    def system_prompt_template(self) -> str:
        return self._system_prompt_template

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    def serialize(self) -> dict:
        data = super().serialize()
        data["system_prompt_template"] = self._system_prompt_template
        data["prompt_template"] = self._prompt_template
        return data

    @classmethod
    def deserialize(cls, data: dict):
        llm_spec = HuggingFaceTextGenInfLLMPortObjectSpec.deserialize(data)
        return cls(
            llm_spec,
            system_prompt_template=data["system_prompt_template"],
            prompt_template=data["prompt_template"],
        )


class HFTGIChatModelPortObject(HuggingFaceTextGenInfLLMPortObject, ChatModelPortObject):
    def __init__(self, spec: HFTGIChatModelPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HFTGIChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> LLMChatModelAdapter:
        llm = super().create_model(ctx)
        return LLMChatModelAdapter(
            llm=llm,
            system_prompt_template=self.spec.system_prompt_template,
            prompt_template=self.spec.prompt_template,
        )


huggingface_textGenInference_chat_port_type = knext.port_type(
    "Hugging Face TGI Chat Model",
    HFTGIChatModelPortObject,
    HFTGIChatModelPortObjectSpec,
)


class HFTEIEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(self, inference_server_url: str) -> None:
        super().__init__()
        self._inference_server_url = inference_server_url

    @property
    def inference_server_url(self):
        return self._inference_server_url

    def serialize(self) -> dict:
        return {"inference_server_url": self.inference_server_url}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["inference_server_url"])


class HFTEIEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(self, spec: HFTEIEmbeddingsPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HFTEIEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> HuggingFaceHubEmbeddings:
        return HuggingFaceHubEmbeddings(model=self.spec.inference_server_url)


huggingface_tei_embeddings_port_type = knext.port_type(
    "Hugging Face TEI Embeddings Model",
    HFTEIEmbeddingsPortObject,
    HFTEIEmbeddingsPortObjectSpec,
)


class HuggingFaceAuthenticationPortObjectSpec(AIPortObjectSpec):
    def __init__(self, credentials: str) -> None:
        super().__init__()
        self._credentials = credentials

    @property
    def credentials(self) -> str:
        return self._credentials

    def validate_context(self, ctx: knext.ConfigurationContext):
        if not self.credentials in ctx.get_credential_names():
            raise knext.InvalidParametersError(
                f"The selected credentials '{self.credentials}' holding the Hugging Face Hub API token are not present."
            )
        hub_token = ctx.get_credentials(self.credentials)
        if not hub_token.password:
            raise knext.InvalidParametersError(
                f"The Hugging Face Hub token in the credentials '{self.credentials}' is not present."
            )

    def serialize(self) -> dict:
        return {"credentials": self._credentials}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"])


class HuggingFaceAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: HuggingFaceAuthenticationPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: HuggingFaceAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


huggingface_authentication_port_type = knext.port_type(
    "Hugging Face Hub Authentication",
    HuggingFaceAuthenticationPortObject,
    HuggingFaceAuthenticationPortObjectSpec,
)


class HuggingFaceHubLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self,
        credentials: HuggingFaceAuthenticationPortObjectSpec,
        repo_id,
        task,
        model_kwargs,
    ) -> None:
        super().__init__()
        self._credentials = credentials
        self._repo_id = repo_id
        self._task = task
        self._model_kwargs = model_kwargs

    def validate_context(self, ctx: knext.ConfigurationContext):
        return self._credentials.validate_context(ctx)

    @property
    def credentials(self) -> str:
        return self._credentials.credentials

    @property
    def repo_id(self):
        return self._repo_id

    @property
    def task(self):
        return self._task

    @property
    def model_kwargs(self):
        return self._model_kwargs

    def serialize(self) -> dict:
        return {
            **self._credentials.serialize(),
            "repo_id": self._repo_id,
            "task": self._task,
            "model_kwargs": self._model_kwargs,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            HuggingFaceAuthenticationPortObjectSpec.deserialize(data),
            data["repo_id"],
            data["task"],
            data["model_kwargs"],
        )


class HuggingFaceHubLLMPortObject(LLMPortObject):
    def __init__(self, spec: HuggingFaceHubLLMPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HuggingFaceHubLLMPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        return HuggingFaceHub(
            huggingfacehub_api_token=ctx.get_credentials(
                self.spec.credentials
            ).password,
            repo_id=self.spec.repo_id,
            task=self.spec.task,
            model_kwargs=self.spec.model_kwargs,
        )


huggingface_hub_llm_port_type = knext.port_type(
    "Hugging Face LLM", HuggingFaceHubLLMPortObject, HuggingFaceHubLLMPortObjectSpec
)


class HFHubChatModelPortObjectSpec(
    HuggingFaceHubLLMPortObjectSpec, ChatModelPortObjectSpec
):
    def __init__(
        self,
        llm_spec: HuggingFaceHubLLMPortObjectSpec,
        system_prompt_template: str,
        prompt_template: str,
    ) -> None:
        super().__init__(
            llm_spec._credentials,
            llm_spec.repo_id,
            llm_spec.task,
            llm_spec.model_kwargs,
        )
        self._system_prompt_template = system_prompt_template
        self._prompt_template = prompt_template

    @property
    def system_prompt_template(self) -> str:
        return self._system_prompt_template

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    def serialize(self) -> dict:
        data = super().serialize()
        data["system_prompt_template"] = self.system_prompt_template
        data["prompt_template"] = self.prompt_template
        return data

    @classmethod
    def deserialize(cls, data: dict):
        llm_spec = HuggingFaceHubLLMPortObjectSpec.deserialize(data)
        return cls(
            llm_spec,
            system_prompt_template=data["system_prompt_template"],
            prompt_template=data["prompt_template"],
        )


class HFHubChatModelPortObject(HuggingFaceHubLLMPortObject, ChatModelPortObject):
    @property
    def spec(self) -> HFHubChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> LLMChatModelAdapter:
        llm = super().create_model(ctx)
        return LLMChatModelAdapter(
            llm=llm,
            system_prompt_template=self.spec.system_prompt_template,
            prompt_template=self.spec.prompt_template,
        )


hf_hub_chat_model_port_type = knext.port_type(
    "HF Hub Chat Model", HFHubChatModelPortObject, HFHubChatModelPortObjectSpec
)


class HFHubEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self, hub_credentials: HuggingFaceAuthenticationPortObjectSpec, repo_id: str
    ) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._hub_credentials = hub_credentials

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def hub_credentials_name(self) -> str:
        return self._hub_credentials.credentials

    def validate_context(self, ctx: knext.ConfigurationContext):
        self._hub_credentials.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            **self._hub_credentials.serialize(),
            "repo_id": self.repo_id,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            HuggingFaceAuthenticationPortObjectSpec.deserialize(data), data["repo_id"]
        )


class HFHubEmbeddingsPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> HFHubEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> HuggingFaceHubEmbeddings:
        hub_api_token = ctx.get_credentials(self.spec.hub_credentials_name).password
        return HuggingFaceHubEmbeddings(
            repo_id=self.spec.repo_id, huggingfacehub_api_token=hub_api_token
        )


huggingface_embeddings_port_type = knext.port_type(
    "Hugging Face Hub Embeddings",
    HFHubEmbeddingsPortObject,
    HFHubEmbeddingsPortObjectSpec,
)

# == Nodes ==


@knext.node(
    "HF TGI LLM Connector",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_category,
)
@knext.output_port(
    "Huggingface TGI Configuration",
    "Connection to an LLM hosted on a TGI server.",
    huggingface_textGenInference_llm_port_type,
)
class HuggingfaceTextGenInferenceConnector:
    """
    Connects to a dedicated Text Generation Inference Server.

    The [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
    is a Rust, Python, and gRPC server specifically designed for text generation inference.
    It can be self-hosted to power LLM APIs and inference widgets.

    Please note that this node does not connect to the Hugging Face Hub,
    but to a Text Generation Inference Server that can be hosted both locally and remotely.

    For more details and information about integrating with the Hugging Face TextGen Inference
    and setting up a local server, refer to the
    [LangChain documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_textgen_inference).
    """

    settings = HuggingFaceTextGenInferenceInputSettings()
    model_settings = HuggingFaceModelSettings()

    def configure(self, ctx: knext.ConfigurationContext):
        if not self.settings.server_url:
            raise knext.InvalidParametersError("Server URL missing")

        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext):
        return HuggingFaceTextGenInfLLMPortObject(self.create_spec())

    def create_spec(self):
        return HuggingFaceTextGenInfLLMPortObjectSpec(
            self.settings.server_url,
            self.model_settings.max_new_tokens,
            self.model_settings.top_k,
            self.model_settings.top_p,
            self.model_settings.typical_p,
            self.model_settings.temperature,
            self.model_settings.repetition_penalty,
        )


@knext.node(
    "HF TGI Chat Model Connector",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_category,
)
@knext.output_port(
    "Huggingface TGI LLM Connection",
    "Connection to a chat model hosted on a Text Generation Inference server.",
    huggingface_textGenInference_chat_port_type,
)
class HFTGIChatModelConnector:
    """
    Connects to a dedicated Text Generation Inference Server.

    The [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
    is a Rust, Python, and gRPC server specifically designed for text generation inference.
    It can be self-hosted to power LLM APIs and inference widgets.

    Please note that this node does not connect to the Hugging Face Hub,
    but to a Text Generation Inference Server that can be hosted both locally and remotely.

    For more details and information about integrating with the Hugging Face TextGen Inference
    and setting up a local server, refer to the
    [LangChain documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_textgen_inference).
    """

    settings = HuggingFaceTextGenInferenceInputSettings()
    templates = HFPromptTemplateSettings()
    model_settings = HuggingFaceModelSettings()

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> HFTGIChatModelPortObjectSpec:
        if not self.settings.server_url:
            raise knext.InvalidParametersError("Server URL missing")

        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> HFTGIChatModelPortObject:
        return HFTGIChatModelPortObject(self.create_spec())

    def create_spec(self) -> HFTGIChatModelPortObjectSpec:
        llm_spec = HuggingFaceTextGenInfLLMPortObjectSpec(
            self.settings.server_url,
            self.model_settings.max_new_tokens,
            self.model_settings.top_k,
            self.model_settings.top_p,
            self.model_settings.typical_p,
            self.model_settings.temperature,
            self.model_settings.repetition_penalty,
        )

        return HFTGIChatModelPortObjectSpec(
            llm_spec,
            self.templates.system_prompt_template,
            self.templates.prompt_template,
        )


@knext.node(
    "HF TEI Embeddings Connector",
    knext.NodeType.SOURCE,
    huggingface_icon,
    huggingface_category,
)
@knext.output_port(
    "Embeddings Model",
    "Connection to an embeddings model hosted on a Text Embeddings Inference server.",
    huggingface_tei_embeddings_port_type,
)
class HFTEIEmbeddingsConnector:
    """
    Connects to a dedicated Text Embeddings Inference Server.

    The [Text Embeddings Inference Server](https://github.com/huggingface/text-embeddings-inference)
    is a toolkit for deploying and serving open source text embeddings and sequence classification models.

    Please note that this node does only connect to public embedding inference endpoints that do not require an API key.

    For more details and information about integrating with the Hugging Face Embeddings Inference
    and setting up a server, refer to
    [Text Embeddings Inference GitHub](https://github.com/huggingface/text-embeddings-inference).
    """

    server_url = knext.StringParameter(
        "Text Embeddings Inference Server URL",
        "The URL where the Text Embeddings Inference server is hosted e.g. `http://localhost:8010/`.",
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> HFTEIEmbeddingsPortObjectSpec:
        if not self.server_url:
            raise knext.InvalidParametersError("Server URL missing")

        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext):
        return HFTEIEmbeddingsPortObject(self.create_spec())

    def create_spec(self) -> HFTEIEmbeddingsPortObjectSpec:
        return HFTEIEmbeddingsPortObjectSpec(self.server_url)


@knext.node(
    "HF Hub Authenticator",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_hub_category,
)
@knext.output_port(
    "Hugging Face Hub Authentication",
    "Validated authentication for Hugging Face Hub.",
    huggingface_authentication_port_type,
)
class HuggingFaceHubAuthenticator:
    """
    Authenticates the Hugging Face Hub API key.

    This node provides the authentication for all Hugging Face Hub models.

    It allows you to select the credentials that contain a valid OpenAI API key in their *password* field (the *username* is ignored).
    Credentials can be set on the workflow level or created inside the workflow e.g. with the [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and fed into this node via flow variable.

    If you don't have a Hugging Face API key yet, you can generate one by visiting [Hugging Face](https://huggingface.co/settings/tokens).
    Follow the instructions provided to generate your API key.
    """

    credentials_settings = CredentialsSettings(
        label="Hugging Face API Key",
        description="""
            The credentials containing the Hugging Face Hub API key in its *password* field (the *username* is ignored).
            """,
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> HuggingFaceAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            # credentials_settings.credentials_param = "Placeholder value"
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        spec = self.create_spec()
        spec.validate_context(ctx)
        return spec

    def execute(self, ctx: knext.ExecutionContext):
        try:
            huggingface_hub.whoami(
                ctx.get_credentials(
                    self.credentials_settings.credentials_param
                ).password
            )
        except:
            raise knext.InvalidParametersError("Invalid API Key.")

        return HuggingFaceAuthenticationPortObject(self.create_spec())

    def create_spec(self):
        return HuggingFaceAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param,
        )


@knext.node(
    "HF Hub LLM Connector",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_hub_category,
)
@knext.input_port(
    "Hugging Face Authentication",
    "Validated authentication for Hugging Face Hub.",
    huggingface_authentication_port_type,
)
@knext.output_port(
    "Hugging Face LLM",
    "Connection to a specific LLM from Hugging Face Hub.",
    huggingface_hub_llm_port_type,
)
class HuggingFaceHubConnector:
    """
    Connects to an LLM hosted on the Hugging Face Hub.

    This node establishes a connection to a specific LLM hosted on the Hugging Face Hub.
    To use this node, you need to successfully authenticate with the Hugging Face Hub using the **HF Hub Authenticator node**.

    Provide the name of the desired LLM repository available on the [Hugging Face Hub](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_hub) as an input.

    For more details and information about integrating LLMs from the Hugging Face Hub, refer to the [LangChain documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_hub).
    """

    hub_settings = HuggingFaceHubSettings()
    model_settings = HuggingFaceModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        huggingface_auth_spec: HuggingFaceAuthenticationPortObjectSpec,
    ) -> HuggingFaceHubLLMPortObjectSpec:
        if not self.hub_settings.repo_id:
            raise knext.InvalidParametersError("Please enter a repo ID.")
        huggingface_auth_spec.validate_context(ctx)
        _validate_repo_id(self.hub_settings.repo_id)
        return self.create_spec(huggingface_auth_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        huggingface_auth: HuggingFaceAuthenticationPortObject,
    ) -> HuggingFaceHubLLMPortObject:
        return HuggingFaceHubLLMPortObject(self.create_spec(huggingface_auth.spec))

    def create_spec(
        self, huggingface_auth_spec: HuggingFaceAuthenticationPortObjectSpec
    ):
        model_kwargs = {
            "temperature": self.model_settings.temperature,
            "top_p": self.model_settings.top_p,
            "top_k": self.model_settings.top_k,
            "max_new_tokens": self.model_settings.max_new_tokens,
        }

        return HuggingFaceHubLLMPortObjectSpec(
            huggingface_auth_spec,
            self.hub_settings.repo_id,
            HFHubTask[self.hub_settings.task].label,
            model_kwargs,
        )


@knext.node(
    "HF Hub Chat Model Connector",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_hub_category,
)
@knext.input_port(
    "Hugging Face Authentication",
    "Validated authentication for Hugging Face Hub.",
    huggingface_authentication_port_type,
)
@knext.output_port(
    "Hugging Face Hub Chat Model",
    "Connection to a specific chat model from Hugging Face Hub.",
    hf_hub_chat_model_port_type,
)
class HFHubChatModelConnector:
    """
    Connects to a chat model hosted on the Hugging Face Hub.

    This node establishes a connection to a specific chat model hosted on the Hugging Face Hub.
    The difference to the HF Hub LLM Connector is that this node allows you to provide prompt templates which are crucial for
    obtaining the best output from many models that have been fine-tuned for chatting.
    To use this node, you need to successfully authenticate with the Hugging Face Hub using the **HF Hub Authenticator node**.

    Provide the name of the desired chat model repository available on the
    [Hugging Face Hub](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_hub)
    as an input.
    """

    hub_settings = HuggingFaceHubSettings()
    template_settings = HFPromptTemplateSettings()
    model_settings = HuggingFaceModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: HuggingFaceAuthenticationPortObjectSpec,
    ) -> HFHubChatModelPortObjectSpec:
        auth.validate_context(ctx)
        if not self.hub_settings.repo_id:
            raise knext.InvalidParametersError("Please enter a repo ID.")
        _validate_repo_id(self.hub_settings.repo_id)
        return self._create_spec(auth)

    def _create_spec(self, auth: HuggingFaceAuthenticationPortObjectSpec):
        model_kwargs = {
            "temperature": self.model_settings.temperature,
            "top_p": self.model_settings.top_p,
            "top_k": self.model_settings.top_k,
            "max_new_tokens": self.model_settings.max_new_tokens,
        }

        llm_spec = HuggingFaceHubLLMPortObjectSpec(
            auth,
            self.hub_settings.repo_id,
            HFHubTask[self.hub_settings.task].label,
            model_kwargs,
        )
        return HFHubChatModelPortObjectSpec(
            llm_spec,
            self.template_settings.system_prompt_template,
            self.template_settings.prompt_template,
        )

    def execute(
        self, ctx, auth: HuggingFaceAuthenticationPortObject
    ) -> HFHubChatModelPortObject:
        return HFHubChatModelPortObject(self._create_spec(auth.spec))


def _validate_repo_id(repo_id):
    try:
        huggingface_hub.model_info(repo_id)
    except:
        raise knext.InvalidParametersError("Please provide a valid repo ID.")


@knext.node(
    "HF Hub Embeddings Connector",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface_hub_category,
)
@knext.input_port(
    "Hugging Face Hub Authentication",
    "The authentication for the Hugging Face Hub.",
    huggingface_authentication_port_type,
)
@knext.output_port(
    "Hugging Face Hub Embeddings",
    "An embeddings model connected to Hugging Face Hub.",
    huggingface_embeddings_port_type,
)
class HFHubEmbeddingsConnector:
    """
    Connects to an Embeddings model hosted on the Hugging Face Hub.

    This node establishes a connection to a specific Embeddings model hosted on the Hugging Face Hub.
    To use this node, you need to successfully authenticate with the Hugging Face Hub using the **HF Hub Authenticator node**.

    Provide the name of the desired Embeddings repository available on the [Hugging Face Hub](https://huggingface.co/) as an input.

    """

    repo_id = _create_repo_id_parameter()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication_spec: HuggingFaceAuthenticationPortObjectSpec,
    ):
        if not self.repo_id:
            raise knext.InvalidParametersError("Please enter a repo ID.")
        _validate_repo_id(self.repo_id)
        authentication_spec.validate_context(ctx)
        return self.create_spec(authentication_spec)

    def create_spec(
        self, authentication_spec: HuggingFaceAuthenticationPortObjectSpec
    ) -> HFHubEmbeddingsPortObjectSpec:
        return HFHubEmbeddingsPortObjectSpec(authentication_spec, self.repo_id)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        authentication: HuggingFaceAuthenticationPortObject,
    ):
        output = HFHubEmbeddingsPortObject(self.create_spec(authentication.spec))
        # TODO validate that repo does supports what we want to do
        output.create_model(ctx)
        return output
