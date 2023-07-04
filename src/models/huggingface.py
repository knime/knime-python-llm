# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods

# KNIME / own imports
import knime.extension as knext
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    model_category,
    GeneralSettings,
)

# Langchain imports
from langchain import HuggingFaceHub
from langchain.llms import HuggingFaceTextGenInference
from langchain.embeddings import HuggingFaceEmbeddings

# Other imports
import huggingface_hub

# TODO: Get someone to do new icons
huggingface_icon = "icons/huggingface.png"
huggingface = knext.category(
    path=model_category,
    level_id="hugging",
    name="Hugging Face",
    description="",
    icon=huggingface_icon,
)

# == SETTINGS ==


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Hugging Face API Key",
        description="""
        Credentials parameter for accessing the Hugging Face API key
        """,
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


# @knext.parameter_group(label="Model Settings") -- Imported
class HuggingFaceModelSettings(GeneralSettings):
    max_tokens = knext.IntParameter(
        label="Max tokens",
        description="""
        The maximum number of tokens to generate in the completion.

        The token count of your prompt plus 
        max_tokens cannot exceed the model's context length.
        """,
        default_value=50,
        max_value=250,
        min_value=0,
    )

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
        label="Max tokens",
        description="""
        The maximum number of tokens to generate in the completion.

        The token count of your prompt plus 
        max_tokens cannot exceed the model's context length.
        """,
        default_value=50,
        max_value=256,
        min_value=0,
    )


@knext.parameter_group(label="Hugging Face TextGen Inference Server Settings")
class HuggingFaceTextGenInferenceInputSettings:
    server_url = knext.StringParameter(
        label="Inference Server URL",
        description="The URL of the inference server to use.",
        default_value="",
    )


class HFHubTask(knext.EnumParameterOptions):
    TEXT_GENERATION = (
        "text-generation",
        """A popular variant of Text Generation that predicts the next word given a bunch of words.
        The most popular models for this task are GPT-based models (such as GPT-3).
        Note that model capability has to match the task.

        See available [models](https://huggingface.co/models?pipeline_tag=text-generation).
        """,
    )
    TEXT2TEXT_GENERATION = (
        "text2text-generation",
        """Task that is used for mapping between a pair of texts 
        (e.g. translation from one language to another).
        Note that model capability has to match the task.

        See available [models](https://huggingface.co/models?pipeline_tag=text2text-generation).
        """,
    )
    SUMMARIZATION = (
        "summarization",
        """Task that is used to summarize text.
        Note that model capability has to match the task.

        See available [models](https://huggingface.co/models?pipeline_tag=summarization).
        """,
    )


@knext.parameter_group(label="Hugging Face Hub Settings")
class HuggingFaceHubSettings:
    repo_id = knext.StringParameter(
        label="Repo ID",
        description="""Model name to use e.g 'Writer/camel-5b-hf'. The repo ID corresponds to '<organizetion_name>/<model_name>'. 
        [Available model repositories](https://huggingface.co/models)""",
        default_value="",
    )

    task = knext.EnumParameter(
        "Model Task",
        "Task that the model is supposed to follow.",
        HFHubTask.TEXT_GENERATION.name,
        HFHubTask,
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
    "Hugging Face LLM",
    HuggingFaceTextGenInfLLMPortObject,
    HuggingFaceTextGenInfLLMPortObjectSpec,
)


class HuggingFaceHubLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self,
        credentials,
        repo_id,
        task,
        model_kwargs,
    ) -> None:
        super().__init__()
        self._credentials = credentials
        self._repo_id = repo_id
        self._task = task
        self._model_kwargs = model_kwargs

    @property
    def credentials(self):
        return self._credentials

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
            "credentials": self._credentials,
            "repo_id": self._repo_id,
            "task": self._task,
            "model_kwargs": self._model_kwargs,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["credentials"],
            data["repo_id"],
            data["task"],
            data["model_kwargs"],
        )


class HuggingFaceHubLLMPortObject(LLMPortObject):
    def __init__(self, spec: HuggingFaceHubLLMPortObjectSpec) -> None:
        super().__init__(spec)

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


class HuggingFaceAuthenticationPortObjectSpec(knext.PortObjectSpec):
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


class HuggingFaceEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(self) -> None:
        super().__init__()

    def serialize(self) -> dict:
        return {}

    @classmethod
    def deserialize(cls, data: dict):
        return cls()


class HuggingFacePortObject(EmbeddingsPortObject):
    def __init__(self, spec: HuggingFaceEmbeddingsPortObjectSpec):
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec):
        return cls(spec)

    def create_model(self, ctx):
        return HuggingFaceEmbeddings()


huggingface_embeddings_port_type = knext.port_type(
    "Hugging Face Embeddings Port Type",
    HuggingFacePortObject,
    HuggingFaceEmbeddingsPortObjectSpec,
)


# == Nodes ==


@knext.node(
    "HF TextGen Inference Connector",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface,
)
@knext.output_port(
    "Huggingface TextGen Inference Configuration",
    "Connection to LLM hosted on a Text Generation Inference Server",
    huggingface_textGenInference_llm_port_type,
)
class HuggingfaceTextGenInferenceConnector:
    """

    Connect to a dedicated TextGen Inference Server

    [Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a Rust,
    Python and gRPC server for text generation inference. It is used in production at HuggingFace or
    to self-host and power LLMs api-inference widgets.

    Note, this does not connect to the Hubbing Face Hub, but to your own Text Generation Inference Server.


    See [LangChain documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_textgen_inference) of the Hugging Face TextGen Inference for more details.

    """

    settings = HuggingFaceTextGenInferenceInputSettings()
    model_settings = HuggingFaceModelSettings()

    def configure(self, ctx: knext.ConfigurationContext):
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
    "HF Hub Authenticator",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface,
)
@knext.output_port(
    "Hugging Face Hub Authentication",
    "Successful authentication to Hugging Face Hub.",
    huggingface_authentication_port_type,
)
class HuggingFaceHubAuthenticator:
    """

    Authenticates the Hugging Face API Key.

    This node validates the provided Hugging Face API key by making a
    request to the Hugging Face "whoami endpoint. The valid API key
    is used with the Hub Connection node.

    Use the
    [Credentials Configuration Node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    to provide the API key in a credentials object.

    If you dont have a Hugging Face API key yet, generate one at
    [Hugging Face](https://huggingface.co/settings/tokens).

    """

    credentials_settings = CredentialsSettings()

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> HuggingFaceAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        return self.create_spec()

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
    category=huggingface,
)
@knext.input_port(
    "Hugging Face Authentication",
    "Successfull authentication to Hugging Face.",
    huggingface_authentication_port_type,
)
@knext.output_port(
    "Hugging Face LLM",
    "Connection to a specific LLM from Hugging Face.",
    huggingface_hub_llm_port_type,
)
class HuggingFaceHubConnector:
    """

    Connects to a Hugging Face Hub hosted Large Language Model.

    Given the successfull authentication through the Hugging Face Authenticator Node,
    input one of the available LLM repository names from [Hugging Face Hub](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_hub), and
    keyword arguments for the given connection.

    See [LangChain documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_hub) of the LLM integration for more details.

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

        try:
            huggingface_hub.model_info(self.hub_settings.repo_id)
        except:
            raise knext.InvalidParametersError("Please provide a valid repo ID.")

        return self.create_spec(huggingface_auth_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        huggingface_auth_spec: HuggingFaceAuthenticationPortObjectSpec,
    ) -> HuggingFaceHubLLMPortObject:
        return HuggingFaceHubLLMPortObject(self.create_spec(huggingface_auth_spec.spec))

    def create_spec(
        self, huggingface_auth_spec: HuggingFaceAuthenticationPortObjectSpec
    ):
        model_kwargs = {
            "temperature": self.model_settings.temperature,
            "top_p": self.model_settings.top_p,
            "top_k": self.model_settings.top_k,
            "max_new_tokens": self.model_settings.max_tokens,
        }

        return HuggingFaceHubLLMPortObjectSpec(
            huggingface_auth_spec.credentials,
            self.hub_settings.repo_id,
            HFHubTask[self.hub_settings.task].label,
            model_kwargs,
        )


@knext.node(
    "HF Embeddings Configurator",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface,
)
@knext.output_port(
    "Hugging Face Embeddings",
    "An embeddings model configuration from Hugging Face Hub.",
    huggingface_embeddings_port_type,
)
class HuggingFaceEmbeddingsConfigurator:
    def configure(self, ctx: knext.ConfigurationContext):
        return HuggingFaceEmbeddingsPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext):
        return HuggingFacePortObject(HuggingFaceEmbeddingsPortObjectSpec())
