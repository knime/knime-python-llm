# KNIME / own imports
import knime.extension as knext
from knime.extension.nodes import FilestorePortObject
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    model_category,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    GeneralSettings,
    LLMChatModelAdapter,
)

from pydantic import ValidationError, root_validator
from typing import Optional, Dict
import shutil
import os
from gpt4all import Embed4All

# Langchain imports
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.embeddings import Embeddings

gpt4all_icon = "icons/gpt4all.png"
gpt4all_category = knext.category(
    path=model_category,
    level_id="gpt4all",
    name="GPT4All",
    description="Contains nodes for connecting to GPT4All models.",
    icon=gpt4all_icon,
)


@knext.parameter_group(label="Model Usage")
class GPT4AllInputSettings:
    local_path = knext.LocalPathParameter(
        label="Model path",
        description="""Path to the pre-trained GPT4All model file eg. my/path/model.gguf.
        You can find the folder through settings -> application in the gpt4all desktop application.""",
    )

    n_threads = knext.IntParameter(
        label="Thread Count",
        description="""Number of CPU threads used by GPT4All. Default is 0, then the number of threads 
        are determined automatically. 
        """,
        default_value=0,
        min_value=0,
        is_advanced=True,
        since_version="5.2.0",
    )


class GPT4AllModelParameterSettings(GeneralSettings):
    max_token = knext.IntParameter(
        label="Maximum Response Length (token)",
        description="""
        The maximum number of tokens to generate.

        The token count of your prompt plus 
        max_tokens cannot exceed the model's context length.
        """,
        default_value=250,
        min_value=1,
    )

    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 1.0. 
        Higher values means the model will take more risks. 
        Try 0.9 for more creative applications, and 0 for ones with a well-defined answer.
        It is generally recommend altering this or top_p but not both.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
    )

    top_k = knext.IntParameter(
        label="Top-k sampling",
        description="""
        Set the "k" value to limit the vocabulary used during text generation. Smaller values (e.g., 10) restrict the choices 
        to the most probable words, while larger values (e.g., 50) allow for more variety.
        """,
        default_value=20,
        min_value=1,
        is_advanced=True,
    )

    prompt_batch_size = knext.IntParameter(
        label="Prompt batch size",
        description="""Amount of prompt tokens to process at once. 
                    NOTE: On CPU higher values can speed up reading prompts but will also use more RAM.
                    On GPU, a batch size of 1 has outperformed other batch sizes in our experiments.""",
        default_value=128,
        min_value=1,
        is_advanced=True,
    )

    device = knext.StringParameter(
        label="Device",
        description="""The processing unit on which the GPT4All model will run. It can be set to:

        - "cpu": Model will run on the central processing unit.
        - "gpu": Model will run on the best available graphics processing unit, irrespective of its vendor.
        - "amd", "nvidia", "intel": Model will run on the best available GPU from the specified vendor. 
        
        Alternatively, a specific GPU name can also be provided, and the model will run on the GPU that matches the name if it's available. 
        Default is "cpu".

        Note: If a selected GPU device does not have sufficient RAM to accommodate the model, an error will be thrown.
        It's advised to ensure the device has enough memory before initiating the model.""",
        default_value="cpu",
        is_advanced=True,
    )


@knext.parameter_group(label="Prompt Templates")
class GPT4AllPromptSettings:
    system_prompt_template = knext.MultilineStringParameter(
        "System Prompt Template",
        """ Model specific system template. Defaults to "%1". Refer to the 
        [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json) 
        for the correct template for your model:

        1. Locate the model you are using under the field "name". 
        2. Within the Model object, locate the "systemPrompt" field and use these values.""",
        default_value="%1",
    )

    prompt_template = knext.MultilineStringParameter(
        "Prompt Template",
        """ Model specific prompt template. Defaults to "%1". Refer to the 
        [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json) 
        for the correct template for your model:

        1. Locate the model you are using under the field "name". 
        2. Within the Model object, locate the "promptTemplate" field and use these values.

        Note: For instruction based models, it is recommended to use "[INST] %1 [/INST]" as the 
        prompt template for better output if the "promptTemplate" field is not specified in the model list.""",
        default_value="""### Human:
%1
### Assistant:
""",
    )


class GPT4AllLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self,
        local_path: str,
        n_threads: int,
        temperature: float,
        top_k: int,
        top_p: float,
        max_token: int,
        prompt_batch_size: int,
        device: str,
    ) -> None:
        super().__init__()
        self._local_path = local_path
        self._n_threads = n_threads
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._max_token = max_token
        self._prompt_batch_size = prompt_batch_size
        self._device = device

    @property
    def local_path(self) -> str:
        return self._local_path

    @property
    def n_threads(self) -> int:
        return self._n_threads

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def top_p(self) -> float:
        return self._top_p

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def max_token(self) -> int:
        return self._max_token

    @property
    def prompt_batch_size(self) -> int:
        return self._prompt_batch_size

    @property
    def device(self) -> str:
        return self._device

    def serialize(self) -> dict:
        return {
            "local_path": self._local_path,
            "n_threads": self._n_threads,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "max_token": self._max_token,
            "prompt_batch_size": self.prompt_batch_size,
            "device": self._device,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            local_path=data["local_path"],
            n_threads=data.get("n_threads", None),
            temperature=data.get("temperature", 0.2),
            top_k=data.get("top_k", 20),
            top_p=data.get("top_p", 0.15),
            max_token=data.get("max_token", 250),
            prompt_batch_size=data.get("prompt_batch_size", 128),
            device=data.get("device", "cpu"),
        )


class GPT4AllLLMPortObject(LLMPortObject):
    @property
    def spec(self) -> GPT4AllLLMPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> GPT4All:
        try:
            return GPT4All(
                model=self.spec.local_path,
                n_threads=self.spec.n_threads,
                temp=self.spec.temperature,
                top_p=self.spec.top_p,
                top_k=self.spec.top_k,
                n_predict=self.spec.max_token,
                n_batch=self.spec.prompt_batch_size,
                device=self.spec.device,
            )
        except ValidationError as e:
            error_msg = e.errors()[0]["msg"]
            if "Unable to initialize model on GPU:" in error_msg:
                raise knext.InvalidParametersError(error_msg) from e
            raise knext.InvalidParametersError(
                "Could not create model. Please provide a model in GGUF format."
            )


class GPT4AllChatModelPortObjectSpec(GPT4AllLLMPortObjectSpec, ChatModelPortObjectSpec):
    def __init__(
        self,
        llm_spec: GPT4AllLLMPortObjectSpec,
        system_prompt_template: str,
        prompt_template: str,
    ) -> None:
        super().__init__(
            local_path=llm_spec._local_path,
            n_threads=llm_spec._n_threads,
            temperature=llm_spec._temperature,
            top_k=llm_spec._top_k,
            top_p=llm_spec._top_p,
            max_token=llm_spec._max_token,
            prompt_batch_size=llm_spec.prompt_batch_size,
            device=llm_spec.device,
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
    def deserialize(cls, data):
        return cls(
            GPT4AllLLMPortObjectSpec.deserialize(data),
            system_prompt_template=data["system_prompt_template"],
            prompt_template=data["prompt_template"],
        )


class GPT4AllChatModelPortObject(GPT4AllLLMPortObject, ChatModelPortObject):
    @property
    def spec(self) -> GPT4AllChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> LLMChatModelAdapter:
        llm = super().create_model(ctx)
        system_prompt_template = self.spec.system_prompt_template
        prompt_template = self.spec.prompt_template

        return LLMChatModelAdapter(
            llm=llm,
            system_prompt_template=system_prompt_template,
            prompt_template=prompt_template,
        )


gpt4all_llm_port_type = knext.port_type(
    "GPT4All LLM", GPT4AllLLMPortObject, GPT4AllLLMPortObjectSpec
)

gpt4all_chat_model_port_type = knext.port_type(
    "GPT4All Chat Model", GPT4AllChatModelPortObject, GPT4AllChatModelPortObjectSpec
)


def is_valid_model(model_path: str):
    import os

    if not model_path:
        raise knext.InvalidParametersError("Path to local model is missing")

    if not os.path.isfile(model_path):
        raise knext.InvalidParametersError(f"No file found at path: {model_path}")

    if not model_path.endswith(".gguf"):
        raise knext.InvalidParametersError(
            "Models needs to be of type '.gguf'. Find the latest models at: https://gpt4all.io/"
        )


@knext.node(
    "Local GPT4All LLM Connector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Local Large Language Model",
    ],
)
@knext.output_port(
    "GPT4All LLM",
    "A GPT4All large language model.",
    gpt4all_llm_port_type,
)
class GPT4AllLLMConnector:
    """
    Connects to a locally installed GPT4All LLM.

    This connector allows you to connect to a local GPT4All LLM. To get started,
    you need to download a specific model from the [GPT4All](https://gpt4all.io/index.html) model explorer on the website.
    It is not needed to install the GPT4All software. Once you have downloaded the model, specify its file path in the
    configuration dialog to use it.

    Some models (e.g. Llama 2) have been fine-tuned for chat applications,
    so they might behave unexpectedly if their prompts don't follow a chat like structure:

        User: <The prompt you want to send to the model>
        Assistant:

    Use the prompt template for the specific model from the
    [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json)
    if one is provided.

    The currently supported models are based on GPT-J, LLaMA, MPT, Replit, Falcon and StarCoder.

    For more information and detailed instructions on downloading compatible models, please visit the [GPT4All GitHub repository](https://github.com/nomic-ai/gpt4all).

    Note: This node can not be used on the KNIME Hub, as the models can't be embedded into the workflow due to their large size.
    """

    settings = GPT4AllInputSettings()
    params = GPT4AllModelParameterSettings(since_version="5.2.0")

    def configure(self, ctx: knext.ConfigurationContext) -> GPT4AllLLMPortObjectSpec:
        is_valid_model(self.settings.local_path)
        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> GPT4AllLLMPortObject:
        return GPT4AllLLMPortObject(self.create_spec())

    def create_spec(self) -> GPT4AllLLMPortObjectSpec:
        n_threads = None if self.settings.n_threads == 0 else self.settings.n_threads

        return GPT4AllLLMPortObjectSpec(
            local_path=self.settings.local_path,
            n_threads=n_threads,
            temperature=self.params.temperature,
            top_k=self.params.top_k,
            top_p=self.params.top_p,
            max_token=self.params.max_token,
            prompt_batch_size=self.params.prompt_batch_size,
            device=self.params.device,
        )


@knext.node(
    "Local GPT4All Chat Model Connector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Local Large Language Model",
    ],
)
@knext.output_port(
    "GPT4All Chat Model",
    "A GPT4All chat model.",
    gpt4all_chat_model_port_type,
)
class GPT4AllChatModelConnector:
    """
    Connects to a locally installed GPT4All LLM.

    This connector allows you to connect to a local GPT4All LLM. To get started,
    you need to download a specific model from the [GPT4All](https://gpt4all.io/index.html) model explorer on the website.
    It is not needed to install the GPT4All software. Once you have downloaded the model, specify its file path in the
    configuration dialog to use it.

    It is recommended to use models (e.g. Llama 2) that have been fine-tuned for chat applications. For model specifications
    including prompt templates, see [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json).

    The currently supported models are based on GPT-J, LLaMA, MPT, Replit, Falcon and StarCoder.

    For more information and detailed instructions on downloading compatible models, please visit the
    [GPT4All GitHub repository](https://github.com/nomic-ai/gpt4all).

    Note: This node can not be used on the KNIME Hub, as the models can't be embedded into the workflow due to their large size.
    """

    settings = GPT4AllInputSettings()
    templates = GPT4AllPromptSettings()
    params = GPT4AllModelParameterSettings()

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> GPT4AllChatModelPortObjectSpec:
        is_valid_model(self.settings.local_path)
        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> GPT4AllChatModelPortObject:
        return GPT4AllChatModelPortObject(self.create_spec())

    def create_spec(self) -> GPT4AllChatModelPortObjectSpec:
        n_threads = None if self.settings.n_threads == 0 else self.settings.n_threads

        llm_spec = GPT4AllLLMPortObjectSpec(
            local_path=self.settings.local_path,
            n_threads=n_threads,
            temperature=self.params.temperature,
            top_p=self.params.top_p,
            top_k=self.params.top_k,
            max_token=self.params.max_token,
            prompt_batch_size=self.params.prompt_batch_size,
            device=self.params.device,
        )

        return GPT4AllChatModelPortObjectSpec(
            llm_spec=llm_spec,
            system_prompt_template=self.templates.system_prompt_template,
            prompt_template=self.templates.prompt_template,
        )


_embeddings4all_model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"


# TODO: Delete the wrapper if Langchain is > 0.2.x and instantiate Embedd4All class
class _GPT4ALLEmbeddings(GPT4AllEmbeddings):
    model_name: str
    model_path: str
    num_threads: Optional[int] = None
    allow_download: bool = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that GPT4All library is installed."""

        try:
            values["client"] = Embed4All(
                model_name=values["model_name"],
                model_path=values["model_path"],
                n_threads=values.get("num_threads"),
                allow_download=values["allow_download"],
            )
            return values
        except ConnectionError:
            raise knext.InvalidParametersError(
                "Connection error. Please ensure that your internet connection is enabled to download the model."
            )


class Embeddings4AllPortObjectSpec(EmbeddingsPortObjectSpec):
    """The Embeddings4All port object spec."""

    def __init__(self, num_threads: int = 0) -> None:
        super().__init__()
        self._num_threads = num_threads

    @property
    def num_threads(self) -> int:
        return self._num_threads

    def serialize(self) -> dict:
        return {
            "num_threads": self._num_threads,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["num_threads"])


class Embeddings4AllPortObject(EmbeddingsPortObject, FilestorePortObject):
    """
    The Embeddings4All port object.

    The port object copies the Embeddings4All model into a filestore in order
    to make workflows containing such models portable.
    """

    def __init__(
        self,
        spec: EmbeddingsPortObjectSpec,
        model_name: str = _embeddings4all_model_name,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(spec)
        self._model_name = model_name
        self._model_path = model_path

    @property
    def spec(self) -> Embeddings4AllPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> Embeddings:
        try:
            return _GPT4ALLEmbeddings(
                model_name=self._model_name,
                model_path=self._model_path,
                num_threads=self.spec.num_threads,
                allow_download=False,
            )
        except Exception as e:
            unsupported_model_exception = (
                "Unable to instantiate model: Unsupported model architecture: bert"
            )
            if str(e) == unsupported_model_exception:
                raise knext.InvalidParametersError(
                    "An incompatible embeddings mode has been detected. Please obtain a more recent version."
                    "For additional details on available models, please refer to: "
                    "https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models3.json"
                )
            raise ValueError(f"The model at path {self.model_path} is not valid.")

    def write_to(self, file_path: str) -> None:
        os.makedirs(file_path)
        if self._model_path:
            # should be verified in the connector
            shutil.copy(
                os.path.join(self._model_path, self._model_name),
                os.path.join(file_path, self._model_name),
            )
        else:
            _GPT4ALLEmbeddings(
                model_path=file_path,
                model_name=_embeddings4all_model_name,
                num_threads=1,
                allow_download=True,
            )

    @classmethod
    def read_from(
        cls, spec: Embeddings4AllPortObjectSpec, file_path: str
    ) -> "Embeddings4AllPortObject":
        model_name = os.listdir(file_path)[0]
        return cls(spec, model_name, file_path)


embeddings4all_port_type = knext.port_type(
    "GPT4All Embeddings", Embeddings4AllPortObject, Embeddings4AllPortObjectSpec
)


class ModelRetrievalOptions(knext.EnumParameterOptions):
    DOWNLOAD = (
        "Download",
        "Downloads the model from GPT4All during execution. Requires an internet connection.",
    )
    READ = ("Read", "Reads the model from the local file system.")


@knext.node(
    "GPT4All Embeddings Connector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    gpt4all_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Local RAG",
        "Local Retrieval Augmented Generation",
    ],
)
@knext.output_port(
    "GPT4All Embeddings model",
    "A GPT4All Embeddings model that calculates embeddings on the local machine.",
    embeddings4all_port_type,
)
class Embeddings4AllConnector:
    """
    Connects to an embeddings model that runs on the local machine.

    Connect to an embeddings model that runs on the local machine via GPT4All.
    The default model was trained on sentences and short paragraphs of English text.
    It ignores special characters like 'ß' i.e. the embeddings for 'Schloß' are the same as for 'Schlo'.
    If downstream nodes fail with 'Execute failed: Error while sending a command.', then this is likely
    caused by an input that consists entirely of characters the model doesn't support.

    Note: Unlike the other GPT4All nodes, this node can be used on the KNIME Hub.
    """

    model_retrieval = knext.EnumParameter(
        "Model retrieval",
        "Defines how the model is retrieved during execution.",
        ModelRetrievalOptions.DOWNLOAD.name,
        ModelRetrievalOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    model_path = knext.LocalPathParameter(
        "Path to model", "The local file system path to the model."
    ).rule(
        knext.OneOf(model_retrieval, [ModelRetrievalOptions.READ.name]),
        knext.Effect.SHOW,
    )

    num_threads = knext.IntParameter(
        "Number of threads",
        """The number of threads the model uses. 
        More threads may reduce the runtime of queries to the model.
        Default is 0, then the number of threads are determined automatically.""",
        0,
        min_value=0,
        is_advanced=True,
    )

    def configure(self, ctx) -> Embeddings4AllPortObjectSpec:
        return self._create_spec()

    def _create_spec(self) -> Embeddings4AllPortObjectSpec:
        n_threads = None if self.num_threads == 0 else self.num_threads
        return Embeddings4AllPortObjectSpec(n_threads)

    def execute(self, ctx) -> Embeddings4AllPortObject:
        if self.model_retrieval == ModelRetrievalOptions.DOWNLOAD.name:
            model_path = None
            model_name = _embeddings4all_model_name
        else:
            if not os.path.exists(self.model_path):
                raise ValueError(
                    f"The provided model path {self.model_path} does not exist."
                )
            model_path, model_name = os.path.split(self.model_path)
            try:
                _GPT4ALLEmbeddings(
                    model_name=model_name,
                    model_path=model_path,
                    n_threads=self.num_threads,
                    allow_download=False,
                )
            except Exception as e:
                unsupported_model_exception = (
                    "Unable to instantiate model: Unsupported model architecture: bert"
                )
                if str(e) == unsupported_model_exception:
                    raise knext.InvalidParametersError(
                        "An incompatible embeddings mode has been detected. Please obtain a more recent version."
                        "For additional details on available models, please refer to: "
                        "https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models3.json"
                    )
                raise ValueError(f"The model at path {self.model_path} is not valid.")

        return Embeddings4AllPortObject(
            self._create_spec(),
            model_name=model_name,
            model_path=model_path,
        )
