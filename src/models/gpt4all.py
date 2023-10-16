# KNIME / own imports
import knime.extension as knext
from knime.extension.nodes import FilestorePortObject
from models.base import EmbeddingsPortObjectSpec
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
from pydantic import model_validator, BaseModel, ValidationError
from typing import Any, Dict, List, Optional
from gpt4all import GPT4All as _GPT4All
import shutil
import os

# Langchain imports
from langchain.llms import GPT4All
from langchain.embeddings.base import Embeddings

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
    local_path = knext.StringParameter(
        label="Model path",
        description="""Path to the pre-trained GPT4All model file eg. my/path/model.bin.
        You can find the folder through seetings -> application in the gpt4all desktop application""",
        default_value="",
    )

    n_threads = knext.IntParameter(
        label="Thread Count",
        description="""Number of CPU threads used by GPT4All. Default is 0, then the number of threads 
        are determined automatically. 
        """,
        default_value=0,
        min_value=0,
        max_value=64,
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


class GPT4AllLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self,
        local_path: str,
        n_threads: int,
        temperature: float,
        top_k: int,
        top_p: float,
        max_token: int,
    ) -> None:
        super().__init__()
        self._local_path = local_path
        self._n_threads = n_threads
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._max_token = max_token

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

    def serialize(self) -> dict:
        return {
            "local_path": self._local_path,
            "n_threads": self._n_threads,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "max_token": self._max_token,
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
            )
        except ValidationError:
            raise knext.InvalidParametersError(
                """
                Could not validate model due to version incompatibility. Please provide a model based on one
                of the following infrastructures: GPT-J, LLaMA, MPT, Replit, Falcon and StarCoder
                """
            )


class GPT4AllChatModelPortObjectSpec(GPT4AllLLMPortObjectSpec, ChatModelPortObjectSpec):
    pass


class GPT4AllChatModelPortObject(GPT4AllLLMPortObject, ChatModelPortObject):
    @property
    def spec(self) -> GPT4AllChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> LLMChatModelAdapter:
        llm = super().create_model(ctx)
        return LLMChatModelAdapter(llm=llm)


gpt4all_llm_port_type = knext.port_type(
    "GPT4All LLM", GPT4AllLLMPortObject, GPT4AllLLMPortObjectSpec
)

gpt4all_chat_model_port_type = knext.port_type(
    "GPT4All Chat Model", GPT4AllChatModelPortObject, GPT4AllChatModelPortObjectSpec
)


@knext.node(
    "GPT4All LLM Connector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all_category,
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
    [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models.json)
    if one is provided.

    The currently supported models are based on GPT-J, LLaMA, MPT, Replit, Falcon and StarCoder.

    For more information and detailed instructions on downloading compatible models, please visit the [GPT4All GitHub repository](https://github.com/nomic-ai/gpt4all).
    """

    settings = GPT4AllInputSettings()
    params = GPT4AllModelParameterSettings(since_version="5.2.0")

    def configure(self, ctx: knext.ConfigurationContext) -> GPT4AllLLMPortObjectSpec:
        import os

        if not self.settings.local_path:
            raise knext.InvalidParametersError("Path to local model is missing")

        if not os.path.isfile(self.settings.local_path):
            raise knext.InvalidParametersError(
                f"No file found at path: {self.settings.local_path}"
            )

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
        )


@knext.node(
    "GPT4All Chat Model Connector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all_category,
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
    including prompt templates, see [GPT4All model list]
    (https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models.json).

    The currently supported models are based on GPT-J, LLaMA, MPT, Replit, Falcon and StarCoder.

    For more information and detailed instructions on downloading compatible models, please visit the
    [GPT4All GitHub repository](https://github.com/nomic-ai/gpt4all).
    """

    settings = GPT4AllInputSettings()
    params = GPT4AllModelParameterSettings()

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> GPT4AllChatModelPortObjectSpec:
        if not self.settings.local_path:
            raise knext.InvalidParametersError("Path to local model is missing.")

        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> GPT4AllChatModelPortObject:
        return GPT4AllChatModelPortObject(self.create_spec())

    def create_spec(self) -> GPT4AllChatModelPortObjectSpec:
        n_threads = None if self.settings.n_threads == 0 else self.settings.n_threads

        return GPT4AllChatModelPortObjectSpec(
            local_path=self.settings.local_path,
            n_threads=n_threads,
            temperature=self.params.temperature,
            top_p=self.params.top_p,
            top_k=self.params.top_k,
            max_token=self.params.max_token,
        )


_embeddings4all_model_name = "ggml-all-MiniLM-L6-v2-f16.bin"


class _Embeddings4All(BaseModel, Embeddings):
    model_name: str
    model_path: str
    num_threads: int = 1
    client: Any  #: :meta private:

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        values["client"] = _GPT4All(
            values["model_name"],
            model_path=values["model_path"],
            n_threads=values["num_threads"],
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = [self.embed_query(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        if text is None:
            raise ValueError("None values are not supported.")
        elif not text.strip():
            raise ValueError("Empty documents are not supported.")
        return self.client.model.generate_embedding(text)


class Embeddings4AllPortObjectSpec(EmbeddingsPortObjectSpec):
    """The Embeddings4All port object spec."""

    def __init__(self, num_threads: int = 1) -> None:
        super().__init__()
        self._num_threads = num_threads

    @property
    def num_threads(self) -> int:
        return self._num_threads

    def serialize(self) -> dict:
        return {"num_threads": self.num_threads}

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
        model_name=_embeddings4all_model_name,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(spec)
        self._model_name = model_name
        self._model_path = model_path

    @property
    def spec(self) -> Embeddings4AllPortObjectSpec:
        return super().spec

    def create_model(self, ctx) -> Embeddings:
        return _Embeddings4All(
            model_name=self._model_name,
            model_path=self._model_path,
            num_threads=self.spec.num_threads,
        )

    def write_to(self, file_path: str) -> None:
        os.makedirs(file_path)
        if self._model_path:
            # should be verified in the connector
            shutil.copy(
                os.path.join(self._model_path, self._model_name),
                os.path.join(file_path, self._model_name),
            )
        else:
            _Embeddings4All(model_path=file_path, model_name=_embeddings4all_model_name)

    @classmethod
    def read_from(
        cls, spec: Embeddings4AllPortObjectSpec, file_path: str
    ) -> "Embeddings4AllPortObject":
        model_name = os.listdir(file_path)[0]
        return cls(spec, model_name, file_path)


embeddings4all_port_type = knext.port_type(
    "Embeddings4All", Embeddings4AllPortObject, Embeddings4AllPortObjectSpec
)


class ModelRetrievalOptions(knext.EnumParameterOptions):
    DOWNLOAD = (
        "Download",
        "Downloads the model from GPT4All during execution. Requires an internet connection.",
    )
    READ = ("Read", "Reads the model from the local file system.")


@knext.node(
    "Embeddings4All Connector", knext.NodeType.SOURCE, gpt4all_icon, gpt4all_category
)
@knext.output_port(
    "Embeddings4All model",
    "An Embeddings4All model that calculates embeddings on the local machine.",
    embeddings4all_port_type,
)
class Embeddings4AllConnector:
    """
    Connects to an embeddings model that runs on the local machine.

    Connect to an embeddings model that runs on the local machine via GPT4All.
    The default model was trained on sentences and short paragrpahs of English text.
    It ignores special characters like 'ß' i.e. the embeddings for 'Schloß' are the same as for 'Schlo'.
    If downstream nodes fail with 'Execute failed: Error while sending a command.', then this is likely caused by an input that
    consists entirely of characters the model doesn't support.
    """

    model_retrieval = knext.EnumParameter(
        "Model retrieval",
        "Defines how the model is retrieved during execution.",
        ModelRetrievalOptions.DOWNLOAD.name,
        ModelRetrievalOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    model_path = knext.StringParameter(
        "Path to model", "The local file system path to the model."
    ).rule(
        knext.OneOf(model_retrieval, [ModelRetrievalOptions.READ.name]),
        knext.Effect.SHOW,
    )

    num_threads = knext.IntParameter(
        "Number of threads",
        """The number of threads the model uses. 
        More threads may reduce the runtime of queries to the model.""",
        1,
        min_value=1,
        is_advanced=True,
    )

    def configure(self, ctx):
        return Embeddings4AllPortObjectSpec(self.num_threads)

    def execute(self, ctx):
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
                _Embeddings4All(
                    model_name=model_name,
                    model_path=model_path,
                    num_threads=self.num_threads,
                )
            except:
                raise ValueError(f"The model at path {self.model_path} is not valid.")

        return Embeddings4AllPortObject(
            Embeddings4AllPortObjectSpec(self.num_threads),
            model_name=model_name,
            model_path=model_path,
        )
