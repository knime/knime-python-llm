# KNIME / own imports
import knime.extension as knext
from knime.extension.nodes import FilestorePortObject
from models.base import EmbeddingsPortObjectSpec
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    model_category,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
)

# Langchain imports
from langchain.llms import GPT4All
from langchain.embeddings.base import Embeddings
from pydantic import root_validator, BaseModel
from gpt4all import GPT4All as _GPT4All
import shutil
import os

from typing import Optional, Any, Dict, List


gpt4all_icon = "icons/gpt4all.png"
gpt4all_category = knext.category(
    path=model_category,
    level_id="gpt4all",
    name="GPT4All",
    description="Contains nodes for connecting to GPT4ALL models.",
    icon=gpt4all_icon,
)


# TODO: Add more configuration options https://python.langchain.com/docs/modules/model_io/models/llms/integrations/gpt4all
@knext.parameter_group(label="GPT4All Settings")
class GPT4AllInputSettings:
    local_path = knext.StringParameter(
        label="Model path",
        description="Path to the pre-trained GPT4All model file eg. my/path/model.bin.",
        default_value="",
    )


class GPT4AllLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(self, local_path) -> None:
        super().__init__()
        self._local_path = local_path

    @property
    def local_path(self):
        return self._local_path

    def serialize(self) -> dict:
        return {
            "local_path": self._local_path,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["local_path"])


class GPT4AllLLMPortObject(LLMPortObject):
    def __init__(self, spec: GPT4AllLLMPortObjectSpec) -> None:
        super().__init__(spec)

    def create_model(self, ctx):
        return GPT4All(model=self.spec._local_path)


gpt4all_llm_port_type = knext.port_type(
    "GPT4ALL LLM", GPT4AllLLMPortObject, GPT4AllLLMPortObjectSpec
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
    Connects to a locally installed GPT4ALL LLM.

    This connector allows you to connect to a LLM that was installed locally with GPT4ALL. To get started,
    you need to download a specific model from [GPT4All](https://gpt4all.io/index.html) using the installer.
    Once you have downloaded the model, specify its file path in the 'Model path' setting to use it.

    For more information and detailed instructions on downloading compatible models, please visit the [GPT4All GitHub repository](https://github.com/nomic-ai/gpt4all).
    """

    settings = GPT4AllInputSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> GPT4AllLLMPortObjectSpec:
        return GPT4AllLLMPortObjectSpec(self.settings.local_path)

    def execute(self, ctx: knext.ExecutionContext) -> GPT4AllLLMPortObject:
        return GPT4AllLLMPortObject(
            GPT4AllLLMPortObjectSpec(local_path=self.settings.local_path)
        )


_embeddings4all_model_name = "ggml-all-MiniLM-L6-v2-f16.bin"


class _Embeddings4All(BaseModel, Embeddings):
    model_name: str
    model_path: str
    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["client"] = _GPT4All(
            values["model_name"], model_path=values["model_path"]
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

    def create_model(self, ctx) -> Embeddings:
        return _Embeddings4All(model_name=self._model_name, model_path=self._model_path)

    @property
    def full_model_path(self):
        return

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
    The default model is for English text and ignores special characters like 'ß' i.e. the embeddings for 'Schloß' are the same as for 'Schlo'.
    If downstream nodes fail with 'Execute failed: Error while sending a command.', then this is likely caused by an input that
    consists entirely of characters the model doesn't support.
    """

    # TODO add advanced threads option

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

    def configure(self, ctx):
        return Embeddings4AllPortObjectSpec()

    def execute(self, ctx):
        model_retrieval = self.model_retrieval
        if model_retrieval == ModelRetrievalOptions.DOWNLOAD.name:
            model_path = None
            model_name = _embeddings4all_model_name
        else:
            if not os.path.exists(self.model_path):
                raise ValueError(
                    f"The provided model path {self.model_path} does not exist."
                )
            model_path, model_name = os.path.split(self.model_path)
            try:
                _Embeddings4All(model_name=model_name, model_path=model_path)
            except:
                raise ValueError(f"The model at path {self.model_path} is not valid.")

        return Embeddings4AllPortObject(
            Embeddings4AllPortObjectSpec(), model_name=model_name, model_path=model_path
        )
