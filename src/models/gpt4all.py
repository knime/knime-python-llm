# KNIME / own imports
import knime.extension as knext
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    model_category,
)

# Langchain imports
from langchain.llms import GPT4All


gpt4all_icon = "icons/gpt4all.png"
gpt4all = knext.category(
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
        description="Path to the pre-trained GPT4All model file eg. my/path/model.gguf.",
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
    "GPT4All LLM Connector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all,
)
@knext.output_port(
    "GPT4All LLM",
    "A GPT4All large language model.",
    gpt4all_llm_port_type,
)
class GPT4AllLLMConnector:
    """
    Connects to a locally installed GPT4ALL LLM.

    This connector allows you to connect to a local GPT4All LLM. To get started,
    you need to download a specific model from the [GPT4All](https://gpt4all.io/index.html) model explorer on the website.
    It is not needed to install the GPT4All software. Once you have downloaded the model, specify its file path in the
    configuration dialog to use it.

    **Important Note:** GPT4All discontinued support for the old .bin model format and switched to the new .gguf format.
    Because of this switch, workflows using models in .bin format will no longer work.
    You can find models in the new format on the [GPT4All](https://gpt4all.io/index.html) website or on [Hugging Face Hub](https://huggingface.co/).

    Some models (e.g. Llama 2) have been fine-tuned for chat applications,
    so they might behave unexpectedly if their prompts don't follow a chat like structure:

        User: <The prompt you want to send to the model>
        Assistant:

    Use the prompt template for the specific model from the
    [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json)
    if one is provided.

    The currently supported models are based on GPT-J, LLaMA, MPT, Replit, Falcon and StarCoder.

    For more information and detailed instructions on downloading compatible models, please visit the [GPT4All GitHub repository](https://github.com/nomic-ai/gpt4all).
    """

    settings = GPT4AllInputSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> GPT4AllLLMPortObjectSpec:
        is_valid_model(self.settings.local_path)
        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> GPT4AllLLMPortObject:
        return GPT4AllLLMPortObject(self.create_spec())

    def create_spec(self) -> GPT4AllLLMPortObjectSpec:
        return GPT4AllLLMPortObjectSpec(local_path=self.settings.local_path)
