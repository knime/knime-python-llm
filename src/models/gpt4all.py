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
    description="",
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


class GPT4AllLLMPortbjectSpec(LLMPortObjectSpec):
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


class GPT4AllLLMPortbject(LLMPortObject):
    def __init__(self, spec: GPT4AllLLMPortbjectSpec) -> None:
        super().__init__(spec)

    def create_model(self, ctx):
        return GPT4All(model=self.spec._local_path, verbose=True)


gpt4all_llm_port_type = knext.port_type(
    "GPT4ALL LLM", GPT4AllLLMPortbject, GPT4AllLLMPortbjectSpec
)


# TODO: Extend node descriptions
@knext.node(
    "GPT4All LLM Configurator",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all,
)
@knext.output_port(
    "GPT4All LLM Configuration",
    "A GPT4All large language model configuration.",
    gpt4all_llm_port_type,
)
class GPT4AllLLMConfigurator:
    """

    Configuration for a local GPT4All LLM.

    Configures a local GPT4All LLM. Use the installer from [GPT4All](https://gpt4all.io/index.html) to download
    a specific model and enter its path in the 'Model path' setting to use it.

    """

    settings = GPT4AllInputSettings()

    def configure(self, ctx: knext.ConfigurationContext) -> GPT4AllLLMPortbjectSpec:
        return GPT4AllLLMPortbjectSpec(self.settings.local_path)

    def execute(self, ctx: knext.ExecutionContext) -> GPT4AllLLMPortbject:
        return GPT4AllLLMPortbject(
            GPT4AllLLMPortbjectSpec(local_path=self.settings.local_path)
        )
