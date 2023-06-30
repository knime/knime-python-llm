# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods


import knime.extension as knext

from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    model_category,
)

from langchain.llms import GPT4All


class GPT4AllLLMPortbjectSpec(LLMPortObjectSpec):
    def __init__(self, local_path) -> None:
        super().__init__()
        self._local_path = local_path



    def serialize(self) -> dict:
        return {
            "local_path": self._local_path,
            }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["local_path"],
        )

class GPT4AllLLMPortbject(LLMPortObject):

    def __init__(self, spec: GPT4AllLLMPortbjectSpec) -> None:
        super().__init__(spec)

    def create_model(self, ctx):
        return GPT4All(
            model=self.spec._local_path, 
            verbose=True
        )

gpt4all_port_type = knext.port_type("GPT4ALL LLM", GPT4AllLLMPortbject, GPT4AllLLMPortbjectSpec)

gpt4all_icon = "icons/gtp4all.png"
gpt4all = knext.category(
    path=model_category,
    level_id="gpt4all",
    name="GPT4All",
    description="",
    icon=gpt4all_icon,
)

@knext.parameter_group(label="GPT4All Settings")
class GPT4AllInputSettings:
    
    # TODO: More settings
    local_path = knext.StringParameter(
        label="Model path",
        description="Path to the pre-trained GPT4All model file eg. my/path/model.bin.",
        default_value="",
    )

@knext.node(
    "GPT4All LLM Configurator",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all,
)
@knext.output_port(
    "GPT4All LLM Configuration",
    "A GPT4All large language model configuration.",
    gpt4all_port_type,
)
class GPT4AllLLMConfigurator:

    settings = GPT4AllInputSettings()

    def configure(self, ctx: knext.ConfigurationContext):
        return GPT4AllLLMPortbjectSpec(self.settings.local_path)

    def execute(self, ctx: knext.ExecutionContext):

        return GPT4AllLLMPortbject(
            GPT4AllLLMPortbjectSpec(local_path=self.settings.local_path)
        )
