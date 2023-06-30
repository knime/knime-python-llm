# TODO: Have the same naming standard for all specs and objects in general as well as in the configure and execute methods


import knime.extension as knext
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
    model_category,

    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
)

from langchain import HuggingFaceHub
from langchain.llms import HuggingFaceTextGenInference
from langchain.embeddings import HuggingFaceEmbeddings


huggingface_icon = "icons/huggingface.svg"
huggingface = knext.category(
    path=model_category,
    level_id="hugging",
    name="Hugging Face",
    description="",
    icon=huggingface_icon,
)
#TODO: Remove serialize

class HuggingFaceTextGenInfLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
            self, 
            inference_server_url, 
            max_new_tokens,
            top_k,
            top_p,
            typical_p,
            temperature,
            repetition_penalty
            ) -> None:
        super().__init__()
        self._inference_server_url = inference_server_url
        self._max_new_tokens = max_new_tokens
        self._top_k = top_k
        self._top_p = top_p
        self._typical_p = typical_p
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty



    def serialize(self) -> dict:
        return {
            "inference_server_url": self._inference_server_url, 
            "max_new_tokens": self._max_new_tokens,
            "top_k": self._top_k,
            "top_p": self._top_p,
            "typical_p": self._typical_p,
            "temperature": self._temperature,
            "repetition_penalty": self._repetition_penalty,
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
        spec = self.spec.serialize()

        return HuggingFaceTextGenInference(
            inference_server_url=spec["inference_server_url"],
            max_new_tokens=spec["max_new_tokens"],
            top_k=spec["top_k"],
            top_p=spec["top_p"],
            typical_p=spec["typical_p"],
            temperature=spec["temperature"],
            repetition_penalty=spec["repetition_penalty"],
        )
    
huggingface_textGenInf_llm_port_type = knext.port_type("Huggingface LLM", HuggingFaceTextGenInfLLMPortObject, HuggingFaceTextGenInfLLMPortObjectSpec)

class HuggingFacHubLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
            self, 
            credentials, 
            repo_id,
            model_kwargs,
            ) -> None:
        super().__init__()
        self._credentials = credentials
        self._repo_id = repo_id
        self._model_kwargs = model_kwargs

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials, 
            "repo_id": self._repo_id,
            "model_kwargs": self._model_kwargs,
            }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["credentials"], 
            data["repo_id"],
            data["model_kwargs"],
        )

class HuggingFaceHubLLMPortObject(LLMPortObject):

    def __init__(self, spec: HuggingFacHubLLMPortObjectSpec) -> None:
        super().__init__(spec)

    def create_model(self, ctx):
        import json
        spec = self.spec.serialize()

        return HuggingFaceHub(
            huggingfacehub_api_token=ctx.get_credentials(spec["credentials"]).password,
            repo_id=spec["repo_id"],
            model_kwargs=json.loads(spec["model_kwargs"]),
        )

huggingface_hub_llm_port_type = knext.port_type("Huggingface LLM", HuggingFaceHubLLMPortObject, HuggingFacHubLLMPortObjectSpec)

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

huggingface_embeddings_port_type = knext.port_type("Huggingface Embeddings Port Type", HuggingFacePortObject, HuggingFaceEmbeddingsPortObjectSpec)


@knext.parameter_group(label="Huggingface TextGen Inference Settings")
class TextGenInferenceInputSettings:
    
    server_url = knext.StringParameter(
        label="Inference Server Url",
        description="The URL of the inference server to use.",
        default_value="",
    )

    max_new_tokens = knext.IntParameter(
        label="Max new tokens",
        description="The maximum number of tokens to generate.",
        default_value=1,
        min_value=1,
    )

    top_k = knext.IntParameter(
        label="Top k",
        description="The number of top-k tokens to consider when generating text.",
        default_value=1,
        min_value=1,
    )

    top_p = knext.DoubleParameter(
        label="Top p",
        description="The cumulative probability threshold for generating text.",
        default_value=1.0,
        max_value=1.0,
        min_value=0.0,
    )

    typical_p = knext.DoubleParameter(
        label="Typical p",
        description="The typical probability threshold for generating text.",
        default_value=1.0,
        max_value=1.0,
        min_value=0.0,
    )

    temperature = knext.DoubleParameter(
        label="Temperature",
        description="The temperature to use when generating text.",
        default_value=1.0,
        max_value=1.0,
        min_value=0.0,
    )

    repetition_penalty = knext.DoubleParameter(
        label="Repetition penalty",
        description="The repetition penalty to use when generating text.",
        default_value=1.0,
        min_value=0.0,
    )

@knext.node(
    "HF TextGen Inference Configurator",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface,
)
@knext.output_port(
    "Huggingface TextGen Inference Configuration",
    "A large language model text gen configuration.",
    huggingface_textGenInf_llm_port_type,
)
class HuggingfaceTextGenInferenceConfigurator:

    settings = TextGenInferenceInputSettings()

    def configure(self, ctx: knext.ConfigurationContext):
        return self.get_spec_content()

    def execute(self, ctx: knext.ExecutionContext):
        return HuggingFaceTextGenInfLLMPortObject(self.get_spec_content())
    
    def get_spec_content(self):
        return HuggingFaceTextGenInfLLMPortObjectSpec(
            self.settings.server_url,
            self.settings.max_new_tokens,
            self.settings.top_k,
            self.settings.top_p,
            self.settings.typical_p,
            self.settings.temperature,
            self.settings.repetition_penalty
        )

@knext.node(
    "HF Hub LLM Configurator",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface,
)
@knext.output_port(
    "Hugging Face LLM", "A large language model from Hugging Face.", huggingface_hub_llm_port_type
)
class HuggingfaceHubConfigurator:

    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing the Huggingface API key",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )

    repo_id = knext.StringParameter(
        label="Repo ID",
        description="Model name to use e.g 'Writer/camel-5b-hf'",
        default_value=""
    )

    model_kwargs = knext.StringParameter(
        label="Model kwargs",
        description="Key word arguments to pass to the model. Expected to be a string type dictionary.",
        default_value="{'argument'}"
    )

    def configure(self, ctx: knext.ConfigurationContext):
        return self.get_spec_content()

    def execute(self, ctx: knext.ExecutionContext):
        return HuggingFaceHubLLMPortObject(
            self.get_spec_content()
        )

    def get_spec_content(self):
        return HuggingFacHubLLMPortObjectSpec(
            self.credentials_param,
            self.repo_id,
            self.model_kwargs,
        )
    

@knext.node(
    "HF Embeddings Configurator",
    knext.NodeType.SOURCE,
    huggingface_icon,
    category=huggingface,
)
@knext.output_port(
    "Hugging Face Embeddings", "An embeddings model configuration from Huggingface Hub.", huggingface_embeddings_port_type
)
class HuggingFaceEmbeddingsConfigurator:
        
    def configure(self, ctx: knext.ConfigurationContext):
        return HuggingFaceEmbeddingsPortObjectSpec()

    def execute(self, ctx: knext.ExecutionContext):
        return HuggingFacePortObject(
            HuggingFaceEmbeddingsPortObjectSpec()
        )