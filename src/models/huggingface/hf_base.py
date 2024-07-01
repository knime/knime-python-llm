# KNIME / own imports
from typing import Any, Optional

from pydantic import root_validator
import requests
import knime.extension as knext
from ..base import (
    model_category,
    GeneralRemoteSettings,
)

from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM


hf_icon = "icons/huggingface.png"
hf_category = knext.category(
    path=model_category,
    level_id="huggingface",
    name="Hugging Face",
    description="",
    icon=hf_icon,
)


class HFModelSettings(GeneralRemoteSettings):
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


class HFLLM(LLM):
    """Custom implementation backed by huggingface_hub.InferenceClient.
    We can't use the implementation of langchain_community because it always requires an api token (and is
    probably going to be deprecated soon) and we also can't use the langchain_huggingface implementation
    since it has torch as a required dependency."""

    model: str
    """Can be a repo id on hugging face hub or the url of a TGI server."""
    hf_api_token: Optional[str] = None
    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = None
    client: Any
    seed: Optional[int] = None

    def _llm_type(self):
        return "hfllm"

    @root_validator()
    def validate_values(cls, values: dict) -> dict:
        values["client"] = InferenceClient(
            model=values["model"], timeout=120, token=values.get("hf_api_token")
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        client: InferenceClient = self.client
        try:
            return client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
                typical_p=self.typical_p,
                seed=self.seed,
            )
        except Exception as ex:
            raise_for(ex)


def raise_for(exception: Exception, default: Optional[Exception] = None):
    if isinstance(exception, requests.exceptions.ProxyError):
        raise RuntimeError(
            "Failed to establish connection due to a proxy error. Validate your proxy settings."
        ) from exception
    if isinstance(exception, requests.exceptions.Timeout):
        raise RuntimeError(
            "The connection to Hugging Face Hub timed out."
        ) from exception
    if default:
        raise default from exception
    raise exception
