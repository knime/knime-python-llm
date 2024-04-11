from langchain.chat_models.openai import ChatOpenAI
import knime.extension as knext
import knime.api.schema as ks
from knime.extension import ExecutionContext
from urllib.parse import urlparse, urlunparse
from .base import GeneralSettings
import requests
from typing import Callable


from .base import (
    model_category,
    ChatModelPortObjectSpec,
    ChatModelPortObject,
)


hub_connector_icon = "icons/Hub_AI_connector.png"

knime_category = knext.category(
    path=model_category,
    level_id="knime",
    name="KNIME",
    description="Models that connect to the KNIME Hub.",
    icon=hub_connector_icon,
)


class KnimeHubChatModelPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(
        self,
        auth_spec: ks.HubAuthenticationPortObjectSpec,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        super().__init__()
        self._auth_spec = auth_spec
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p

    @property
    def auth_spec(self) -> ks.HubAuthenticationPortObjectSpec:
        return self._auth_spec

    @property
    def model_name(self) -> str:
        return self._model_name

    def serialize(self) -> dict:
        return {
            "auth": self.auth_spec.serialize(),
            "model_name": self.model_name,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def top_p(self) -> float:
        return self._top_p

    @classmethod
    def deserialize(cls, data: dict, java_callback):
        return cls(
            ks.HubAuthenticationPortObjectSpec.deserialize(data["auth"], java_callback),
            data["model_name"],
            data["max_tokens"],
            data["temperature"],
            data["top_p"],
        )


class KnimeHubChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> KnimeHubChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx: ExecutionContext) -> ChatOpenAI:
        auth_spec = self.spec.auth_spec
        return ChatOpenAI(
            model=self.spec.model_name,
            default_headers=_create_authorization_headers(auth_spec),
            openai_api_base=_extract_api_base(auth_spec),
            openai_api_key="placeholder",
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
        )


def _create_authorization_headers(
    auth_spec: ks.HubAuthenticationPortObjectSpec,
) -> dict[str, str]:
    return {
        "Authorization": f"{auth_spec.auth_schema} {auth_spec.auth_parameters}",
    }


def _extract_api_base(auth_spec: ks.HubAuthenticationPortObjectSpec) -> str:
    hub_url = auth_spec.hub_url
    parsed_url = urlparse(hub_url)
    # drop params, query and fragment
    ai_proxy_url = (parsed_url.scheme, parsed_url.netloc, "ai-proxy/v1", "", "", "")
    return urlunparse(ai_proxy_url)


knime_chat_model_port_type = knext.port_type(
    "KNIME Hub Chat Model", KnimeHubChatModelPortObject, KnimeHubChatModelPortObjectSpec
)


def _list_models_in_dialog(
    mode: str,
) -> Callable[[knext.DialogCreationContext], list[str]]:
    def list_models(ctx: knext.DialogCreationContext) -> list[str]:
        if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
            return _list_models(auth_spec, mode)
        return [""]

    return list_models


def _list_models(auth_spec, mode: str) -> list[str]:
    model_list = [
        model_data["model_name"]
        for model_data in _get_model_data(auth_spec)
        if mode == _get_mode(model_data)
    ]
    if len(model_list) == 0:
        model_list.append("")
    else:
        model_list.sort()
        return model_list


def _get_mode(model_data: dict):
    model_info: dict = model_data.get("model_info")
    if model_info:
        return model_info.get("mode")
    return None


def _get_model_data(auth_spec):
    api_base = _extract_api_base(auth_spec)
    model_info = api_base + "/model/info"
    response = requests.get(
        url=model_info, headers=_create_authorization_headers(auth_spec)
    )
    if response.status_code == 404:
        raise ValueError(
            "The AI proxy is not reachable. Is it activated in the connected KNIME Hub?"
        )
    response.raise_for_status()
    return response.json()["data"]


class ModelSettings(GeneralSettings):
    max_tokens = knext.IntParameter(
        label="Maximum Response Length (token)",
        description="""
        The maximum number of tokens to generate.

        The token count of your prompt plus 
        max_tokens cannot exceed the model's context length.
        """,
        default_value=200,
        min_value=1,
    )

    # Altered from GeneralSettings because OpenAI has temperatures going up to 2
    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 2.0. 
        Higher values means the model will take more risks. 
        Try 0.9 for more creative applications, and 0 for ones with a well-defined answer.
        It is generally recommend altering this or top_p but not both.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=2.0,
    )


@knext.node(
    name="KNIME Hub Chat Model Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=hub_connector_icon,
    category=knime_category,
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_port(
    name="KNIME Hub Chat Model",
    description="A chat model that connects to the KNIME hub to make requests.",
    port_type=knime_chat_model_port_type,
)
class KnimeHubChatModelConnector:
    """Connects to a Chat Model configured in the AI proxy of the connected KNIME Hub.

    Connects to a Chat Model configured in the AI proxy of the connected KNIME Hub using the authentication
    provided via the input port.
    """

    model_name = knext.StringParameter(
        "Model", "Select the model to use.", choices=_list_models_in_dialog("chat")
    )

    model_settings = ModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> KnimeHubChatModelPortObjectSpec:
        return self._create_spec(authentication)

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> KnimeHubChatModelPortObject:
        available_models = _list_models(authentication.spec, "chat")
        if self.model_name not in available_models:
            raise knext.InvalidParametersError(
                f"The selected model {self.model_name} is not served by the connected Hub."
            )
        return KnimeHubChatModelPortObject(self._create_spec(authentication.spec))

    def _create_spec(
        self, authentication: ks.HubAuthenticationPortObjectSpec
    ) -> KnimeHubChatModelPortObjectSpec:
        return KnimeHubChatModelPortObjectSpec(
            authentication,
            self.model_name,
            max_tokens=self.model_settings.max_tokens,
            temperature=self.model_settings.temperature,
            top_p=self.model_settings.top_p,
        )
