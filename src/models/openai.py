# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------


# KNIME / own imports
import knime.extension as knext
import knime.extension.nodes as kn
from .base import (
    AIPortObjectSpec,
    LLMPortObjectSpec,
    LLMPortObject,
    llm_port_type,
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    model_category,
    GeneralRemoteSettings,
    CredentialsSettings,
    OutputFormatOptions,
)

# Other imports
import io
import re
import util
import time
import json
import base64
import tempfile
from typing import Callable, List, Optional
import logging

LOGGER = logging.getLogger(__name__)

openai_icon = "icons/openai/openai.png"
openai_category = knext.category(
    path=model_category,
    level_id="openai",
    name="OpenAI",
    description="",
    icon=openai_icon,
)

# == SETTINGS ==

_default_openai_api_base = "https://api.openai.com/v1"


def _to_choice(model: tuple[str, str, str]) -> knext.StringParameter.Choice:
    return knext.StringParameter.Choice(*model)


def _to_choices(
    models: list[tuple[str, str, str]],
) -> list[knext.StringParameter.Choice]:
    return list(map(_to_choice, models))


completion_models = _to_choices(
    [
        (
            "gpt-3.5-turbo-instruct",
            "GPT-3.5 Turbo Instruct",
            "Legacy instruct-style GPT-3.5 model; optimized for following single-turn instructions. Use only for backward compatibility; prefer GPT-4o mini. [Docs](https://platform.openai.com/docs/models/gpt-3.5-turbo-instruct)",
        ),
        (
            "babbage-002",
            "Babbage-002",
            "Legacy GPT-3.5 base model for simple tasks. Use only if migrating old applications; prefer GPT-4o mini for new projects. [Docs](https://platform.openai.com/docs/models/babbage-002)",
        ),
        (
            "davinci-002",
            "Davinci-002",
            "Legacy GPT-3.5 base model; more capable than Babbage-002 but deprecated. Use only for legacy compatibility; modern tasks should use GPT-4o or GPT-4.1. [Docs](https://platform.openai.com/docs/models/davinci-002)",
        ),
    ]
)
completion_default = "gpt-3.5-turbo-instruct"
chat_models = _to_choices(
    [
        (
            "gpt-3.5-turbo",
            "GPT-3.5 Turbo",
            "Legacy chat/text model; OpenAI recommends GPT-4o mini instead. Use only for backward compatibility. [Docs](https://platform.openai.com/docs/models/gpt-3.5-turbo)",
        ),
        (
            "gpt-3.5-turbo-16k",
            "GPT-3.5 Turbo (16k)",
            "Legacy 16k-context variant of GPT-3.5 Turbo. Use only for legacy apps needing longer context. [Docs](https://platform.openai.com/docs/models/gpt-3.5-turbo-16k-0613)",
        ),
        (
            "gpt-4",
            "GPT-4",
            "Legacy high-intelligence GPT model. Avoid for new projects; prefer GPT-4o or GPT-4.1. [Docs](https://platform.openai.com/docs/models/gpt-4)",
        ),
        (
            "gpt-4-32k",
            "GPT-4 (32k)",
            "Legacy 32k-context variant of GPT-4. Superseded by GPT-4o with larger context and lower cost.",
        ),
        (
            "gpt-4-turbo",
            "GPT-4 Turbo",
            "Legacy GPT-4 variant; superseded by GPT-4o. Use GPT-4o instead for cheaper, faster performance. [Docs](https://platform.openai.com/docs/models/gpt-4-turbo)",
        ),
        (
            "gpt-4o",
            "GPT-4o",
            "Fast, intelligent, flexible GPT model. Best general-purpose choice with multimodal support. [Docs](https://platform.openai.com/docs/models/gpt-4o)",
        ),
        (
            "gpt-4o-mini",
            "GPT-4o mini",
            "Smaller 4o variant for efficiency; supports multimodality. Great for lightweight or cost-sensitive tasks. [Docs](https://platform.openai.com/docs/models/gpt-4o-mini)",
        ),
        (
            "gpt-4.1",
            "GPT-4.1",
            "Smartest non-reasoning model. Choose when you need the highest quality for complex instructions. [Docs](https://platform.openai.com/docs/models/gpt-4.1)",
        ),
        (
            "gpt-4.1-mini",
            "GPT-4.1 mini",
            "Smaller, faster 4.1 model for cost-efficient tasks. Good balance between speed, cost, and quality. [Docs](https://platform.openai.com/docs/models/gpt-4.1-mini)",
        ),
        (
            "gpt-4.1-nano",
            "GPT-4.1 nano",
            "Ultra-lightweight GPT-4.1 model. Best when efficiency and low cost matter most. [Docs](https://platform.openai.com/docs/models/gpt-4.1-nano)",
        ),
        (
            "gpt-5",
            "GPT-5",
            "Latest flagship model with advanced reasoning and coding. Best for cutting-edge apps and agentic tasks. [Docs](https://platform.openai.com/docs/models/gpt-5)",
        ),
        (
            "gpt-5-mini",
            "GPT-5 mini",
            "Smaller, faster GPT-5 variant. Use when you want GPT-5 quality at lower cost. [Docs](https://platform.openai.com/docs/models/gpt-5-mini)",
        ),
        (
            "gpt-5-nano",
            "GPT-5 nano",
            "Ultra-small GPT-5 optimized for efficiency-critical tasks. Ideal for real-time or embedded use. [Docs](https://platform.openai.com/docs/models/gpt-5-nano)",
        ),
        (
            "o1",
            "o1",
            "Reasoning-first model that spends more time thinking before responding. Best for challenging reasoning problems. [Docs](https://platform.openai.com/docs/models/o1)",
        ),
        (
            "o1-mini",
            "o1 mini",
            "Lightweight variant of o1 optimized for STEM reasoning. Good for math-heavy or technical tasks. [Docs](https://platform.openai.com/docs/models/o1-mini)",
        ),
        (
            "o3",
            "o3",
            "Well-rounded reasoning model; excels at math, coding, and vision. Strong choice for multi-domain reasoning. [Docs](https://platform.openai.com/docs/models/o3)",
        ),
        (
            "o3-mini",
            "o3 mini",
            "Smaller o3-class model balancing speed and accuracy. Good for reasoning tasks when cost is a concern. [Docs](https://platform.openai.com/docs/models/o3-mini)",
        ),
        (
            "o4-mini",
            "o4 mini",
            "Efficient reasoning model optimized for fast, affordable performance. Best for reasoning tasks with tight budgets. [Docs](https://platform.openai.com/docs/models/o4-mini)",
        ),
    ]
)
chat_default = "gpt-4.1-nano"
embeddings_models = _to_choices(
    [
        (
            "text-embedding-3-small",
            "text-embedding-3-small",
            "Small and cheap embedding model sufficient for most use-cases. [Docs](https://platform.openai.com/docs/models/text-embedding-3-small)",
        ),
        (
            "text-embedding-3-large",
            "text-embedding-3-large",
            "Most capable but more expensive embedding model. [Docs](https://platform.openai.com/docs/models/text-embedding-3-large)",
        ),
        (
            "text-embedding-ada-002",
            "text-embedding-ada-002",
            "Older embedding model. [Docs](https://platform.openai.com/docs/models/text-embedding-ada-002)",
        ),
    ]
)
embeddings_default = "text-embedding-3-small"
models_w_embed_dims_api = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

_reserved_request_headers = [
    "Accept",
    "Accept-Encoding",
    "Connection",
    "Content-Length",
    "Content-Type",
    "Host",
]
_overwritable_request_headers = [
    "Authorization",
    "User-Agent",
    "OpenAI-Organization",
    "OpenAI-Project",
    "OpenAI-Beta",
]

credentials_port_type_id = "org.knime.credentials.base.CredentialPortObject"


class _EnumToStringParameter(knext.StringParameter):
    """Custom parameter implementation that maps enum names to their label in order to allow
    backwards compatibility.
    The knext.EnumParameter does not support the backwards compatible removal of values which is
    necessary because OpenAI is shutting down some of their models.
    Our initial strategy of switching to StringParameter does not work though because the names
    we used to store the settings are not the actual model names. This class works around this
    problem by mapping the enum constant names to the OpenAI model names before setting them on
    the node.
    """

    def __init__(
        self,
        label: str | None = None,
        description: str | None = None,
        default_value="",
        enum: List[str] | None = None,
        validator: Callable[[str], None] | None = None,
        since_version=None,
        is_advanced: bool = False,
        choices=None,
        options: knext.EnumParameterOptions = None,
    ):
        super().__init__(
            label,
            description,
            default_value,
            enum,
            validator,
            since_version,
            is_advanced,
            choices,
        )
        self._options = options

    def _set_value(self, obj, value, name, exclude_validations):
        try:
            value = self._options[value].value[0]
        except KeyError:
            pass
        return super()._set_value(obj, value, name)


# @knext.parameter_group(label="Model Settings") -- Imported
class OpenAIGeneralSettings(GeneralRemoteSettings):
    max_tokens = knext.IntParameter(
        label="Maximum response length (token)",
        description="""
        The maximum number of tokens to generate.

        This value, plus the token count of your prompt, cannot exceed the model's context length.
        """,
        default_value=1024,
        min_value=1,
    )

    # Altered from GeneralSettings because OpenAI has temperatures going up to 2
    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 2.0.

        Higher values will lead to less deterministic answers.

        Try 0.9 for more creative applications, and 0 for ones with a well-defined answer.
        It is generally recommended altering this, or Top-p, but not both.

        **Note**: this setting is ignored for reasoning models like GPT-5 or o-series models.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=2.0,
    )

    seed = knext.IntParameter(
        label="Seed",
        description="""
        Set the seed parameter to any integer of your choice to have (mostly) deterministic outputs.
        The default value of 0 means that no seed is specified.

        If the seed and other model parameters are the same for each request, 
        then responses will be mostly identical. There is a chance that responses 
        will differ, due to the inherent non-determinism of OpenAI models.

        Please note that this feature is in beta and only currently supported for gpt-4-1106-preview and 
        gpt-3.5-turbo-1106 [1].

        [1] [OpenAI Cookbook](https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter)
        """,
        default_value=0,
        is_advanced=True,
        since_version="5.3.0",
    )


def _create_specific_model_name(api_name: str) -> knext.StringParameter:
    def list_models(ctx: knext.DialogCreationContext) -> list[str]:
        model_list = []
        if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
            auth_spec: OpenAIAuthenticationPortObjectSpec
            model_list = auth_spec.get_model_list(ctx)
            model_list.sort()
        return model_list

    return knext.StringParameter(
        label="Specific model ID",
        description=f"""Select from a list of all available OpenAI models.
            The model chosen has to be compatible with OpenAI's {api_name} API.
            This configuration will **overwrite** the default model configurations when set.""",
        choices=list_models,
    )


class OpenAIModelOptions(knext.EnumParameterOptions):
    DEFAULT_MODELS = (
        "Default models",
        "Shows default models for this model type.",
    )
    ALL_MODELS = (
        "All models",
        """
        Shows all models available for the provided API key. This includes models that 
        may not be compatible with this specific endpoint, so it is the responsibility 
        of the user to select a model that is compatible with this node.
        """,
    )


def _get_model_selection_value_switch() -> knext.EnumParameter:
    return knext.EnumParameter(
        "Model selection",
        "Whether all available models are listed or only selected compatible ones.",
        OpenAIModelOptions.DEFAULT_MODELS.name,
        OpenAIModelOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.3.0",
    )


def _get_model_name(
    ctx,
    selection_mode: str,
    model: str,
    specific_model: str,
    default_model_list: List[knext.StringParameter.Choice],
    default_model: str,
) -> str:
    if selection_mode == OpenAIModelOptions.ALL_MODELS.name:
        return specific_model
    if model not in [choice.id for choice in default_model_list]:
        ctx.set_warning(
            f"Configured deprecated model, switching to fallback model: {default_model}"
        )
        return default_model
    return model


def _set_selection_parameter(parameters: dict) -> dict:
    input_settings = parameters["input_settings"]
    if not input_settings.get("selection"):
        input_settings["selection"] = (
            "ALL_MODELS"
            if input_settings["specific_model_name"] != "unselected"
            else "DEFAULT_MODELS"
        )
    return parameters


@knext.parameter_group(label="OpenAI Model Selection")
class LLMLoaderInputSettings:
    class OpenAIModelCompletionsOptions(knext.EnumParameterOptions):
        Ada = (
            "text-ada-001",
            "Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.",
        )
        Babbage = (
            "text-babbage-001",
            "Capable of straightforward tasks, very fast, and lower cost.",
        )
        Curie = (
            "text-curie-001",
            "Very capable, but faster and lower cost than Davinci.",
        )
        DaVinci2 = (
            "text-davinci-002",
            "Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models.",
        )
        DaVinci3 = (
            "text-davinci-003",
            "Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models.",
        )
        Gpt35TurboInstruct = (
            "gpt-3.5-turbo-instruct",
            "Recommended model for all completion tasks. As capable as text-davinci-003 but faster and lower in cost.",
        )

    selection = _get_model_selection_value_switch()

    default_model_name = _EnumToStringParameter(
        label="Model ID",
        description="Select an OpenAI completions model to be used.",
        choices=lambda c: completion_models,
        default_value=completion_default,
        options=OpenAIModelCompletionsOptions,
    ).rule(
        knext.OneOf(
            selection,
            [OpenAIModelOptions.DEFAULT_MODELS.name],
        ),
        knext.Effect.SHOW,
    )

    specific_model_name = _create_specific_model_name("Completions").rule(
        knext.OneOf(
            selection,
            [OpenAIModelOptions.ALL_MODELS.name],
        ),
        knext.Effect.SHOW,
    )


@knext.parameter_group(label="OpenAI chat model selection")
class ChatModelLoaderInputSettings:
    class OpenAIModelCompletionsOptions(knext.EnumParameterOptions):
        """Only used for backwards compatibility."""

        GPT35_TURBO_INSTRUCT = (
            "gpt-3.5-turbo-instruct",
            """Legacy instruct model. Use this for compatibility with prompts designed for instruct models.""",
        )
        Turbo = (
            "gpt-3.5-turbo",
            """Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003.""",
        )
        Turbo_16k = (
            "gpt-3.5-turbo-16k",
            "Same capabilities as the standard gpt-3.5-turbo model but with 4 times the context.",
        )
        GPT_4 = (
            "gpt-4",
            """More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat.""",
        )
        GPT_4_32k = (
            "gpt-4-32k",
            """Same capabilities as the base gpt-4 mode but with 4x the context length.""",
        )
        GPT_4_O = (
            "gpt-4o",
            """Most advanced model that’s cheaper and faster than GPT-4 Turbo.""",
        )
        GPT_4_O_MINI = (
            "gpt-4o-mini",
            """Cheaper and more capable than GPT-3.5 Turbo but just as fast.""",
        )
        TURBO_4 = (
            "gpt-4-turbo",
            """Faster and more cost-effective while maintaining comparable performance to GPT-4.
            It is recommended to use gpt-4o instead of this model.""",
        )
        O1 = (
            "o1",
            "Reasoning model designed to solve hard problems across a wide range of domains. Note that the temperature parameter is ignored for this model.",
        )
        O1_MINI = (
            "o1-mini",
            "Fast and affordable reasoning model for specialized tasks such as programming. Note that this model does not support system messages, and the temperature parameter is ignored.",
        )
        O3_MINI = (
            "o3-mini",
            "Fast and affordable reasoning model designed to excel at science, math, and coding tasks. Note that the temperature parameter is ignored for this model.",
        )

    selection = _get_model_selection_value_switch()

    model_name = _EnumToStringParameter(
        label="Model",
        description="Select a chat-optimized OpenAI model to be used.",
        # combined instruct and chat models after deprecating Instruct Model Selector nodes
        choices=completion_models + chat_models,
        default_value=chat_default,
        options=OpenAIModelCompletionsOptions,
    ).rule(
        knext.OneOf(selection, [OpenAIModelOptions.DEFAULT_MODELS.name]),
        knext.Effect.SHOW,
    )

    specific_model_name = _create_specific_model_name("Chat").rule(
        knext.OneOf(selection, [OpenAIModelOptions.ALL_MODELS.name]),
        knext.Effect.SHOW,
    )


@knext.parameter_group(label="OpenAI Embeddings Selection")
class EmbeddingsLoaderInputSettings:
    class OpenAIEmbeddingsOptions(knext.EnumParameterOptions):
        Ada1 = (
            "text-search-ada-doc-001",
            "Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.",
        )
        Ada2 = (
            "text-embedding-ada-002",
            "Capable of straightforward tasks, very fast, and lower cost.",
        )

    class DimensionOption(knext.EnumParameterOptions):
        AUTO = (
            "Auto",
            "Use the default embedding dimension for the model.",
        )
        CUSTOM = (
            "Custom",
            "Specify a custom value for the embedding dimension.",
        )

    selection = _get_model_selection_value_switch()

    model_name = _EnumToStringParameter(
        label="Model",
        description="Select an embedding OpenAI model to be used.",
        choices=embeddings_models,
        default_value=embeddings_default,
        options=OpenAIEmbeddingsOptions,
    ).rule(
        knext.OneOf(selection, [OpenAIModelOptions.DEFAULT_MODELS.name]),
        knext.Effect.SHOW,
    )

    specific_model_name = _create_specific_model_name("Embeddings").rule(
        knext.OneOf(selection, [OpenAIModelOptions.ALL_MODELS.name]),
        knext.Effect.SHOW,
    )

    dimension_settings = knext.EnumParameter(
        "Embedding dimension",
        "Whether to use the model's default embedding dimension or to specify a custom value.",
        default_value=DimensionOption.AUTO.name,
        enum=DimensionOption,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.3.0",
    ).rule(knext.OneOf(model_name, models_w_embed_dims_api), knext.Effect.SHOW)

    dimension = knext.IntParameter(
        "Embedding dimension size",
        """Embedding dimensions refer to the size of the vector space into which documents 
        are embedded.
        Lower dimensional embeddings capture less details of the documents but require less 
        memory space to store and are faster to process.

        Find more information about embeddings and maximum dimensions on 
        [OpenAIs Documentation](https://platform.openai.com/docs/models/embeddings).""",
        default_value=1536,
        since_version="5.3.0",
    ).rule(
        knext.And(
            knext.OneOf(dimension_settings, [DimensionOption.CUSTOM.name]),
            knext.OneOf(model_name, models_w_embed_dims_api),
        ),
        knext.Effect.SHOW,
    )


@knext.parameter_group(label="DALL-E 3 Settings")
class ImageLoaderInputSettings:
    size = knext.StringParameter(
        "Image size",
        """
        The size of the image that will be generated by DALL-E-3.
        Generating images with higher resolution will increase the costs.
        Please refer to 
        [OpenAI Docs](https://platform.openai.com/docs/pricing#image-generation)
        for more information about the pricing.
        """,
        choices=lambda c: ["1024x1024", "1792x1024", "1024x1792"],
        default_value="1024x1024",
    )

    quality = knext.StringParameter(
        "Quality",
        """
        The quality of the produced image where hd creates images with finer details 
        and greater consistency across the image. Generating higher quality images will 
        increase the costs. Please refer to 
        [OpenAI Docs](https://platform.openai.com/docs/pricing#image-generation)
        for more information about the pricing.
        """,
        choices=lambda c: ["standard", "hd"],
        default_value="standard",
    )

    style = knext.StringParameter(
        "Style",
        """
        The quality of the produced image where vivid causes the model to lean 
        towards generating hyper-real and dramatic images.
        Natural causes the model to produce more natural, less hyper-real looking images.
        """,
        choices=lambda c: ["vivid", "natural"],
        default_value="vivid",
    )


@knext.parameter_group(label="GPT Image 1 Settings", since_version="5.4.4")
class GPTImage1Settings:
    class ImageSizeOptions(knext.EnumParameterOptions):
        Square = ("Square", "1024x1024")
        Landscape = ("Landscape", "1536x1024")
        Portrait = ("Portrait", "1024x1536")

        @staticmethod
        def size_to_dimension(image_size) -> str:
            _size_map = {
                "Default": "1024x1024",
                "Landscape": "1536x1024",
                "Portrait": "1024x1536",
            }

            return _size_map.get(image_size, "1024x1024")

    size = knext.EnumParameter(
        "Image dimension",
        """
        The size of the image that will be generated by GPT Image 1.
        Generating images with higher resolution will increase the costs.
        Please refer to 
        [OpenAI Docs](https://platform.openai.com/docs/guides/images-vision#calculating-costs)
        for more information about the pricing.
        """,
        enum=ImageSizeOptions,
        default_value=ImageSizeOptions.Square.name,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    class QualityOptions(knext.EnumParameterOptions):
        High = (
            "High",
            "Generates images with the highest level of detail and consistency across the entire image.",
        )
        Medium = (
            "Medium",
            "Balances image quality and generation speed for general-purpose use.",
        )
        Low = (
            "Low",
            "Produces lower quality images with faster generation time and reduced computational load.",
        )

    quality = knext.EnumParameter(
        "Quality",
        """
        The quality of the produced image.
        
        Generating higher quality images will increase the costs. 
        Please refer to 
        [OpenAI Docs](https://platform.openai.com/docs/guides/images-vision#calculating-costs)
        for more information about the pricing.
        """,
        enum=QualityOptions,
        default_value=QualityOptions.Low.name,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    class BackgroundOptions(knext.EnumParameterOptions):
        Transparent = ("Transparent", "Generates images with a transparent background.")
        Opaque = ("Opaque", "Creates images with a solid, non-transparent background.")
        Auto = (
            "Auto",
            "Automatically selects the most appropriate background type based on the image content.",
        )

    background = knext.EnumParameter(
        label="Background",
        description="Allows to set transparency for the background of the generated image.",
        enum=BackgroundOptions,
        default_value=BackgroundOptions.Auto.name,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    class ModerationOptions(knext.EnumParameterOptions):
        Auto = (
            "Auto",
            "Basic filtering that tries to block certain "
            "types of content that might not be appropriate for all ages",
        )
        Low = ("Low", "Less restrictive filtering")

    moderation = knext.EnumParameter(
        label="Moderation",
        description="""
        All prompts and generated images are filtered in accordance with OpenAI's 
        [content policy](https://labs.openai.com/policies/content-policy). 
        This parameter sets the content moderation level for the generated image.
        """,
        enum=ModerationOptions,
        default_value=ModerationOptions.Auto.name,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    image_columns = knext.ColumnFilterParameter(
        "Images to edit",
        "Columns containing images to edit.",
        port_index=1,
        column_filter=util.image_column_filter,
    ).rule(
        knext.DialogContextCondition(util.image_table_present),
        knext.Effect.SHOW,
    )


@knext.parameter_group(label="Data")
class FineTuneFileSettings:
    id_column = knext.ColumnParameter(
        "Conversation ID column",
        "Column containing references to group rows into conversations.",
        port_index=1,
        column_filter=util.create_type_filter(knext.string()),
    )

    role_column = knext.ColumnParameter(
        "Role column",
        "Column containing the message role. Can be either 'system', 'assistant' or 'user'.",
        port_index=1,
        column_filter=util.create_type_filter(knext.string()),
    )

    content_column = knext.ColumnParameter(
        "Content column",
        "Column containing the message contents.",
        port_index=1,
        column_filter=util.create_type_filter(knext.string()),
    )


class AutomationOptions(knext.EnumParameterOptions):
    AUTOMATE = (
        "Auto",
        "OpenAI will determine a reasonable value for the configuration.",
    )

    MANUAL = ("Custom", "Allows to specify a custom value for the configuration.")


@knext.parameter_group(label="Fine-tuning")
class FineTunerInputSettings:
    automate_epochs = knext.EnumParameter(
        "Training epochs",
        """An epoch refers to one full cycle through the training dataset. If set to 'Auto',
        OpenAI will determine a reasonable value.""",
        default_value=AutomationOptions.AUTOMATE.name,
        enum=AutomationOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.3.0",
    )

    n_epochs = knext.IntParameter(
        "Number of training epochs",
        "An epoch refers to one full cycle through the training dataset.",
        default_value=1,
        min_value=1,
        since_version="5.3.0",
    ).rule(knext.OneOf(automate_epochs, ["MANUAL"]), knext.Effect.SHOW)

    automate_batch_size = knext.EnumParameter(
        "Batch size",
        """A larger batch size means that model parameters are updated less frequently, but with lower variance.
        If set to 'Auto', OpenAI will determine a reasonable value.""",
        default_value=AutomationOptions.AUTOMATE.name,
        enum=AutomationOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.3.0",
    )

    batch_size = knext.IntParameter(
        "Custom batch size",
        "A larger batch size means that model parameters are updated less frequently, but with lower variance.",
        default_value=1,
        min_value=1,
    ).rule(knext.OneOf(automate_batch_size, ["MANUAL"]), knext.Effect.SHOW)

    automate_learning_rate_multiplier = knext.EnumParameter(
        "Learning rate factor",
        """A smaller learning rate may be useful to avoid overfitting.
        If set to 'Auto', OpenAI will determine a reasonable value.""",
        default_value=AutomationOptions.AUTOMATE.name,
        enum=AutomationOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.3.0",
    )

    learning_rate_multiplier = knext.DoubleParameter(
        "Custom scaling factor",
        "A smaller learning rate may be useful to avoid overfitting.",
        default_value=1.0,
        min_value=0.01,
    ).rule(
        knext.OneOf(automate_learning_rate_multiplier, ["MANUAL"]), knext.Effect.SHOW
    )


@knext.parameter_group(label="Output")
class FineTunerResultSettings:
    suffix = knext.StringParameter(
        "Model name suffix",
        "A string of up to 18 characters that will be added to your fine-tuned model name.",
    )

    progress_interval = knext.IntParameter(
        "Polling interval (s)",
        "The time interval in seconds in which the node will check the progress of the fine-tuning job.",
        default_value=1,
        min_value=1,
    )

    # TODO: Add eventual validation file results


# == Port Objects ==


class OpenAIAuthenticationPortObjectSpec(AIPortObjectSpec):
    def __init__(
        self,
        credentials: str,
        base_url: str,
        credential_spec: Optional[knext.CredentialPortObjectSpec] = None,
        custom_headers: dict[str, str] = None,
    ) -> None:
        super().__init__()
        self._credentials = credentials
        self._base_url = base_url
        self._credential_spec = credential_spec
        self._custom_headers = {} if custom_headers is None else custom_headers

    @property
    def credentials(self) -> str:
        return self._credentials

    @property
    def custom_headers(self) -> dict[str, str]:
        return self._custom_headers

    @property
    def normalized_custom_headers(self) -> dict[str, str]:
        """Custom headers that are used by OpenAI internally are mapped to OpenAI's capitalization.
        OpenAI does not overwrite internal headers if the capitalization is different."""

        headers, _ = self._normalize_custom_headers()
        return headers

    @property
    def base_url(self) -> str:
        return self._base_url

    def validate_context(self, ctx: knext.ConfigurationContext):
        self._validate_credentials(ctx)
        if not self.base_url:
            raise knext.InvalidParametersError("Please provide a base URL.")

    def _validate_credentials(self, ctx: knext.ConfigurationContext):
        if self._credential_spec is not None:
            try:
                auth_schema = self._credential_spec.auth_schema
            except ValueError:
                raise knext.InvalidParametersError(
                    """The OpenAI API key is not available. Make sure that the node you are using to pass the
                    credential port object is still passing the valid API key to the downstream nodes."""
                )
            if auth_schema != "Bearer":
                raise knext.InvalidParametersError(
                    f"The credential uses unsupported schema '{auth_schema}'. Only 'Bearer' is supported."
                )
            return

        if self.credentials not in ctx.get_credential_names():
            raise knext.InvalidParametersError(
                f"""The selected credentials '{self.credentials}' holding the OpenAI API key do not exist. 
                Make sure that you have selected the correct credentials and that they are still available."""
            )
        api_token = ctx.get_credentials(self.credentials)
        if not api_token.password:
            raise knext.InvalidParametersError(
                f"""The OpenAI API key '{self.credentials}' does not exist. Make sure that the node you are using to pass the credentials 
                (e.g. the Credentials Configuration node) is still passing the valid API key as a flow variable to the downstream nodes."""
            )

    def validate_custom_headers(
        self,
        custom_headers,
    ):
        """Makes sure that common headers that are used by OpenAI are not overwritten.
        For some headers where it can be useful, overwriting them is allowed."""

        lowered_keys = [header.key.lower() for header in custom_headers]

        if "" in lowered_keys:
            raise knext.InvalidParametersError(
                "Every custom header needs to specify a key. Leaving the key empty is not allowed."
            )

        if len(lowered_keys) != len(set(lowered_keys)):
            raise knext.InvalidParametersError(
                "The custom headers must not contain any duplicate keys. The capitalization is ignored."
            )

        header_key_re = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")  # RFC 9110/7230
        # printable ASCII, no leading/trailing space, no tabs/CR/LF
        header_value_re = re.compile(r"^(?! )[ -~]*(?<! )$")

        for header in custom_headers:
            if header.key.lower() in [h.lower() for h in _reserved_request_headers]:
                raise knext.InvalidParametersError(
                    f"The header '{header.key}' is not allowed as it would overwrite a header used by OpenAI."
                )
            if not header_key_re.match(header.key):
                raise knext.InvalidParametersError(
                    f"The header key '{header.key}' contains invalid characters."
                )
            if not header_value_re.match(header.value):
                raise knext.InvalidParametersError(
                    f"The value for header '{header.key}' contains invalid characters. "
                    "Valid characters are printable ASCII characters without leading/trailing whitespaces."
                )

    def warn_if_headers_overwrite(self, ctx):
        _, overwritten = self._normalize_custom_headers()
        if overwritten:
            ctx.set_warning(
                f"The following custom headers overwrite headers used by OpenAI which can lead to failures in downstream nodes: {', '.join(overwritten)}."
            )

    def _normalize_custom_headers(
        self,
    ) -> tuple[dict[str, str], List[str]]:
        header_map = {h.lower(): h for h in _overwritable_request_headers}

        normalized = {}
        overwritten_headers = []
        for key, value in self.custom_headers.items():
            lower_header = key.lower()
            if lower_header in header_map:
                normalized[header_map[lower_header]] = value
                overwritten_headers.append(key)
            else:
                normalized[key] = value

        return normalized, overwritten_headers

    def get_api_key(self, ctx: knext.ConfigurationContext) -> str:
        if self._credential_spec is not None:
            return self._credential_spec.auth_parameters
        return ctx.get_credentials(self.credentials).password

    def get_openai_client(self, ctx: knext.ConfigurationContext):
        from openai import Client as OpenAIClient

        key = self.get_api_key(ctx)
        return OpenAIClient(
            api_key=key,
            base_url=self.base_url,
            default_headers=self.normalized_custom_headers,
        )

    def get_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        try:
            client = self.get_openai_client(ctx)
            model_list = [
                model.id for model in client.models.list().data if model.id is not None
            ]
        except Exception:
            # TODO Add warning to user once possible
            LOGGER.warning(
                "Could not fetch model list from OpenAI. Returning empty model list.",
                exc_info=True,
            )
            model_list = []

        model_list.sort()

        return model_list

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
            "base_url": self._base_url,
            "credential_spec": None
            if self._credential_spec is None
            else self._credential_spec.serialize(),
            "custom_headers": self.custom_headers,
        }

    @classmethod
    def deserialize(cls, data: dict):
        serialized_credential_spec = data.get("credential_spec")
        credential_spec = (
            None
            if serialized_credential_spec is None
            else knext.CredentialPortObjectSpec.deserialize(serialized_credential_spec)
        )
        return cls(
            data["credentials"],
            data.get("base_url", _default_openai_api_base),
            credential_spec,
            data.get("custom_headers", {}),
        )


class OpenAIAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: OpenAIAuthenticationPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> OpenAIAuthenticationPortObjectSpec:
        return super().spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: OpenAIAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


openai_authentication_port_type = knext.port_type(
    "OpenAI Authentication",
    OpenAIAuthenticationPortObject,
    OpenAIAuthenticationPortObjectSpec,
)


class OpenAIModelPortObjectSpec(AIPortObjectSpec):
    def __init__(self, credentials: OpenAIAuthenticationPortObjectSpec) -> None:
        self._credentials = credentials

    @property
    def credentials(self) -> str:
        return self._credentials.credentials

    @property
    def auth_spec(self) -> OpenAIAuthenticationPortObjectSpec:
        return self._credentials

    @property
    def base_url(self) -> str:
        return self._credentials.base_url

    def validate_context(self, ctx: knext.ConfigurationContext):
        self._credentials.validate_context(ctx)

    def serialize(self) -> dict:
        return self._credentials.serialize()

    @classmethod
    def deserialize_credentials_spec(
        cls, data: dict
    ) -> OpenAIAuthenticationPortObjectSpec:
        return OpenAIAuthenticationPortObjectSpec.deserialize(data)


class OpenAILLMPortObjectSpec(OpenAIModelPortObjectSpec, LLMPortObjectSpec):
    def __init__(
        self,
        credentials: OpenAIAuthenticationPortObjectSpec,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        n_requests: int,
    ) -> None:
        super().__init__(credentials)
        self._model = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._seed = seed
        self._n_requests = n_requests

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def top_p(self) -> float:
        return self._top_p

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def n_requests(self) -> int:
        return self._n_requests

    @property
    def supported_output_formats(self) -> list[OutputFormatOptions]:
        return [OutputFormatOptions.Text, OutputFormatOptions.JSON]

    def serialize(self) -> dict:
        return {
            **super().serialize(),
            "model": self._model,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "seed": self._seed,
            "n_requests": self._n_requests,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            cls.deserialize_credentials_spec(data),
            data["model"],
            data["temperature"],
            data["top_p"],
            data["max_tokens"],
            seed=data.get("seed", 0),
            n_requests=data.get("n_requests", 1),
        )


def _create_instruct_model(
    po_instance: "OpenAILLMPortObject", ctx: knext.ExecutionContext
):
    from langchain_openai import OpenAI

    key = po_instance.spec.auth_spec.get_api_key(ctx)
    return OpenAI(
        openai_api_key=key,
        base_url=po_instance.spec.base_url,
        model=po_instance.spec.model,
        temperature=po_instance.spec.temperature,
        top_p=po_instance.spec.top_p,
        max_tokens=po_instance.spec.max_tokens,
        seed=po_instance.spec.seed,
        default_headers=po_instance.spec.auth_spec.normalized_custom_headers,
    )


def _create_model(
    po_instance: "OpenAIChatModelPortObject",
    ctx: knext.ExecutionContext,
    output_format: OutputFormatOptions,
):
    from ._openai import _ChatOpenAI

    model_kwargs = {}
    if output_format == OutputFormatOptions.JSON:
        model_kwargs["response_format"] = {"type": "json_object"}

    def is_reasoning(model: str):
        is_o_series = bool(re.match(r"^o\d", model))
        is_gpt_5 = bool(re.match(r"^gpt-5", model))
        return is_o_series or is_gpt_5

    key = po_instance.spec.auth_spec.get_api_key(ctx)
    return _ChatOpenAI(
        openai_api_key=key,
        base_url=po_instance.spec.base_url,
        model=po_instance.spec.model,
        temperature=1.0
        if is_reasoning(po_instance.spec.model)
        else po_instance.spec.temperature,
        max_tokens=po_instance.spec.max_tokens,
        seed=po_instance.spec.seed,
        model_kwargs=model_kwargs,
        default_headers=po_instance.spec.auth_spec.normalized_custom_headers,
        ctx=ctx,
    )


class OpenAILLMPortObject(LLMPortObject):
    @property
    def spec(self) -> OpenAILLMPortObjectSpec:
        return super().spec

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        return _create_instruct_model(self, ctx)


openai_llm_port_type = knext.port_type(
    "OpenAI LLM", OpenAILLMPortObject, OpenAILLMPortObjectSpec
)


class OpenAIChatModelPortObjectSpec(OpenAILLMPortObjectSpec, ChatModelPortObjectSpec):
    """Spec of an OpenAI chat model."""

    @property
    def is_instruct_model(self) -> bool:
        return self._model in completion_models


class OpenAIChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> OpenAIChatModelPortObjectSpec:
        return super().spec

    def create_model(
        self,
        ctx: knext.ExecutionContext,
        output_format: OutputFormatOptions = OutputFormatOptions.Text,
    ):
        if self.spec.is_instruct_model:
            return _create_instruct_model(self, ctx)

        return _create_model(self, ctx, output_format)


openai_chat_port_type = knext.port_type(
    "OpenAI Chat Model", OpenAIChatModelPortObject, OpenAIChatModelPortObjectSpec
)


class OpenAIEmbeddingsPortObjectSpec(
    OpenAIModelPortObjectSpec, EmbeddingsPortObjectSpec
):
    def __init__(
        self,
        credentials: OpenAIAuthenticationPortObjectSpec,
        model_name: str,
        dimensions: int = None,
    ) -> None:
        super().__init__(credentials)
        self._model = model_name
        self._dimensions = dimensions

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def serialize(self) -> dict:
        return {
            **super().serialize(),
            "model": self._model,
            "dimensions": self._dimensions,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            cls.deserialize_credentials_spec(data),
            data["model"],
            data.get("dimensions"),
        )


class OpenAIEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(self, spec: OpenAIEmbeddingsPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> OpenAIEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from langchain_openai import OpenAIEmbeddings

        key = self.spec.auth_spec.get_api_key(ctx)
        return OpenAIEmbeddings(
            openai_api_key=key,
            base_url=self.spec.base_url,
            model=self.spec.model,
            dimensions=self.spec.dimensions,
            default_headers=self.spec.auth_spec.normalized_custom_headers,
        )


openai_embeddings_port_type = knext.port_type(
    "OpenAI Embedding Model",
    OpenAIEmbeddingsPortObject,
    OpenAIEmbeddingsPortObjectSpec,
)

credentials_port_type = kn.get_port_type_for_id(credentials_port_type_id)


# == Nodes ==

# region Authenticator


@knext.parameter_group("Custom header")
class CustomHeader:
    key = knext.StringParameter("Key", "Key of the header.")
    value = knext.StringParameter("Value", "Value of the header.")


def _has_auth_port(ctx: knext.DialogCreationContext) -> bool:
    specs = ctx.get_input_specs()
    return len(specs) == 1 and specs[0] is not None


@knext.node(
    "OpenAI Authenticator",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI"],
)
@knext.input_port(
    "Credentials",
    "Credentials for connecting to the OpenAI API.",
    port_type=credentials_port_type,
    optional=True,
)
@knext.output_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
class OpenAIAuthenticator:
    """
    Authenticates the OpenAI API key.

    This node provides the authentication for all OpenAI models.
    It allows you to select the credentials that contain a valid OpenAI API key in their *password* field (the *username* is ignored).
    Credentials can be set on the workflow level (right-click the workflow in the KNIME Explorer and click "Workflow Credentials")
    or created inside the workflow e.g. with the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory) and fed into this node via a flow variable.

    Alternatively, you can access an OpenAI compatible API via OAuth authentication by connecting a credential
    port object to the dynamic input port. The base URL needs to be set accordingly.

    When this node is executed, it validates the OpenAI API key by sending a request to the https://api.openai.com/v1/models endpoint.
    This request does not consume any tokens.

    If you do not have an OpenAI API key yet, you can generate one at
    [OpenAI](https://platform.openai.com/account/api-keys).
    """

    credentials_settings = CredentialsSettings(
        label="OpenAI API key",
        description="""
        The credentials containing the OpenAI API key in its *password* field (the *username* is ignored).
        Only shown if the optional credential port is not connected.
        """,
    ).rule(knext.DialogContextCondition(_has_auth_port), knext.Effect.HIDE)

    base_url = knext.StringParameter(
        "OpenAI base URL",
        """Sets the destination of the API requests to OpenAI or an OpenAI compatible API.""",
        default_value=_default_openai_api_base,
        since_version="5.2.0",
        is_advanced=True,
    )

    custom_headers = knext.ParameterArray(
        parameters=CustomHeader(),
        button_text="Add header",
        array_title="Header",
        label="Custom headers",
        description="Any custom headers to include in requests to the API.",
        since_version="5.7.0",
        is_advanced=True,
    )

    verify_settings = knext.BoolParameter(
        "Verify settings",
        "Whether to verify the settings by calling the list models endpoint.",
        default_value=True,
        since_version="5.2.1",
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        credential: Optional[knext.CredentialPortObjectSpec],
    ) -> OpenAIAuthenticationPortObjectSpec:
        if credential is None:
            if not ctx.get_credential_names():
                raise knext.InvalidParametersError("Credentials not provided.")

            if not self.credentials_settings.credentials_param:
                raise knext.InvalidParametersError("Credentials not selected.")

        spec = self.create_spec(credential)
        spec.validate_context(ctx)

        spec.validate_custom_headers(self.custom_headers)
        spec.warn_if_headers_overwrite(ctx)
        return spec

    def execute(
        self, ctx: knext.ExecutionContext, credential: Optional[knext.PortObject]
    ) -> OpenAIAuthenticationPortObject:
        spec = self.create_spec(None if credential is None else credential.spec)
        if self.verify_settings:
            self._verify_settings(ctx, spec)
        spec.warn_if_headers_overwrite(ctx)

        return OpenAIAuthenticationPortObject(spec)

    def _verify_settings(
        self, ctx: knext.ExecutionContext, spec: OpenAIAuthenticationPortObjectSpec
    ):
        from openai import (
            NotFoundError,
            AuthenticationError,
            APIConnectionError,
        )

        try:
            spec.get_openai_client(ctx).models.list()
        except AuthenticationError:
            raise knext.InvalidParametersError("Invalid API key provided.")
        except APIConnectionError:
            raise knext.InvalidParametersError(
                f"""API connection failed. Please make sure that your base URL '{self.base_url}' 
                is valid and uses a supported ('http://' or 'https://') protocol.
                It might also be caused by your network settings, proxy configuration, SSL certificates, or firewall rules."""
            )
        except NotFoundError:
            raise knext.InvalidParametersError(
                f"""The OpenAI base URL '{self.base_url}' is invalid or does not support OpenAI's '/v1/models' endpoint.
                This endpoint is used to list the models available for your API key.
                If your base URL does not support this endpoint, consider disabling the 'Verify settings' option."""
            )

    def create_spec(
        self,
        credential: Optional[knext.CredentialPortObjectSpec],
    ) -> OpenAIAuthenticationPortObjectSpec:
        return OpenAIAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param,
            base_url=self.base_url,
            credential_spec=credential,
            custom_headers={header.key: header.value for header in self.custom_headers},
        )


# TODO: Check proxy settings and add them to configuration
# TODO: Generate prompts as configuration dialog as seen on langchain llm.generate(["Tell me a joke", "Tell me a poem"]*15)
# region LLM Connector (Deprecated)
@knext.node(
    "OpenAI LLM Connector",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "Large Language Model", "OpenAI"],
    is_deprecated=True,
)
@knext.input_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI LLM",
    "Configured OpenAI LLM connection.",
    openai_llm_port_type,
)
class OpenAILLMConnector:
    """
    Select an LLM from OpenAI.

    This node establishes a connection with an OpenAI LLM.

    After successfully authenticating using the **OpenAI Authenticator node**, you can select an LLM from a predefined list
    or explore advanced options to get a list of all models available for your API key (including fine-tuned models).

    Note that only models compatible with OpenAI's Completions API will work with this node (unfortunately this information is not available programmatically).
    Find documentation about all models at [OpenAI](https://platform.openai.com/docs/models/models).

    **Note**: If you use the [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    input_settings = LLMLoaderInputSettings()
    model_settings = OpenAIGeneralSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        openai_auth_spec: OpenAIAuthenticationPortObjectSpec,
    ) -> OpenAILLMPortObjectSpec:
        if hasattr(openai_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use Azure Model Connectors")

        openai_auth_spec.validate_context(ctx)
        return self.create_spec(openai_auth_spec, ctx)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        openai_auth_port: OpenAIAuthenticationPortObject,
    ) -> OpenAILLMPortObject:
        return OpenAILLMPortObject(self.create_spec(openai_auth_port.spec, ctx))

    def create_spec(
        self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec, ctx
    ) -> OpenAILLMPortObjectSpec:
        model_name = _get_model_name(
            ctx,
            self.input_settings.selection,
            self.input_settings.default_model_name,
            self.input_settings.specific_model_name,
            completion_models,
            completion_default,
        )
        LOGGER.info(f"Selected model: {model_name}")

        seed = None if self.model_settings.seed == 0 else self.model_settings.seed

        return OpenAILLMPortObjectSpec(
            openai_auth_spec,
            model_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            seed=seed,
            n_requests=self.model_settings.n_requests,
        )

    def _modify_parameters(self, parameters):
        return _set_selection_parameter(parameters)


# region LLM Selector
@knext.node(
    "OpenAI LLM Selector",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI"],
)
@knext.input_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI Language Model",
    "Configured OpenAI Language Model.",
    openai_chat_port_type,
)
class OpenAIChatModelConnector:
    """
    Select an LLM from OpenAI or an OpenAI API-compatible provider.

    This node establishes a connection with an OpenAI Large Language Model (LLM). After successfully authenticating
    using the **OpenAI Authenticator** node, you can select a model from a predefined list.

    If OpenAI releases a new model that is not among the listed models, you can also select from a list
    of all available OpenAI models.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    input_settings = ChatModelLoaderInputSettings()
    model_settings = OpenAIGeneralSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        openai_auth_spec: OpenAIAuthenticationPortObjectSpec,
    ) -> OpenAIChatModelPortObjectSpec:
        if hasattr(openai_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use Azure Model Connectors")

        openai_auth_spec.validate_context(ctx)
        return self.create_spec(openai_auth_spec, ctx)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        openai_auth_port: OpenAIAuthenticationPortObject,
    ) -> OpenAIChatModelPortObject:
        return OpenAIChatModelPortObject(self.create_spec(openai_auth_port.spec, ctx))

    def create_spec(
        self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec, ctx
    ) -> OpenAIChatModelPortObjectSpec:
        model_name = _get_model_name(
            ctx,
            self.input_settings.selection,
            self.input_settings.model_name,
            self.input_settings.specific_model_name,
            # allow selecting instruct models here after deprecating Instruct Model Selector nodes
            completion_models + chat_models,
            chat_default,
        )

        LOGGER.info(f"Selected model: {model_name}")

        seed = None if self.model_settings.seed == 0 else self.model_settings.seed

        return OpenAIChatModelPortObjectSpec(
            openai_auth_spec,
            model_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            seed=seed,
            n_requests=self.model_settings.n_requests,
        )

    def _modify_parameters(self, parameters):
        return _set_selection_parameter(parameters)


# region Embedding Selector
@knext.node(
    "OpenAI Embedding Model Selector",
    knext.NodeType.SOURCE,
    openai_icon,
    category=openai_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "OpenAI",
        "RAG",
        "Retrieval Assisted Generation",
    ],
)
@knext.input_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
@knext.output_port(
    "OpenAI Embedding Model",
    "Configured OpenAI Embedding Model connection.",
    openai_embeddings_port_type,
)
class OpenAIEmbeddingsConnector:
    """
    Select an OpenAI embedding model.

    This node establishes a connection with an OpenAI Embedding Model. After successfully authenticating
    using the **OpenAI Authenticator** node, you can select an embedding model. Follow
    [OpenAI](https://platform.openai.com/docs/models/models) to find the latest embedding models.

    If OpenAI releases a new embedding model that is not contained in the predefined list, you can select it from
    the list in the advanced settings which contains all OpenAI models available for your OpenAI API key.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    input_settings = EmbeddingsLoaderInputSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        openai_auth_spec: OpenAIAuthenticationPortObjectSpec,
    ) -> OpenAIEmbeddingsPortObjectSpec:
        if hasattr(openai_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use Azure Model Connectors")

        openai_auth_spec.validate_context(ctx)
        return self.create_spec(openai_auth_spec, ctx)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        openai_auth_port: OpenAIAuthenticationPortObject,
    ) -> OpenAIEmbeddingsPortObject:
        return OpenAIEmbeddingsPortObject(self.create_spec(openai_auth_port.spec, ctx))

    def create_spec(
        self, openai_auth_spec: OpenAIAuthenticationPortObjectSpec, ctx
    ) -> OpenAIEmbeddingsPortObjectSpec:
        model_name = _get_model_name(
            ctx,
            self.input_settings.selection,
            self.input_settings.model_name,
            self.input_settings.specific_model_name,
            embeddings_models,
            embeddings_default,
        )
        LOGGER.info(f"Selected model: {model_name}")

        custom_dimension = (
            self.input_settings.dimension_settings
            == EmbeddingsLoaderInputSettings.DimensionOption.CUSTOM.name
            and self.input_settings.model_name in models_w_embed_dims_api
        )

        dimensions = self.input_settings.dimension if custom_dimension else None

        return OpenAIEmbeddingsPortObjectSpec(openai_auth_spec, model_name, dimensions)

    def _modify_parameters(self, parameters):
        return _set_selection_parameter(parameters)


class ImageModels(knext.EnumParameterOptions):
    GPT_IMAGE_1 = (
        "GPT Image 1",
        "State-of-the-art quality, excels at detailed and complex prompts (recommended).",
    )
    DALL_E_3 = (
        "DALL-E 3",
        "Previous-generation model: faster outputs with solid everyday quality.",
    )


# region Image Generator
@knext.node(
    "OpenAI Image Generator",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=openai_icon,
    category=openai_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI", "Image generation"],
)
@knext.input_port(
    "OpenAI Authentication",
    "Validated authentication for OpenAI.",
    openai_authentication_port_type,
)
@knext.input_table("Image table", "Table containing the images to edit.", optional=True)
@knext.output_image("Generated image", "The image generated by the model.")
@knext.output_view("View", "View of the generated image.")
class OpenAIDALLEView:
    """
    Generate or edit images with OpenAI's DALL-E 3 or GPT Image 1.

    This node allows you to generate or edit images using OpenAI's DALL-E 3 or GPT Image 1.
    GPT Image 1 also supports image editing, while DALL-E 3 does not.

    To generate an image, provide a prompt and select the model you want to use.
    To edit images, provide an input table that contains image columns.

    You can include multiple image columns in the table and choose which ones to use for editing.
    For example, one column could contain images of objects and another could contain backgrounds.
    After selecting both columns and providing a prompt, the model will generate a new image that combines the objects and backgrounds.
    Currently, the node only supports PNG images.

    The use of GPT Image 1 may require your organization to be verified by OpenAI.
    To do so visit the [OpenAI platform](https://platform.openai.com/settings/organization/general).

    Generating images is **significantly more expensive** than text generation. Please refer to
    [OpenAI Docs](https://platform.openai.com/docs/guides/images-vision#calculating-costs) for more details on
    how the costs are calculated.

    ---

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    prompt_size_limits = {
        ImageModels.DALL_E_3.name: 4000,
        ImageModels.GPT_IMAGE_1.name: 32000,
    }

    model = knext.EnumParameter(
        label="Model",
        enum=ImageModels,
        default_value=lambda v: ImageModels.GPT_IMAGE_1.name
        if v >= knext.Version(5, 4, 4)
        else ImageModels.DALL_E_3.name,
        since_version="5.4.4",
        description="The model to use for image generation.",
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    prompt = knext.MultilineStringParameter(
        "Prompt",
        """
        The prompt the model uses to generate an image.
        A more descriptive prompt will usually generate a better image. 
        The maximum length is 32000 characters for GPT Image 1 and 4000 characters for DALL-E 3.
        """,
    )
    settings = ImageLoaderInputSettings().rule(
        knext.OneOf(model, [ImageModels.DALL_E_3.name]), knext.Effect.SHOW
    )

    gpt_image_1_settings = GPTImage1Settings().rule(
        knext.OneOf(model, [ImageModels.GPT_IMAGE_1.name]), knext.Effect.SHOW
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: OpenAIAuthenticationPortObjectSpec,
        image_table_spec: Optional[knext.Schema],
    ):
        authentication.validate_context(ctx)

        if self.model not in ImageModels.__members__:
            raise knext.InvalidParametersError(
                f"Invalid model '{self.model}' selected. Please select a valid model."
            )

        if not self.prompt:
            raise knext.InvalidParametersError(
                "The image generation prompt cannot be empty."
            )

        prompt_size_limit = self.prompt_size_limits[self.model]

        if len(self.prompt) > prompt_size_limit:
            knext.InvalidParametersError(
                f"Prompt cannot exceed a length of {prompt_size_limit} characters. Prompt length is {len(self.prompt)}."
            )

        has_image_table = image_table_spec is not None
        if has_image_table and self.model == ImageModels.GPT_IMAGE_1.name:
            if self.gpt_image_1_settings.image_columns is None:
                raise knext.InvalidParametersError(
                    "Please select at least one image column for editing."
                )

        return knext.ImagePortObjectSpec(knext.ImageFormat.PNG)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        authentication: OpenAIAuthenticationPortObject,
        image_table: Optional[knext.Table],
    ):
        has_image_table = image_table is not None
        client = authentication.spec.get_openai_client(ctx)

        if self.model == ImageModels.DALL_E_3.name:
            img_bytes = self._create_with_dalle_3(client)
        elif self.model == ImageModels.GPT_IMAGE_1.name:
            if has_image_table:
                images = util.prepare_images(
                    image_table, self.gpt_image_1_settings.image_columns
                )
                img_bytes = self._edit_with_gpt_image_1(client, images)
            else:
                img_bytes = self._create_with_gpt_image_1(client)
        else:
            raise knext.InvalidParametersError(
                f"Invalid model '{self.model}' selected. Please select a valid model."
            )

        return img_bytes, knext.view_png(img_bytes)

    def _create_with_dalle_3(self, client) -> bytes:
        import requests

        response = client.images.generate(
            prompt=self.prompt,
            model="dall-e-3",
            n=1,
            quality=self.settings.quality,
            response_format="url",
            size=self.settings.size,
            style=self.settings.style,
        )

        url = response.data[0].url

        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError("Could not retrieve image from OpenAI.")
        return response.content

    def _create_with_gpt_image_1(self, client) -> bytes:
        size = GPTImage1Settings.ImageSizeOptions.size_to_dimension(
            self.gpt_image_1_settings.size
        )

        try:
            response = client.images.generate(
                prompt=self.prompt,
                model="gpt-image-1",
                n=1,
                quality=self.gpt_image_1_settings.quality.lower(),
                size=size,
                moderation=self.gpt_image_1_settings.moderation.lower(),
                background=self.gpt_image_1_settings.background.lower(),
            )
        except Exception as e:
            error_message = str(e)
            if (
                "Error code: 400" in error_message
                and "moderation_blocked" in error_message
            ):
                raise RuntimeError(
                    "Your request was rejected by OpenAI due to safety concerns."
                    "Please ensure your prompt complies with OpenAI's content policy or try moderation setting 'Low'."
                )
            else:
                raise RuntimeError(
                    f"Failed to generate an image. Error message: {error_message}"
                )

        b_64 = response.data[0].b64_json

        return base64.b64decode(b_64)

    def _edit_with_gpt_image_1(self, client, images) -> bytes:
        size = GPTImage1Settings.ImageSizeOptions.size_to_dimension(
            self.gpt_image_1_settings.size
        )

        try:
            response = client.images.edit(
                prompt=self.prompt,
                model="gpt-image-1",
                n=1,
                image=images,
                size=size,
                quality=self.gpt_image_1_settings.quality.lower(),
            )
        except Exception as e:
            error_message = str(e)
            raise RuntimeError(
                f"Failed to edit image(s). Error message: {error_message}"
            )
        b_64 = response.data[0].b64_json

        return base64.b64decode(b_64)


# region Model Deleter
@knext.node(
    "OpenAI Fine-Tuned Model Deleter",
    knext.NodeType.SINK,
    openai_icon,
    openai_category,
)
@knext.input_port(
    "Fine-Tuned OpenAI Model", "A fine-tuned model from OpenAI", llm_port_type
)
class OpenAIFineTuneDeleter:
    """
    Deletes a fine-tuned model from an OpenAI account.

    This node allows you to delete a fine-tuned model from your OpenAI account.

    The fine-tuned model must be selected via the **OpenAI LLM Selector** node.

    If the provided API key possesses the necessary permissions, the model will then be irreversibly removed from the
    authenticated OpenAI account.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    def configure(
        self, ctx: knext.ConfigurationContext, llm_spec: OpenAILLMPortObjectSpec
    ):
        client = llm_spec.auth_spec.get_openai_client(ctx)

        response = client.models.retrieve(llm_spec.model)

        owned_by_openai = ["openai", "openai-internal", "system"]

        if response.owned_by in owned_by_openai:
            raise knext.InvalidParametersError(
                f"This model is owned by OpenAI ('{response.owned_by}') and cannot be deleted."
            )

    def execute(self, ctx: knext.ExecutionContext, model: OpenAILLMPortObject):
        from openai import NotFoundError

        client = model.spec.auth_spec.get_openai_client(ctx)

        try:
            response = client.models.delete(model.spec.model)
        except NotFoundError:
            raise RuntimeError(
                f"Could not delete model. Please make sure that you have the right permissions to delete '{model.spec.model}'"
            )

        LOGGER.info(f"{response.id} was successfully deleted.")


# region Fine-tuner
@knext.node(
    "OpenAI Chat Model Fine-Tuner",
    node_type=knext.NodeType.LEARNER,
    icon_path=openai_icon,
    category=openai_category,
)
@knext.input_port(
    "OpenAI Chat Model",
    "Configured OpenAI Chat Model which supports fine-tuning.",
    openai_chat_port_type,
)
@knext.input_table(
    "Fine-tuning Data",
    """
    The data should be presented across 3 columns: 
    
    One column specifying a conversation ID, one representing the role of a message (system, assistant and user), 
    and the third for the content of the message.

    The table has to include at least 10 conversations, each of which must contain at least one system message.
    """,
)
@knext.output_port(
    "OpenAI Chat Model",
    "Configured fine-tuned OpenAI Chat Model connection.",
    openai_chat_port_type,
)
@knext.output_table(
    "Fine-tuning Metrics",
    """
    Metrics to evaluate the fine-tuning performance. The values of the metrics are: 
    'train loss', 'train accuracy', 'valid loss', and 'valid mean token accuracy' for each step of training.
    """,
)
class OpenAIFineTuner:
    """
    Fine-tunes an OpenAI Chat Model using a table of example conversations.

    This node fine-tunes an OpenAI Chat Model using structured conversation data.
    It is useful when you want to adapt a model to a specific tone, domain, or workflow — for example, tailoring it for financial advice, customer support, or internal knowledge assistants.

    Each row in the input table represents a message in a conversation. The table must contain at least 10 distinct conversations, and each must include at least one `system` message to define the assistant’s behavior.
    The fine-tuning process learns from examples: it does not memorize answers, but generalizes from the patterns in the assistant replies. You define *how* the assistant should respond to user inputs by providing example dialogues with the desired outputs.

    Fine-tuned models are stored on OpenAI's servers and can afterwards be selected in the `OpenAI LLM Selector`.
    To delete a fine-tuned model, use the `OpenAI Fine-Tuned Model Deleter` node.

    For pricing, see the [OpenAI documentation](https://platform.openai.com/docs/pricing#fine-tuning).

    To fine-tune a model for the finance domain, you might provide example conversations that emphasize clear, compliant financial guidance.
    Here is an example fine-tuning table:

    | ID | Role     | Content                                                                 |
    |----|----------|-------------------------------------------------------------------------|
    | 1  | system   | You are a financial assistant who gives concise, compliant guidance.    |
    | 1  | user     | Should I invest in tech stocks right now?                               |
    | 1  | assistant| I can't give specific advice, but tech stocks are volatile. Consider your risk profile. |
    | 2  | user     | What's diversification?                                                  |
    | 2  | assistant| Diversification spreads assets across sectors to reduce risk.           |


    **Credential Handling**:
    To pass your API key securely, use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory).
    If "Save password in configuration (weakly encrypted)" is not enabled, the credentials will not persist after closing the workflow.
    """

    file_settings = FineTuneFileSettings()
    ft_settings = FineTunerInputSettings()
    ft_result_settings = FineTunerResultSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        model_spec: OpenAIChatModelPortObjectSpec,
        table_spec: knext.Schema,
    ) -> OpenAIChatModelPortObjectSpec:
        util.pick_default_columns(table_spec, knext.string(), 3)  # raises an exception

        if not self._unique_columns_selected():
            raise knext.InvalidParametersError("Selected columns need to be unique")

        if len(self.ft_result_settings.suffix) > 18:
            raise knext.InvalidParametersError(
                f"Suffix can at maximum be 18 characters long. Used characters: {len(self.ft_result_settings.suffix)}."
            )

        metric_spec = knext.Schema.from_columns(
            [
                knext.Column(knext.int32(), "step"),
                knext.Column(knext.double(), "train_loss"),
                knext.Column(knext.double(), "train_accuracy"),
                knext.Column(knext.double(), "valid_loss"),
                knext.Column(knext.double(), "valid_mean_token_accuracy"),
            ]
        )
        return model_spec, metric_spec

    def execute(
        self,
        ctx: knext.ExecutionContext,
        model: OpenAIChatModelPortObject,
        table: knext.Table,
    ):
        from openai.types import FileObject
        from openai.types.fine_tuning.fine_tuning_job import FineTuningJob
        from openai import BadRequestError

        client = model.spec.auth_spec.get_openai_client(ctx)

        hyperparameters = self._build_hyper_params()

        df = table.to_pandas()
        conversation_list = self._validate_fine_tune_table(df)

        try:
            training_file: FileObject = self._prepare_training_file(
                client, conversation_list
            )

            # This is needed if the request gets canceled before response is 'initialized' through _run_fine_tuning()
            response = None
            response: FineTuningJob = self._run_fine_tuning(
                client=client,
                model_name=model.spec.model,
                training_file=training_file,
                hyper_params=hyperparameters,
                ctx=ctx,
            )

            result_df = self._response_to_df(client, response)

            fine_tuned_model = OpenAIChatModelPortObject(
                OpenAIChatModelPortObjectSpec(
                    credentials=model.spec._credentials,
                    model_name=response.fine_tuned_model,
                    temperature=model.spec.temperature,
                    top_p=model.spec.top_p,
                    max_tokens=model.spec.max_tokens,
                    seed=model.spec.seed,
                    n_requests=model.spec.n_requests,
                )
            )

        except BadRequestError as e:
            if e.message.endswith("'code': 'model_not_available'}}"):
                raise knext.InvalidParametersError(
                    "Selected model is not available or does not support fine-tuning."
                )
            else:
                raise knext.InvalidParametersError(e.message)

        finally:
            client.files.delete(training_file.id)  # deletes the tmp file at openai

            if response:
                for f in response.result_files:
                    client.files.delete(f)

        return fine_tuned_model, result_df

    def _unique_columns_selected(self) -> bool:
        selected_columns = set(
            [
                self.file_settings.id_column,
                self.file_settings.role_column,
                self.file_settings.content_column,
            ]
        )

        return len(selected_columns) == 3

    def _build_hyper_params(self):
        from openai.types.fine_tuning.fine_tuning_job import Hyperparameters

        batch_size: int = (
            self.ft_settings.batch_size
            if self.ft_settings.automate_batch_size == AutomationOptions.MANUAL.name
            else "auto"
        )

        learning_rate_multiplier: float = (
            self.ft_settings.learning_rate_multiplier
            if self.ft_settings.automate_learning_rate_multiplier
            == AutomationOptions.MANUAL.name
            else "auto"
        )

        n_epochs: int = (
            self.ft_settings.n_epochs
            if self.ft_settings.automate_epochs == AutomationOptions.MANUAL.name
            else "auto"
        )

        return Hyperparameters(
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
            n_epochs=n_epochs,
        )

    def _prepare_training_file(self, client, conversation_list: List):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".jsonl"
        ) as temp_jsonl_file:
            temp_file_path = temp_jsonl_file.name

            for conversation in conversation_list:
                conversation_bytes = json.dumps(conversation).encode("utf-8")
                temp_jsonl_file.write(conversation_bytes + b"\n")

        file = client.files.create(file=open(temp_file_path, "rb"), purpose="fine-tune")

        return file

    def _validate_fine_tune_table(self, df) -> List[dict[str, List[dict[str, any]]]]:
        grouped_ids = df.groupby(self.file_settings.id_column)

        if len(grouped_ids) < 10:
            raise knext.InvalidParametersError(
                "The fine-tuning data table needs to contain at least 10 conversations."
            )

        if df[self.file_settings.content_column].isnull().values.any():
            raise knext.InvalidParametersError(
                "Missing values in the content column are not allowed."
            )

        has_assistant_roles = grouped_ids[self.file_settings.role_column].apply(
            lambda x: "assistant" in x.values
        )

        if not has_assistant_roles.all():
            missing_assistant_groups = [
                index for index, value in has_assistant_roles.items() if not value
            ]

            raise knext.InvalidParametersError(
                f"The following conversations are missing 'assistant' messages: {', '.join(map(str, missing_assistant_groups))}"
            )

        conversation_list = [
            {
                "messages": [
                    {"role": role.lower(), "content": content}
                    for role, content in zip(
                        group[self.file_settings.role_column],
                        group[self.file_settings.content_column],
                    )
                ]
            }
            for _, group in grouped_ids
        ]

        return conversation_list

    def _run_fine_tuning(
        self,
        client,
        model_name: str,
        training_file,
        hyper_params,
        ctx: knext.ExecutionContext,
    ):
        job = client.fine_tuning.jobs.create(
            model=model_name,
            training_file=training_file.id,
            hyperparameters=hyper_params,
            suffix=self.ft_result_settings.suffix,
        )

        return self._await_fine_tuning_response(client, job, ctx)

    def _await_fine_tuning_response(self, client, job, ctx: knext.ExecutionContext):
        while True:
            if ctx.is_canceled():
                client.fine_tuning.jobs.cancel(job.id)
                raise RuntimeError("Fine-tuning job has been canceled.")

            response = client.fine_tuning.jobs.retrieve(job.id)
            ctx.set_progress(0.5, f"Current fine-tuning status: {response.status}")

            if response.status == "failed":
                raise ValueError(response.error.message)

            if response.fine_tuned_model and response.status in [
                "succeeded",
                "failed",
                "cancelled",
            ]:
                return response

            time.sleep(self.ft_result_settings.progress_interval)

    def _response_to_df(self, client, response):
        import pandas as pd
        from openai._legacy_response import HttpxBinaryResponseContent

        response_list: List[HttpxBinaryResponseContent] = [
            client.files.content(file_id) for file_id in response.result_files
        ]

        decoded_df_list = [
            pd.read_csv(io.BytesIO(base64.b64decode(response.content)), header=0)
            for response in response_list
        ]

        df = pd.concat(decoded_df_list, ignore_index=True)
        df["step"] = df["step"].astype("int32")
        return knext.Table.from_pandas(df)
