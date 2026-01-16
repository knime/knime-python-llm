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


import knime.extension as knext
import io
from typing import List, Tuple, Optional
from PIL import Image

import util
from ._utils import google_category, gemini_icon
from ._port_types import (
    generic_gemini_connection_port_type,
    GoogleAiStudioAuthenticationPortObjectSpec,
    GoogleAiStudioAuthenticationPortObject,
)

_POLICY_GUIDELINES_URL = "https://policies.google.com/terms/generative-ai/use-policy"

_BLOCKING_REASON_DESCRIPTIONS = {
    # FinishReason values
    "IMAGE_SAFETY": "The generated image was blocked due to safety policy violations.",
    "IMAGE_PROHIBITED_CONTENT": "The generated image was blocked because it contains prohibited content.",
    "IMAGE_OTHER": "The image generation was blocked for unspecified reasons.",
    "NO_IMAGE": "The model was unable to generate an image.",
    "SAFETY": "Content was blocked due to safety policy violations.",
    "RECITATION": "The generated content too closely resembles copyrighted material.",
    "PROHIBITED_CONTENT": "Content was blocked because it contains prohibited material.",
    "BLOCKLIST": "Content was blocked because it contains forbidden terms.",
    "SPII": "Content was blocked because it potentially contains Sensitive Personally Identifiable Information (SPII).",
    "MALFORMED_FUNCTION_CALL": "The request was malformed.",
    "OTHER": "Content was blocked for unspecified reasons.",
    # BlockedReason values (for prompt blocking).
    "MODEL_ARMOR": "The prompt was blocked by content filtering.",
    "JAILBREAK": "The prompt was blocked as a potential policy violation attempt.",
}


def _get_blocking_reason_description(reason) -> str:
    """
    Convert a blocking reason enum to a user-friendly description.

    Args:
        reason: A FinishReason or BlockedReason enum value.

    Returns:
        A user-friendly description of why the content was blocked,
        including a link to Google's policy guidelines.
    """
    reason_str = str(reason)
    # Extract just the value (e.g., "IMAGE_SAFETY" from "FinishReason.IMAGE_SAFETY").
    if "." in reason_str:
        reason_str = reason_str.split(".")[-1]

    description = _BLOCKING_REASON_DESCRIPTIONS.get(
        reason_str,
        f"Reason: {reason_str}."
    )

    description += f" For more information, see: {_POLICY_GUIDELINES_URL}."

    return description


GEMINI_IMAGE_MODELS = [
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
]

IMAGEN_MODELS = [
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001",
]

@knext.parameter_group(label="Imagen Settings")
class ImagenSettings:
    class AspectRatioOptions(knext.EnumParameterOptions):
        Square = ("Square", "Square format")
        Portrait_3_4 = ("Portrait 3:4", "Portrait format")
        Landscape_4_3 = ("Landscape 4:3", "Landscape format")
        Vertical_9_16 = ("Vertical 9:16", "Vertical mobile format")
        Horizontal_16_9 = ("Horizontal 16:9", "Horizontal widescreen format")

        @staticmethod
        def to_api_value(ratio) -> str:
            _ratio_map = {
                "Square": "1:1",
                "Portrait_3_4": "3:4",
                "Landscape_4_3": "4:3",
                "Vertical_9_16": "9:16",
                "Horizontal_16_9": "16:9",
            }
            return _ratio_map.get(ratio, "1:1")

    aspect_ratio = knext.EnumParameter(
        "Aspect Ratio",
        """
        Changes the aspect ratio of the generated image.
        """,
        enum=AspectRatioOptions,
        default_value=AspectRatioOptions.Square.name,
        style=knext.EnumParameter.Style.DROPDOWN,
    )

    class PersonGenerationOptions(knext.EnumParameterOptions):
        Block_People = ("Block People", "Block generation of images containing people.")
        Adults_Only = ("Adults Only", "Generate images of adults only, but not children (default).")

        @staticmethod
        def to_api_value(person_gen) -> str:
            _person_map = {
                "Block_People": "DONT_ALLOW",
                "Adults_Only": "ALLOW_ADULT",
            }
            return _person_map.get(person_gen, "ALLOW_ADULT")

    person_generation = knext.EnumParameter(
        "Person Generation",
        """
        Allow the model to generate images of people.
        """,
        enum=PersonGenerationOptions,
        default_value=PersonGenerationOptions.Adults_Only.name,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

class ImageModels(knext.EnumParameterOptions):
    GEMINI = (
        "Gemini",
        "Multimodal model that supports both text-to-image generation and image editing.",
    )
    IMAGEN = (
        "Imagen",
        "Specialized text-to-image generation model that offers faster generation but no image editing.",
    )


def _get_gemini_model_choices(ctx: knext.DialogCreationContext):
    """Get Gemini models for the choices list."""
    input_specs = ctx.get_input_specs()
    auth_spec: GoogleAiStudioAuthenticationPortObjectSpec = input_specs[0]
    
    if auth_spec is None:
        return GEMINI_IMAGE_MODELS
    
    try:
        all_models = auth_spec.get_image_model_list(ctx)
        return [model for model in all_models if "gemini" in model.lower()]
    except (KeyError, Exception):
        return GEMINI_IMAGE_MODELS

def _get_imagen_model_choices(ctx: knext.DialogCreationContext):
    """Get Imagen models for the choices list."""
    input_specs = ctx.get_input_specs()
    auth_spec: GoogleAiStudioAuthenticationPortObjectSpec = input_specs[0]
    
    if auth_spec is None:
        return IMAGEN_MODELS
    
    try:
        all_models = auth_spec.get_image_model_list(ctx)
        return [model for model in all_models if "imagen" in model.lower()]
    except (KeyError, Exception):
        return IMAGEN_MODELS

@knext.node(
    "Gemini Image Generator",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=gemini_icon,
    category=google_category,
    keywords=["Google", "Gemini", "GenAI", "Image generation"],
)
@knext.input_port(
    "Gemini Authentication",
    "Validated authentication for Google AI Studio.",
    generic_gemini_connection_port_type,
)
@knext.input_table("Image table", "Table containing the images to edit.", optional=True)
@knext.output_image("Generated image", "The image generated by the model.")
@knext.output_view("View", "View of the generated image.")
class GeminiImageGenerator:
    """
    Generate or edit images with Google's image generation models.

    This node allows you to generate or edit images using Gemini or Imagen models.
    Gemini supports image editing, while Imagen models do not.

    To generate an image, provide a descriptive prompt. The more detailed your prompt, the better 
    the resulting image will be.

    To edit images (Gemini only), connect a table containing image columns to the optional input port. 
    You can select which image columns to use for editing and provide instructions on how 
    to modify or combine the images.

    **Note**: Image generation may be significantly more expensive than text generation. 
    Please refer to
    [Gemini Docs](https://ai.google.dev/gemini-api/docs/pricing) for more details on
    how the costs are calculated.
    """

    # Prompt size limits for different model types from API
    # https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash#image
    # https://ai.google.dev/gemini-api/docs/imagen#imagen-4
    prompt_size_limits = {
        "gemini": 32768,
        "imagen": 480,
    }

    model = knext.EnumParameter(
        label="Model Type",
        enum=ImageModels,
        default_value=ImageModels.GEMINI.name,
        description="The type of model to use for image generation.",
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    gemini_model_name = knext.StringParameter(
        "Model",
        """
        Select the specific Gemini model to use.
        """,
        choices=_get_gemini_model_choices,
    ).rule(
        knext.OneOf(model, [ImageModels.GEMINI.name]),
        knext.Effect.SHOW,
    )

    imagen_model_name = knext.StringParameter(
        "Model",
        """
        Select the specific Imagen model to use.
        """,
        choices=_get_imagen_model_choices,
    ).rule(
        knext.OneOf(model, [ImageModels.IMAGEN.name]),
        knext.Effect.SHOW,
    )

    prompt = knext.MultilineStringParameter(
        "Prompt",
        """
        The prompt describing the image you want to generate or how to edit the input images.
        
        For text-to-image generation, describe the image you want to create in detail.
        For image editing, describe how you want to modify, combine, or enhance the input images.
        
        The token limit depends on the selected model:
        - Gemini models: 32,768 tokens
        - Imagen models: 480 tokens
        """,
    )

    image_columns = knext.ColumnFilterParameter(
        "Images to edit",
        """
        Select the image columns to use for editing. When multiple columns are selected,
        the model will use all the images to generate a new image based on your prompt.
        
        For example, you could have one column with objects and another with backgrounds,
        then instruct the model to combine them.
        
        Note: This feature is only available for Gemini models.
        """,
        port_index=1,
        column_filter=util.image_column_filter,
    ).rule(
        knext.And(
            knext.DialogContextCondition(util.image_table_present),
            knext.OneOf(model, [ImageModels.GEMINI.name])
        ),
        knext.Effect.SHOW,
    )


    imagen_settings = ImagenSettings().rule(
        knext.OneOf(model, [ImageModels.IMAGEN.name]),
        knext.Effect.SHOW,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: GoogleAiStudioAuthenticationPortObjectSpec,
        image_table_spec: Optional[knext.Schema],
    ):
        authentication.validate_context(ctx)

        if not self.prompt:
            raise knext.InvalidParametersError(
                "The image generation prompt cannot be empty."
            )

        # Validate model selection matches model type
        is_imagen = self.model == ImageModels.IMAGEN.name
        
        # Check prompt length based on model type (use actual model name for detection)
        model_type = "imagen" if is_imagen else "gemini"
        prompt_size_limit = self.prompt_size_limits[model_type]
        
        # TODO use count tokens API to get the correct count instead of counting characters
        if len(self.prompt) > prompt_size_limit:
            raise knext.InvalidParametersError(
                f"Prompt cannot exceed a length of {prompt_size_limit} characters. Prompt length is {len(self.prompt)}."
            )

        # Check if image editing is requested but not supported
        has_image_table = image_table_spec is not None

        if has_image_table and not is_imagen and (self.image_columns is None):
            raise knext.InvalidParametersError(
                "Please select at least one image column for editing."
            )

        return knext.ImagePortObjectSpec(knext.ImageFormat.PNG)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        authentication: GoogleAiStudioAuthenticationPortObject,
        image_table: Optional[knext.Table],
    ):
        input_images = []
        is_imagen = self.model == ImageModels.IMAGEN.name
        
        if not is_imagen and image_table is not None and self.image_columns is not None:
            input_images = util.prepare_images(image_table, self.image_columns)
            
        img_bytes = self._generate_image(ctx, authentication.spec, input_images)

        return img_bytes, knext.view_png(img_bytes)

    def _generate_image(
        self,
        ctx: knext.ExecutionContext,
        authentication_spec: GoogleAiStudioAuthenticationPortObjectSpec,
        input_images: List[Tuple[str, bytes, str]],
    ) -> bytes:
        from google.genai import Client

        api_key = ctx.get_credentials(authentication_spec.credentials).password
        client = Client(api_key=api_key)

        try:
            if self.model == ImageModels.IMAGEN.name:
                return self._generate_with_imagen(client)
            else:
                return self._generate_with_gemini(client, input_images)
        except Exception as e:
            # If the error is a RuntimeError, raise it directly without wrapping it inside another RuntimeError.
            if isinstance(e, RuntimeError):
                raise
            error_message = str(e)
            if "get_media" in error_message.lower():
                raise RuntimeError(
                    "The model couldn't generate an image based on your prompt. "
                    "Please try with a more descriptive and detailed prompt, "
                    "or check if your prompt complies with Google's policy guidelines."
                )
            raise RuntimeError(f"Failed to generate image: {error_message}")

    def _generate_with_imagen(self, client) -> bytes:
        """Generate an image using the Imagen API."""
        from google.genai import types

        config = types.GenerateImagesConfig(
            aspect_ratio=ImagenSettings.AspectRatioOptions.to_api_value(
                self.imagen_settings.aspect_ratio
            ),
            person_generation=ImagenSettings.PersonGenerationOptions.to_api_value(
                self.imagen_settings.person_generation
            ),
            number_of_images=1,
            include_rai_reason=True, # include RAI (Responsible AI) reasons if the content was blocked.
        )

        response = client.models.generate_images(
            model=self.imagen_model_name,
            prompt=self.prompt,
            config=config,
        )

        if response and response.generated_images:
            for generated_image in response.generated_images:
                if generated_image.rai_filtered_reason:
                    # rai_filtered_reason is already a human-readable message from the API.
                    raise RuntimeError(
                        f"Image generation was blocked. {generated_image.rai_filtered_reason} "
                        f"For more information, see: {_POLICY_GUIDELINES_URL}."
                    )
                if generated_image.image:
                    return generated_image.image.image_bytes

        self._raise_no_image_error()

    def _generate_with_gemini(
        self, client, input_images: List[Tuple[str, bytes, str]]
    ) -> bytes:
        """Generate an image using the Gemini API."""
        contents = [self.prompt]

        if input_images:
            for _, image_bytes, _ in input_images:
                # generate_content expects PIL Image that's why we convert the bytes to image object
                pil_image = Image.open(io.BytesIO(image_bytes))
                contents.append(pil_image)

        response = client.models.generate_content(
            model=self.gemini_model_name,
            contents=contents,
        )

        # Check for blocking at the prompt level.
        if response and response.prompt_feedback and response.prompt_feedback.block_reason:
            description = _get_blocking_reason_description(response.prompt_feedback.block_reason)
            raise RuntimeError(f"Image generation was blocked. {description}")

        if not response or not response.candidates:
            self._raise_no_image_error()

        # Check all candidates for non-standard finish reasons.
        for candidate in response.candidates:
            if candidate.finish_reason and candidate.finish_reason != "STOP":
                description = _get_blocking_reason_description(candidate.finish_reason)
                raise RuntimeError(f"Image generation was blocked. {description}")

        # Extract parts from the first candidate with content.
        parts = None
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                parts = candidate.content.parts
                break

        image_bytes = self._extract_image_from_parts(parts)
        if image_bytes:
            return image_bytes

        # No image found - check if model provided text explanation.
        refusal_text = self._extract_text_from_parts(parts)
        if refusal_text:
            raise RuntimeError(f"Model refused to generate image. Reason: {refusal_text}")

        self._raise_no_image_error()

    def _extract_image_from_parts(self, parts: List) -> Optional[bytes]:
        """Extract image bytes from Gemini candidate parts."""
        if not parts:
            return None

        for part in parts:
            if part.inline_data is not None:
                image_pil = part.as_image()
                if image_pil.mime_type.startswith("image/png"):
                    return image_pil.image_bytes
                return self._to_png_bytes(image_pil.image_bytes)
        return None

    def _extract_text_from_parts(self, parts: List) -> Optional[str]:
        """Extract text from Gemini candidate parts."""
        if not parts:
            return None

        text_parts = []
        for part in parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        return " ".join(text_parts) if text_parts else None

    def _raise_no_image_error(self):
        """Raise a standardized error when no image was generated."""
        raise RuntimeError(
            "No image was generated in the response. This may be due to copyright concerns, "
            "safety policies, or other content restrictions. Please try with a more descriptive "
            "prompt or adjust your request to comply with Google's policy guidelines."
        )

    def _to_png_bytes(self, raw_bytes: bytes) -> bytes:
        """
        Convert raw image bytes to PNG format. This is needed because
        Gemini 3 Pro Image may return other formats like JPEG and WEBP.
        """
        with Image.open(io.BytesIO(raw_bytes)) as img:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()