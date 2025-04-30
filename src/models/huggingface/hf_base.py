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
from typing import Optional

import knime.extension as knext
from ..base import (
    model_category,
    GeneralRemoteSettings,
)


hf_icon = "icons/huggingface/huggingface.png"
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
        "System prompt template",
        """Model specific system prompt template. Defaults to "%1".
        Refer to the Hugging Face Hub model card for information on the correct prompt template.""",
        default_value="%1",
    )

    prompt_template = knext.MultilineStringParameter(
        "Prompt template",
        """Model specific prompt template. Defaults to "%1". 
        Refer to the Hugging Face Hub model card for information on the correct prompt template.""",
        default_value="%1",
    )


def raise_for(exception: Exception, default: Optional[Exception] = None):
    import requests

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
