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

from ..base import GeneralSettings


@knext.parameter_group(label="Model Usage")
class GPT4AllInputSettings:
    local_path = knext.LocalPathParameter(
        label="Model path",
        description="""Path to the pre-trained GPT4All model file eg. my/path/model.gguf.
        You can find the folder through settings -> application in the GPT4All desktop application.""",
    )

    n_threads = knext.IntParameter(
        label="Thread Count",
        description="""Number of CPU threads used by GPT4All. If set to 0, the number of threads 
        is determined automatically. 
        """,
        default_value=0,
        min_value=0,
        is_advanced=True,
        since_version="5.2.0",
    )


class GPT4AllModelParameterSettings(GeneralSettings):
    max_token = knext.IntParameter(
        label="Maximum response length (token)",
        description="""
        The maximum number of tokens to generate.

        This value, plus the token count of your prompt, cannot exceed the model's context length.
        """,
        default_value=250,
        min_value=1,
    )

    n_ctx = knext.IntParameter(
        label="Context length",
        description="""
        The maximum number of tokens a model can process in a single input sequence.

        This value should be greater than the number of tokens in your prompt plus the maximum response length.
        """,
        default_value=2048,
        min_value=1,
        is_advanced=True,
        since_version="5.5.0",
    )

    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 1.0.

        Higher values will lead to less deterministic answers.

        Try 0.9 for more creative applications, and 0 for ones with a well-defined answer.
        It is generally recommended altering this, or Top-p, but not both.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
    )

    top_k = knext.IntParameter(
        label="Top-k sampling",
        description="""
        Set the "k" value to limit the vocabulary used during text generation. Smaller values (e.g., 10) restrict the choices 
        to the most probable words, while larger values (e.g., 50) allow for more variety.
        """,
        default_value=20,
        min_value=1,
        is_advanced=True,
    )

    prompt_batch_size = knext.IntParameter(
        label="Prompt batch size",
        description="""Amount of prompt tokens to process at once. 

                    **NOTE**: On CPU, higher values can speed up reading prompts but will also use more RAM.
                    On GPU, a batch size of 1 has outperformed other batch sizes in our experiments.""",
        default_value=128,
        min_value=1,
        is_advanced=True,
    )

    device = knext.StringParameter(
        label="Device",
        description="""The processing unit on which the GPT4All model will run. It can be set to:

        - "cpu": Model will run on the central processing unit.
        - "gpu": Model will run on the best available graphics processing unit, irrespective of its vendor.
        - "amd", "nvidia", "intel": Model will run on the best available GPU from the specified vendor. 
        
        Alternatively, a specific GPU name can also be provided, and the model will run on the GPU that matches the name if it's available. 
        Default is "cpu".

        **Note**: If a selected GPU device does not have sufficient RAM to accommodate the model, an error will be thrown.
        It's advised to ensure the device has enough memory before initiating the model.""",
        default_value="cpu",
        is_advanced=True,
    )


@knext.parameter_group(label="Prompt Templates")
class GPT4AllPromptSettings:
    system_prompt_template = knext.MultilineStringParameter(
        "System prompt template",
        """ Model specific system template. Defaults to "%1". Refer to the 
        [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json) 
        for the correct template for your model:

        1. Locate the model you are using under the field "name". 
        2. Within the Model object, locate the "systemPrompt" field and use these values.""",
        default_value="%1",
    )

    prompt_template = knext.MultilineStringParameter(
        "Prompt template",
        """ Model specific prompt template. Defaults to "%1". Refer to the 
        [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json) 
        for the correct template for your model:

        1. Locate the model you are using under the field "name". 
        2. Within the Model object, locate the "promptTemplate" field and use these values.

        Note: For instruction based models, it is recommended to use "[INST] %1 [/INST]" as the 
        prompt template for better output if the "promptTemplate" field is not specified in the model list.""",
        default_value="""### Human:
%1
### Assistant:
""",
    )
