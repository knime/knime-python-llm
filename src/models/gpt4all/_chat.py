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
from ..base import (
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    OutputFormatOptions,
    LLMModelType,
    create_model_type_switch,
)

from ._llm import GPT4AllLLMPortObject, GPT4AllLLMPortObjectSpec
from ._utils import gpt4all_icon, gpt4all_category
from ._utils import is_valid_model
from ._base import (
    GPT4AllInputSettings,
    GPT4AllPromptSettings,
    GPT4AllModelParameterSettings,
)


class GPT4AllChatModelPortObjectSpec(GPT4AllLLMPortObjectSpec, ChatModelPortObjectSpec):
    def __init__(
        self,
        llm_spec: GPT4AllLLMPortObjectSpec,
        system_prompt_template: str,
        prompt_template: str,
        model_type: LLMModelType,
    ) -> None:
        super().__init__(
            local_path=llm_spec._local_path,
            n_threads=llm_spec._n_threads,
            temperature=llm_spec._temperature,
            top_k=llm_spec._top_k,
            top_p=llm_spec._top_p,
            max_token=llm_spec._max_token,
            n_ctx=llm_spec._n_ctx,
            prompt_batch_size=llm_spec.prompt_batch_size,
            device=llm_spec.device,
        )
        self._system_prompt_template = system_prompt_template
        self._prompt_template = prompt_template
        self._model_type = model_type

    @property
    def is_instruct_model(self):
        return self._model_type is LLMModelType.INSTRUCT

    @property
    def system_prompt_template(self) -> str:
        return self._system_prompt_template

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    def serialize(self) -> dict:
        data = super().serialize()
        data["system_prompt_template"] = self._system_prompt_template
        data["prompt_template"] = self._prompt_template
        data["model_type"] = self._model_type.name
        return data

    @classmethod
    def deserialize(cls, data):
        return cls(
            GPT4AllLLMPortObjectSpec.deserialize(data),
            system_prompt_template=data["system_prompt_template"],
            prompt_template=data["prompt_template"],
            model_type=LLMModelType[data.get("model_type", LLMModelType.CHAT.name)],
        )


class GPT4AllChatModelPortObject(GPT4AllLLMPortObject, ChatModelPortObject):
    @property
    def spec(self) -> GPT4AllChatModelPortObjectSpec:
        return super().spec

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        from .._adapter import LLMChatModelAdapter

        llm = super().create_model(ctx, output_format)
        system_prompt_template = self.spec.system_prompt_template
        prompt_template = self.spec.prompt_template
        if self.spec.is_instruct_model:
            return llm

        return LLMChatModelAdapter(
            llm=llm,
            system_prompt_template=system_prompt_template,
            prompt_template=prompt_template,
        )


gpt4all_chat_model_port_type = knext.port_type(
    "GPT4All Chat Model",
    GPT4AllChatModelPortObject,
    GPT4AllChatModelPortObjectSpec,
    id="org.knime.python.llm.models.gpt4all.GPT4AllChatModelPortObject",
)


@knext.node(
    "Local GPT4All LLM Selector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Local Large Language Model",
    ],
)
@knext.output_port(
    "GPT4All Large Language Model",
    "A GPT4All Large Language Model.",
    gpt4all_chat_model_port_type,
)
class GPT4AllChatModelConnector:
    """
    Select a local LLM available via GPT4All.

    This node allows you to connect to a local GPT4All LLM. To get started,
    you need to download a specific model either through the [GPT4All client](https://gpt4all.io)
    or by dowloading a GGUF model from [Hugging Face Hub](https://huggingface.co/).
    Once you have downloaded the model, specify its file path in the
    configuration dialog to use it.

    It is not necessary to install the GPT4All client to execute the node.

    It is recommended to use models (e.g. Llama 2) that have been fine-tuned for chat applications. For model specifications
    including prompt templates, see [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json).

    The currently supported models are based on GPT-J, LLaMA, MPT, Replit, Falcon and StarCoder.

    For more information and detailed instructions on downloading compatible models, please visit the
    [GPT4All GitHub repository](https://github.com/nomic-ai/gpt4all).

    **Note**: This node cannot be used on the KNIME Hub, as the models cannot be embedded into the workflow due to their large size.
    """

    settings = GPT4AllInputSettings()
    model_type = create_model_type_switch()
    templates = GPT4AllPromptSettings().rule(knext.OneOf(model_type, [LLMModelType.INSTRUCT.name]), knext.Effect.HIDE)
    params = GPT4AllModelParameterSettings()

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> GPT4AllChatModelPortObjectSpec:
        is_valid_model(self.settings.local_path)
        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> GPT4AllChatModelPortObject:
        return GPT4AllChatModelPortObject(self.create_spec())

    def create_spec(self) -> GPT4AllChatModelPortObjectSpec:
        n_threads = None if self.settings.n_threads == 0 else self.settings.n_threads

        llm_spec = GPT4AllLLMPortObjectSpec(
            local_path=self.settings.local_path,
            n_threads=n_threads,
            temperature=self.params.temperature,
            top_p=self.params.top_p,
            top_k=self.params.top_k,
            n_ctx=self.params.n_ctx,
            max_token=self.params.max_token,
            prompt_batch_size=self.params.prompt_batch_size,
            device=self.params.device,
        )

        return GPT4AllChatModelPortObjectSpec(
            llm_spec=llm_spec,
            system_prompt_template=self.templates.system_prompt_template,
            prompt_template=self.templates.prompt_template,
            model_type=LLMModelType[self.model_type],
        )
