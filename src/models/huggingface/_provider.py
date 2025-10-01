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

import knime_extension as knext

HF_DEFAULT_INFERENCE_PROVIDER = "hf-inference"
HF_LEGACY_INFERENCE_PROVIDER = "hf-inference"  # for port objects created before 5.5


class ProviderSelectionOptions(knext.EnumParameterOptions):
    AUTO = (
        "Auto",
        "The first provider that supports the model is selected automatically.",
    )
    MANUAL = (
        "Manual",
        "Allows to select a specific provider from a list of all providers.",
    )


def create_provider_selection_parameter() -> knext.EnumParameter:
    return knext.EnumParameter(
        "Provider selection",
        "Specify whether the [Inference Provider](https://huggingface.co/docs/inference-providers/en/index) "
        "is selected automatically or manually.",
        ProviderSelectionOptions.AUTO.name,
        ProviderSelectionOptions,
        since_version="5.8.0",
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )


def create_provider_parameter(task: str) -> knext.StringParameter:
    from huggingface_hub.inference._providers import PROVIDERS

    return knext.StringParameter(
        "Inference provider",
        description="The [Inference Provider](https://huggingface.co/docs/inference-providers/en/index) that "
        "runs the model. The HF Hub website shows for each model which providers are available.",
        default_value=HF_DEFAULT_INFERENCE_PROVIDER,
        choices=lambda ctx: [
            provider
            for provider in PROVIDERS.keys()
            if task in PROVIDERS[provider]
            and provider != "openai"  # openai is not listed on HF Hub
        ],
        since_version="5.8.0",
    )


def validate_provider_model_pair(model: str, provider: str, task: str):
    from huggingface_hub.inference._providers import PROVIDERS

    provider_mapping = _get_provider_mapping(model, task)
    supported_providers = [
        x
        for x in provider_mapping
        if x.provider
        in [provider for provider in PROVIDERS.keys() if task in PROVIDERS[provider]]
    ]

    _raise_error_if_no_providers(provider_mapping, supported_providers, task)

    matching = [x for x in supported_providers if x.provider == provider]
    if not matching:
        if len(supported_providers) < 4:
            raise knext.InvalidParametersError(
                f"The model is not supported by provider '{provider}'. Available providers are: "
                f"{', '.join([x.provider for x in supported_providers])}."
            )
        else:
            raise knext.InvalidParametersError(
                f"The model is not supported by provider '{provider}'. Available providers include: "
                f"{', '.join([x.provider for x in supported_providers][:3])}. The complete list of "
                "providers for the model can be found on the HF Hub website."
            )


def select_first_provider(model: str, task: str) -> str:
    """Returns the first provider that is available for the specified model and task.
    Raises an error if the model is not supported by any providers for the task."""
    from huggingface_hub.inference._providers import PROVIDERS

    provider_mapping = _get_provider_mapping(model, task)
    supported_providers = [
        x
        for x in provider_mapping
        if x.provider
        in [provider for provider in PROVIDERS.keys() if task in PROVIDERS[provider]]
    ]
    _raise_error_if_no_providers(provider_mapping, supported_providers, task)
    return next(iter(supported_providers)).provider


def _get_provider_mapping(model: str, task: str):
    from ._session import huggingface_hub

    model_info_data = huggingface_hub.model_info(
        repo_id=model, expand="inferenceProviderMapping"
    )
    return [
        x
        for x in getattr(model_info_data, "inference_provider_mapping", [])
        if x.status == "live" and x.task == task
    ]


def _raise_error_if_no_providers(
    provider_mapping, supported_providers, task: str
) -> None:
    if not supported_providers:
        if provider_mapping:
            raise knext.InvalidParametersError(
                "The model is only available for recently added providers that are currently not supported."
            )
        raise knext.InvalidParametersError(
            f"The model is not supported by any provider for the task {task}."
        )
