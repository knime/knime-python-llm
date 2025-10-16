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
from dataclasses import dataclass
from ..base import GeneralRemoteSettings

_default_ibm_api_base = "https://eu-de.ml.cloud.ibm.com"


@dataclass
class ProjectOrSpace:
    """
    Project or space name and type.
    """

    name: str
    type: str

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            name=data["name"],
            type=data["type"],
        )


class ProjectOrSpaceSelection(knext.EnumParameterOptions):
    PROJECT = (
        "Project",
        "watsonx.ai Studio project to run the selected model.",
    )
    SPACE = (
        "Space",
        "watsonx.ai Studio space to run the selected model.",
    )


@knext.parameter_group(label="Connection configuration")
class IBMwatsonxConnectionSettings:
    project_or_space = knext.EnumParameter(
        "Project or space",
        """
        To provide context for chat completion or to use an embedding model, you must select a project or space.
        You can manage your projects and spaces in [watsonx.ai Studio](https://dataplatform.cloud.ibm.com/wx/home).
        """,
        default_value=ProjectOrSpaceSelection.PROJECT.name,
        enum=ProjectOrSpaceSelection,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    project_name = knext.StringParameter(
        "Project name",
        "Watson Studio project name to run the selected model.",
        default_value="",
    ).rule(
        knext.OneOf(project_or_space, [ProjectOrSpaceSelection.PROJECT.name]),
        knext.Effect.SHOW,
    )

    space_name = knext.StringParameter(
        "Space name",
        "Watson Studio space name to run the selected model",
        default_value="",
    ).rule(
        knext.OneOf(project_or_space, [ProjectOrSpaceSelection.SPACE.name]),
        knext.Effect.SHOW,
    )

    base_url = knext.StringParameter(
        "Base URL",
        """
        The base URL for the IBM watsonx.ai API.
        Refer to the [IBM watsonx.ai API documentation](https://cloud.ibm.com/apidocs/watsonx-ai#endpoint-url)
        to view the list of available base URLs.
        """,
        default_value=_default_ibm_api_base,
        is_advanced=True,
    )

    validate_api_key = knext.BoolParameter(
        "Validate API key",
        "If set, the API key is validated during execution by fetching the available models.",
        True,
        is_advanced=True,
    )


class IBMwatsonxChatModelSettings(GeneralRemoteSettings):
    temperature = knext.DoubleParameter(
        label="Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 2.0. 
        Higher values will make the output more random, 
        while lower values will make it more focused and deterministic.

        It is generally recommended altering this or top_p but not both.
        """,
        default_value=0.2,
        min_value=0.0,
        max_value=2.0,
    )

    max_tokens = knext.IntParameter(
        label="Maximum response length (token)",
        description="""
        The maximum number of tokens to generate.

        This value, plus the token count of your prompt, cannot exceed the model's context length.
        """,
        default_value=2048,
        min_value=1,
    )

    top_p = knext.DoubleParameter(
        label="Top-p sampling",
        description="""
        An alternative to sampling with temperature, 
        where the model considers the results of the tokens (words) 
        with top_p probability mass. Hence, 0.1 means only the tokens 
        comprising the top 10% probability mass are considered.
        """,
        default_value=0.15,
        min_value=0.01,
        max_value=1.0,
        is_advanced=True,
    )


def list_chat_models(ctx: knext.ConfigurationContext):
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_chat_model_list(ctx)
    return []


def list_embedding_models(ctx: knext.ConfigurationContext):
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_embedding_model_list(ctx)
    return []


def check_model(model_id: str) -> None:
    if not model_id:
        raise knext.InvalidParametersError("Select a model.")


def get_project_or_space(
    connection_settings: IBMwatsonxConnectionSettings,
) -> ProjectOrSpace:
    """
    Get the project or space name and type.
    """

    type = connection_settings.project_or_space

    type_to_name_mapping = {
        ProjectOrSpaceSelection.PROJECT.name: connection_settings.project_name,
        ProjectOrSpaceSelection.SPACE.name: connection_settings.space_name,
    }

    name = type_to_name_mapping.get(type)

    return ProjectOrSpace(
        name=name,
        type=type,
    )
