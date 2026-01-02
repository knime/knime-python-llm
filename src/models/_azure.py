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


from langchain_openai.chat_models.azure import AzureChatOpenAI

class _AzureChatOpenAI(AzureChatOpenAI):
    """
    AzureChatOpenAI subclass that provides custom error handling.
    """

    async def abatch(self, *args, **kwargs):
        try:
            return await super().abatch(*args, **kwargs)
        except Exception as e:
            self._raise_if_reasoning_model_error(e)
        
    def invoke(self, *args, **kwargs):
        try:
            return super().invoke(*args, **kwargs)
        except Exception as e:
            self._raise_if_reasoning_model_error(e)


    def _raise_if_reasoning_model_error(self, error):
        from openai import BadRequestError

        error_message = str(error)
        has_unsupported_param = any(
            f"Unsupported parameter: '{param}'" in error_message
            for param in ["max_tokens", "top_p"]
        )
        has_unsupported_value = (
            "Unsupported value" in error_message and "temperature" in error_message
        )

        if (
            isinstance(error, BadRequestError)
            and (has_unsupported_param or has_unsupported_value)
        ):
            raise ValueError(
                "The selected model is a reasoning model. "
                "Please enable the 'The model is a reasoning model' option in the Azure OpenAI LLM Selector, "
                "or choose a different model that is not a reasoning model."
            )
        raise error
