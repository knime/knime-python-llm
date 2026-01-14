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

"""
Error handling logic for Agent Prompter node execution.
"""

import knime.extension as knext
from ._parameters import RecursionLimitMode, ErrorHandlingMode


class AgentPrompterErrorHandler:
    """
    Handles error conditions during agent execution based on configured modes.

    This class encapsulates the error handling logic for the Agent Prompter node,
    including handling of iteration limit errors and general exceptions based on
    configured error handling modes.
    """

    def __init__(
        self,
        conversation,
        recursion_limit: int,
        recursion_limit_handling: RecursionLimitMode,
        error_handling: ErrorHandlingMode,
        chat_model,
        recursion_limit_prompt: str,
    ):
        """
        Initialize the error handler.

        Args:
            conversation: The AgentPrompterConversation instance
            recursion_limit: Maximum number of agent iterations
            recursion_limit_handling: How to handle recursion limit errors
            error_handling: How to handle general errors
            chat_model: The chat model for generating final responses
            recursion_limit_prompt: Prompt to use when generating final response
        """
        self._conversation = conversation
        self._recursion_limit = recursion_limit
        self._recursion_limit_handling = recursion_limit_handling
        self._error_handling = error_handling
        self._chat_model = chat_model
        self._recursion_limit_prompt = recursion_limit_prompt

    def handle_error(self, exception: Exception) -> None:
        """
        Handle an exception based on its type and configured modes.

        Args:
            exception: The exception that was caught
        """
        from ._agent import IterationLimitError
        
        if isinstance(exception, IterationLimitError):
            self._handle_iteration_limit_error()
        else:
            self._handle_general_error(exception)

    def _handle_iteration_limit_error(self) -> None:
        """
        Handle an IterationLimitError based on the configured recursion limit mode.

        If mode is FINAL_RESPONSE, generates a final response using the chat model.
        Otherwise, creates an error message and delegates to error mode handling.
        """
        if self._recursion_limit_handling == RecursionLimitMode.FINAL_RESPONSE:
            self._generate_final_response()
        else:
            error_message = f"""Recursion limit of {self._recursion_limit} reached. 
                You can increase the limit by setting the `recursion_limit` parameter to a higher value."""
            self._handle_error_by_mode(error_message)

    def _handle_general_error(self, exception: Exception) -> None:
        """
        Handle a general exception based on the configured error handling mode.

        Args:
            exception: The exception that was caught
        """
        error_message = f"An error occurred while executing the agent: {exception}"
        self._handle_error_by_mode(error_message)

    def _handle_error_by_mode(self, error_message: str) -> None:
        """
        Handle an error based on the configured error handling mode.

        Args:
            error_message: The error message to handle

        Raises:
            knext.InvalidParametersError: If error_handling mode is FAIL
        """
        if self._error_handling == ErrorHandlingMode.FAIL:
            raise knext.InvalidParametersError(error_message)
        else:
            self._conversation.append_error(Exception(error_message))

    def _generate_final_response(self) -> None:
        """
        Generate a final response when the recursion limit is reached.

        This method appends a human message with the recursion limit prompt,
        invokes the chat model to generate a final response, and appends
        that response to the conversation.
        """
        import langchain_core.messages as lcm

        messages = self._conversation.get_messages()
        messages = messages + [lcm.HumanMessage(self._recursion_limit_prompt)]
        final_response = self._chat_model.invoke(messages)
        self._conversation.append_messages(final_response)
