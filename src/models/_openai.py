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

from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr
import logging

_logger = logging.getLogger(__name__)


class _ChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI subclass that sets a warning if the response content is empty due to reasoning tokens.
    """

    _ctx = PrivateAttr(default=None)

    def __init__(self, *, ctx=None, **kwargs):
        super().__init__(**kwargs)
        self._ctx = ctx

    def _warn_if_reasoning_exhausted(self, msg) -> None:
        def get_reasoning_tokens(msg):
            metadata = getattr(msg, "usage_metadata", {}) or {}
            output_details = metadata.get("output_token_details") or {}
            return output_details.get("reasoning", 0)

        if (
            getattr(msg, "content", None) == ""
            and not getattr(msg, "tool_calls", None)
            and get_reasoning_tokens(msg) == self.max_tokens
        ):
            warn_msg = (
                "The model generated an empty response because it used all response tokens for its internal "
                "reasoning. You can increase the maximum response length in the OpenAI LLM Selector."
            )
            if self._ctx and hasattr(self._ctx, "set_warning"):
                self._ctx.set_warning(warn_msg)
            else:
                # TODO update once warning can be set in Agent Chat View node
                _logger.warning(warn_msg)

    def invoke(
        self,
        input,
        config=None,
        **kwargs,
    ):
        out = super().invoke(input, config=config, **kwargs)
        self._warn_if_reasoning_exhausted(out)
        return out

    async def ainvoke(
        self,
        input,
        config=None,
        **kwargs,
    ):
        out = await super().ainvoke(input, config=config, **kwargs)
        self._warn_if_reasoning_exhausted(out)
        return out
