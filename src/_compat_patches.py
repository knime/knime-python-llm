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

def apply_anthropic_pydantic_patch():
    """
    Monkey-patch for pydantic-core 2.27.x compatibility issue with anthropic SDK.

    The bug causes `by_alias` to be None instead of a boolean in model_dump(),
    which raises: TypeError: argument 'by_alias': 'NoneType' object cannot be
    converted to 'PyBool'

    This patch intercepts the anthropic SDK's model_dump wrapper and ensures
    None values are converted to proper defaults before being passed to pydantic.

    We need to patch in multiple places because the SDK uses `from ._compat import model_dump`
    which binds the function as a local name at import time.

    TODO: Remove this patch once the fix is merged and released.
    See: https://github.com/anthropics/anthropic-sdk-python/issues/1160
    """

    try:
        import anthropic._compat as compat

        _original_model_dump = compat.model_dump

        def _patched_model_dump(
            model,
            *,
            exclude=None,
            exclude_unset=False,
            exclude_defaults=False,
            warnings=True,
            mode="python",
            by_alias=None,
        ):
            # Fix None values that cause TypeError in pydantic-core 2.27.x
            return _original_model_dump(
                model,
                exclude=exclude,
                exclude_unset=exclude_unset if exclude_unset is not None else False,
                exclude_defaults=exclude_defaults
                if exclude_defaults is not None
                else False,
                warnings=warnings if warnings is not None else True,
                mode=mode if mode is not None else "python",
                by_alias=by_alias if by_alias is not None else False,
            )

        # Patch in the _compat module
        compat.model_dump = _patched_model_dump

        # Also patch in _base_client where it's imported as a local name
        try:
            import anthropic._base_client as base_client

            base_client.model_dump = _patched_model_dump
        except (ImportError, AttributeError):
            pass

    except (ImportError, AttributeError):
        # anthropic not installed or API changed, skip patching
        pass


# Apply patches on module import
apply_anthropic_pydantic_patch()
