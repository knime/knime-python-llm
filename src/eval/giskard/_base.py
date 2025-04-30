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
import util
from dataclasses import dataclass

tortoise_icon = "icons/giskard/tortoise_icon.png"

eval_category = knext.category(
    path=util.main_category,
    level_id="evaluation",
    name="Evaluation",
    description="",
    icon=tortoise_icon,
)


def _get_schema_from_workflow_spec(
    workflow_spec, return_input_schema: bool
) -> knext.Schema:
    if workflow_spec is None:
        raise knext.InvalidParametersError(
            "Workflow spec is not available. Execute predecessor nodes."
        )
    if return_input_schema:
        return next(iter(workflow_spec.inputs.values())).schema
    else:
        return next(iter(workflow_spec.outputs.values())).schema


def _get_workflow_schema(
    ctx: knext.DialogCreationContext, port: int, input: bool
) -> knext.Schema:
    return _get_schema_from_workflow_spec(ctx.get_input_specs()[port], input)


def _validate_prediction_workflow_spec(workflow_spec) -> None:
    if len(workflow_spec.inputs) != 1:
        raise knext.InvalidParametersError(
            "Prediction workflow must have exactly one input table."
        )

    if len(workflow_spec.outputs) != 1:
        raise knext.InvalidParametersError(
            "Prediction workflow must produce exactly one output table."
        )


def _pick_default_workflow_column(workflow_spec, input: bool) -> str:
    prediction_workflow_schema = _get_schema_from_workflow_spec(
        workflow_spec, return_input_schema=input
    )
    return util.pick_default_column(prediction_workflow_schema, knext.string())


@dataclass
class ScannerColumn:
    name: str
    knime_type: knext.KnimeType
    pd_type: type
