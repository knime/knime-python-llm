/*
 * ------------------------------------------------------------------------
 *
 *  Copyright by KNIME AG, Zurich, Switzerland
 *  Website: http://www.knime.com; Email: contact@knime.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License, Version 3, as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses>.
 *
 *  Additional permission under GNU GPL version 3 section 7:
 *
 *  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
 *  Hence, KNIME and ECLIPSE are both independent programs and are not
 *  derived from each other. Should, however, the interpretation of the
 *  GNU GPL Version 3 ("License") under any applicable laws result in
 *  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
 *  you the additional permission to use and propagate KNIME together with
 *  ECLIPSE with only the license terms in place for ECLIPSE applying to
 *  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
 *  license terms of ECLIPSE themselves allow for the respective use and
 *  propagation of ECLIPSE together with KNIME.
 *
 *  Additional permission relating to nodes for KNIME that extend the Node
 *  Extension (and in particular that are based on subclasses of NodeModel,
 *  NodeDialog, and NodeView) and that only interoperate with KNIME through
 *  standard APIs ("Nodes"):
 *  Nodes are deemed to be separate and independent programs and to not be
 *  covered works.  Notwithstanding anything to the contrary in the
 *  License, the License does not apply to Nodes, you are not required to
 *  license Nodes under the License, and you are granted a license to
 *  prepare and propagate Nodes, in each case even if such Nodes are
 *  propagated with or for interoperation with KNIME.  The owner of a Node
 *  may freely choose the license terms applicable to such Node, including
 *  when such Node is propagated with or for interoperation with KNIME.
 * ---------------------------------------------------------------------
 *
 * History
 *   May 21, 2025 (hornm): created
 */
package org.knime.ai.core.node.tool.workflow2tool;

import java.util.List;

import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataTableSpec;
import org.knime.filehandling.core.data.location.FSLocationValue;
import org.knime.node.parameters.NodeParameters;
import org.knime.node.parameters.NodeParametersInput;
import org.knime.node.parameters.Widget;
import org.knime.node.parameters.updates.Effect;
import org.knime.node.parameters.updates.Effect.EffectType;
import org.knime.node.parameters.updates.EffectPredicate;
import org.knime.node.parameters.updates.EffectPredicateProvider;
import org.knime.node.parameters.updates.ParameterReference;
import org.knime.node.parameters.updates.ValueReference;
import org.knime.node.parameters.widget.choices.ChoicesProvider;
import org.knime.node.parameters.widget.choices.ColumnChoicesProvider;
import org.knime.node.parameters.widget.choices.Label;
import org.knime.node.parameters.widget.choices.ValueSwitchWidget;

/**
 * @author Martin Horn, KNIME GmbH, Konstanz, Germany
 */
final class WorkflowToToolNodeSettings implements NodeParameters {

    @Widget(title = "Workflow path column",
        description = "The column containing the paths to read the workflows from.")
    @ChoicesProvider(PathColumnChoice.class)
    String m_inputColumnName;

    @Widget(title = "Output column",
        description = "Whether to append the output column or replace original path column.")
    @ValueSwitchWidget
    @ValueReference(OutputColumnPolicyRef.class)
    OutputColumnPolicy m_outputColumnPolicy = OutputColumnPolicy.APPEND;

    @Widget(title = "Output column name", description = "The name of the new column containing the tools.",
        effect = @Effect(predicate = AppendColumn.class, type = EffectType.ENABLE))
    String m_outputColumnName = "Tool";

    enum OutputColumnPolicy {
            @Label("Append")
            APPEND, //
            @Label("Replace")
            REPLACE, //
            @Label("Replace and rename")
            REPLACE_AND_RENAME
    }

    static class OutputColumnPolicyRef implements ParameterReference<OutputColumnPolicy> {
    }

    static final class AppendColumn implements EffectPredicateProvider {

        @Override
        public EffectPredicate init(final PredicateInitializer i) {
            return i.getEnum(OutputColumnPolicyRef.class).isOneOf(OutputColumnPolicy.APPEND,
                OutputColumnPolicy.REPLACE_AND_RENAME);
        }

    }

    static class PathColumnChoice implements ColumnChoicesProvider {

        @Override
        public List<DataColumnSpec> columnChoices(final NodeParametersInput context) {
            var portTypes = context.getInPortTypes();
            for (int i = 0; i < portTypes.length; i++) {
                if (DataTableSpec.class.isAssignableFrom(portTypes[i].getPortObjectSpecClass())) {
                    return context.getInTableSpec(i)
                        .map(tableSpec -> tableSpec.stream()
                            .filter(colSpec -> colSpec.getType().isCompatible(FSLocationValue.class)).toList())
                        .orElse(List.of());
                }
            }
            throw new IllegalStateException("No input table found");
        }

    }

}
